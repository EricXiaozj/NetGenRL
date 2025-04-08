# %%
import os
from scapy.all import rdpcap, IP, TCP, UDP
import json
import ipaddress


# %%
def process_flows(pcap_file):
    flows = {}
    packets = rdpcap(pcap_file)
    
    for packet in packets:
        if IP in packet:
            src_ip = int(ipaddress.ip_address(packet[IP].src))
            dst_ip = int(ipaddress.ip_address(packet[IP].dst))
            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                protocol = 6
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
                protocol = 17
            else:
                continue

            pkt = {}
            pkt['time'] = float(packet.time)
            pkt['pkt_len'] = len(packet)
            pkt['flags'] = int(packet[TCP].flags) if TCP in packet else 0
            pkt['ttl'] = packet[IP].ttl
            
            flow_1 = (src_ip, src_port, dst_ip, dst_port, protocol)
            flow_2 = (dst_ip, dst_port, src_ip, src_port, protocol)  # reverse flow

            if flow_1 in flows.keys():
                pkt['time'] -= flows[flow_1]['last_time']
                pkt['time'] = round(pkt['time'],2)
                flows[flow_1]['last_time'] = float(packet.time)
                flows[flow_1]['series'].append(pkt)
                continue
            if flow_2 in flows.keys():
                pkt['time'] -= flows[flow_2]['last_time']
                pkt['time'] = round(pkt['time'],2)
                flows[flow_2]['last_time'] = float(packet.time)
                pkt['pkt_len'] = -pkt['pkt_len']
                flows[flow_2]['series'].append(pkt)
                continue

            flows[flow_1] = {}
            flows[flow_1]['last_time'] = float(packet.time)
            pkt['time'] = 0
            flows[flow_1]['series'] = []
            flows[flow_1]['series'].append(pkt)
            flows[flow_1]['src_ip'] = src_ip 
            flows[flow_1]['src_port'] = src_port
            flows[flow_1]['dst_ip'] = dst_ip
            flows[flow_1]['dst_port'] = dst_port
            flows[flow_1]['protocol'] = protocol
    
    return list(flows.values())

# %%
def save_flows_to_json(flows, output_file):
    with open(output_file, 'w') as file:
        json.dump(flows, file)  

# %%
def process_pcap(pcap_path, label_dict, json_path):
    
    for label_str in label_dict.keys():
        flows = []
        for filename in os.listdir(pcap_path):
            if filename.endswith('.pcap') or filename.endswith('.pcapng'):
                if label_str not in filename:
                    continue
                pcap_file = os.path.join(pcap_path, filename)
            
                flows += (process_flows(pcap_file))
            
        save_flows_to_json(flows, f'./{json_path}/{label_str}.json')
    
        print(f"Total Flows of {label_str}:", len(flows))

