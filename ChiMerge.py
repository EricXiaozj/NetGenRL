# %%
import numpy as np
import struct
import json
import scipy.stats as stats

## TODO: 改为不关注序列顺序，所有序列位置的区间一致，以配合word2vec

# %%
def threshold_merge(intervals, data, min_unit, threshold = 1):
    new_intervals = []
    new_data = np.empty((data.shape[0],0))
    
    flag = False
    for i in range(len(intervals)):
        if flag and np.max(data[:,i]) <= threshold:
            new_intervals[-1][1] = intervals[i][1]
            new_data[:,-1] = new_data[:,-1] + data[:,i]
            continue
        if not flag and np.max(data[:,i]) <= threshold:
            new_intervals.append(intervals[i])
            if i > 0:
                new_intervals[-1][0] = new_intervals[-2][0] + min_unit
            new_data = np.hstack((new_data,data[:,i].reshape(-1, 1)))
            flag = True
            continue
        if flag and np.max(data[:,i]) > threshold:
            new_intervals[-1][1] = intervals[i][0] - min_unit
            new_intervals.append(intervals[i])
            new_data = np.hstack((new_data,data[:,i].reshape(-1, 1)))
            flag = False
            continue
        new_intervals.append(intervals[i])
        new_data = np.hstack((new_data,data[:,i].reshape(-1, 1)))
        # flag = (np.max(data[:,i]) <= threshold)
        flag = False

    return new_intervals,new_data

# %%
def chi_cal(interval_l,interval_r,num_ls,num_rs):
    left_board = interval_l[0]
    right_board = interval_r[1]
    empty_len = interval_r[0] - interval_l[1] - 1
    left_len = interval_l[1] - interval_l[0] + 1
    right_len = interval_r[1] - interval_r[0] + 1
    totals = num_ls + num_rs
    length = right_board - left_board + 1
    ps = totals/length
    # print(left_len,right_len,ps)
    chi_2 = empty_len * ps + np.where(ps != 0, ((num_ls / left_len - ps) ** 2 / ps + (num_rs / right_len - ps) ** 2 / ps), 0)
    return np.sum(chi_2)

# %%
def chi_merge(intervals, data, chi_threshold=3.84):
    m, _ = data.shape
    
    if len(intervals) == 0:
        return intervals,data

    chi2_vals = np.zeros(len(intervals) - 1)

    if len(chi2_vals) == 0:
        return intervals,data

    for i in range(len(intervals)-1):
        chi2_vals[i] = chi_cal(intervals[i],intervals[i+1],data[:,i],data[:,i+1])

    # print(chi2_vals)

    min_chi2_id = 0

    while True:
        if len(chi2_vals) == 0:
            return intervals,data
        # print(len(chi2_vals),len(intervals),data.shape)
        # 找到卡方值最小的两项（即最接近的相邻区间）
        min_chi2_id = np.argmin(chi2_vals)

        # print(min_chi2_id,chi2_vals[min_chi2_id])
        
        if chi2_vals[min_chi2_id] >= chi_threshold:
            break

        intervals[min_chi2_id][1] = intervals[min_chi2_id+1][1]
        intervals.pop(min_chi2_id+1)

        data[:,min_chi2_id] = data[:,min_chi2_id] + data[:,min_chi2_id+1]
        data = np.delete(data,min_chi2_id+1,axis=1)

        chi2_vals = np.delete(chi2_vals,[min_chi2_id])
        
        # if min_chi2_id == len(min_chi2_id) - 1:
        #     if min_chi2_id != 0:
        #         chi2_vals[min_chi2_id - 1] = chi_cal(intervals[min_chi2_id-1],intervals[min_chi2_id],data[:,min_chi2_id -1],data[:,min_chi2_id])
        #     continue
        
        if min_chi2_id != 0:
            chi2_vals[min_chi2_id - 1] = chi_cal(intervals[min_chi2_id-1],intervals[min_chi2_id],data[:,min_chi2_id -1],data[:,min_chi2_id])
        if min_chi2_id < len(chi2_vals):
            chi2_vals[min_chi2_id] = chi_cal(intervals[min_chi2_id],intervals[min_chi2_id+1],data[:,min_chi2_id],data[:,min_chi2_id+1])

        # print(intervals)
        
    # print(chi2_vals)
    return intervals,data

# %%
def interval_allocate(intervals,data):
    pass

# %%
def construct_data_dic(json_data):
    TOTAL_LEN = 114
    data_dic = {}
    data_dic = {'bytes':{},'packets':{},'port':{},'packet_len':{},'time':{}}
    # for _ in range(0,16):
    #     data_dic['packet_len'].append({})
    #     data_dic['time'].append({})
    
    for item in json_data:
        label_str = item['labels'][0]
        if label_str not in data_dic['bytes']:
            data_dic['bytes'][label_str] = {}
            data_dic['packets'][label_str] = {}
            data_dic['port'][label_str] = {}
            # for i in range(0,16):
            #     data_dic['packet_len'][i][label_str] = {}
            #     data_dic['time'][i][label_str] = {}
            data_dic['packet_len'][label_str] = {}
            data_dic['time'][label_str] = {}
            # data_dic[label_str] = {'bytes':{},'packets':{},'port':{},'packet_len':[],'time':[]}
            # for _ in range(0,16):
            #     data_dic[label_str]['packet_len'].append({})
            #     data_dic[label_str]['time'].append({})
        bytes_len = item['meta']['bytes']/10
        packets_len = item['meta']['packets']
        im = bytes.fromhex(item['nprint'])

        # if packets_len != 2:
        #     continue

        if bytes_len not in data_dic['bytes'][label_str]:
            data_dic['bytes'][label_str][bytes_len] = 0
        data_dic['bytes'][label_str][bytes_len] += 1
        if packets_len not in data_dic['packets'][label_str]:
            data_dic['packets'][label_str][packets_len] = 0
        data_dic['packets'][label_str][packets_len] += 1

        for i in range(min(16,packets_len)):
            line = im[i*TOTAL_LEN:i*TOTAL_LEN+TOTAL_LEN]
            time_h, time_l, pl = struct.unpack("IIh", line[:10])
            time_l //= 1e4
            time = time_h + time_l*0.01
            # time >>= 32
            # time = float(time)/float(1 << 32)
            # if pl not in data_dic['packet_len'][i][label_str]:
            #     data_dic['packet_len'][i][label_str][pl] = 0
            # data_dic['packet_len'][i][label_str][pl] += 1
            # if time not in data_dic['time'][i][label_str]:
            #     data_dic['time'][i][label_str][time] = 0
            # data_dic['time'][i][label_str][time] += 1
            if pl not in data_dic['packet_len'][label_str]:
                data_dic['packet_len'][label_str][pl] = 0
            data_dic['packet_len'][label_str][pl] += 1
            if time not in data_dic['time'][label_str]:
                data_dic['time'][label_str][time] = 0
            data_dic['time'][label_str][time] += 1

        line = im[0:TOTAL_LEN]
        tcp_dport = line[32:34]
        udp_dport = line[92:94]

        dport = bytearray(a | b for a, b in zip(tcp_dport, udp_dport))
        dport = int.from_bytes(dport, 'big')
        if dport not in data_dic['port'][label_str]:
            data_dic['port'][label_str][dport] = 0
        data_dic['port'][label_str][dport] += 1

    return data_dic
    

# %%
def construct_intervals_data(data_dic):

    labels = data_dic.keys()
    dics = list(data_dic.values())

    intervals = []
    data_list = []

    tmp_dict = {}

    id = 0
    for value in dics:
        for k, v in value.items():
            if k not in tmp_dict:
                tmp_dict[k] = [0 for _ in range(len(labels))]
            tmp_dict[k][id] = v
        id += 1

    tmp_lists = sorted(tmp_dict.items(), key=lambda x: x[0])

    for (iter, li) in tmp_lists:
        intervals.append([iter,iter])
        data_list.append(li)

    data = np.array(data_list)

    return intervals,data.transpose()

# %%

def fill_intervals(intervals, data, min_unit, min_value, max_value):
    now_value = min_value
    for i, interval in enumerate(intervals):
        if now_value + min_unit < interval[0]:
            intervals.insert(i,[now_value,interval[0] - min_unit])
            data = np.insert(data,i,0,axis=1)
        now_value = interval[1] + min_unit
    if now_value + min_unit < max_value:
        intervals.append([now_value, max_value])
        data = np.insert(data,len(intervals)-1,0,axis=1)
    return intervals,data

def cal_bins(data_dic, min_unit, min_value, max_value):
    
    intervals, data = construct_intervals_data(data_dic)
    # print(intervals)
    # print(data.astype(int))
    new_intervals,new_data = threshold_merge(intervals,data, min_unit)

    alpha = 0.01
    df = new_data.shape[0] - 1
    critical_value = stats.chi2.ppf(1 - alpha, df)

    merged_intervals,merged_data = chi_merge(new_intervals,new_data,chi_threshold=critical_value)
    
    # filled_intervals,filled_data = fill_intervals(merged_intervals,merged_data,min_unit,min_value,max_value)

    # return filled_intervals,filled_data
    return merged_intervals,merged_data

# %%
if __name__ == '__main__':
    with open('./data/vpn_data_small.json', 'r') as f:
        json_data = json.load(f)['data']

    data_dic = construct_data_dic(json_data)

    np.set_printoptions(linewidth=2000)
    
    result_dic = {}
    params_dic = {'bytes':{'min_unit':0.1,'min_value':0,'max_value':0},
                  'packets':{'min_unit':1,'min_value':1,'max_value':0},
                  'port':{'min_unit':1,'min_value':0,'max_value':65535},
                  'packet_len':{'min_unit':1,'min_value':-1500,'max_value':1500},
                  'time':{'min_unit':0.01,'min_value':0,'max_value':0}}

    for label in ['bytes','packets','port', 'packet_len', 'time']:
        merged_intervals,merged_data = cal_bins(data_dic[label],params_dic[label]['min_unit'],params_dic[label]['min_value'],params_dic[label]['max_value'])
        print(label,':')
        print(len(merged_intervals))
        print(merged_intervals)
        print(merged_data.astype(int))
        print()
        result_dic[label] = {'intervals':merged_intervals,'data':merged_data.tolist()}

    # for label in ['packet_len','time']:
    #     result_dic[label] = []
        # for i in range(0,16):
        #     merged_intervals,merged_data = cal_bins(data_dic[label][i])
        #     print(label,i,':')
        #     print(len(merged_intervals))
        #     print(merged_intervals)
        #     print(merged_data.astype(int))
        #     print()
        #     result_dic[label].append({'intervals':merged_intervals,'data':merged_data.tolist()})
    
    json_str = json.dumps(result_dic)
    with open('./bins/bins_small.json', 'w') as file:
        file.write(json_str)
    