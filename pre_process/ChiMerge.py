# %%
import numpy as np
import struct
import json
import scipy.stats as stats
import os


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
    chi_2 = empty_len * ps + np.where(ps != 0, ((num_ls / left_len - ps) ** 2 / ps * left_len + (num_rs / right_len - ps) ** 2 / ps * right_len), 0)
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

# # %%
# def interval_allocate(intervals,data):
#     pass

# %%
def construct_data_dic(json_data_dic, attribute_list, port_attr_list, ip_attr_list, series_attr_list, max_seq_len):
    data_dic = {attribute: {} for attribute in attribute_list}

    for label, json_data in json_data_dic.items():
        for attribute in attribute_list:
            data_dic[attribute][label] = {}
        for item in json_data:
            for ip_attr in ip_attr_list:
                if item[ip_attr] not in data_dic[ip_attr][label]:
                    data_dic[ip_attr][label][item[ip_attr]] = 0
                data_dic[ip_attr][label][item[ip_attr]] += 1
            for port_attr in port_attr_list:
                if item[port_attr] not in data_dic[port_attr][label]:
                    data_dic[port_attr][label][item[port_attr]] = 0
                data_dic[port_attr][label][item[port_attr]] += 1
            
            for i,pkt in enumerate(item['series']):
                if i >= max_seq_len:
                    break
                for sery_attr in series_attr_list:
                    if pkt[sery_attr] not in data_dic[sery_attr][label]:
                        data_dic[sery_attr][label][pkt[sery_attr]] = 0
                    data_dic[sery_attr][label][pkt[sery_attr]] += 1
                    

    return data_dic
    

# %%
def construct_intervals_data(data_dic, min_value, max_value, min_unit):

    labels = data_dic.keys()
    dics = list(data_dic.values())

    intervals = []
    data_list = []

    tmp_dict = {}

    id = 0
    for value in dics:
        for k, v in value.items():
            k = min(max(k,min_value),max_value)
            k = round(k / min_unit) * min_unit
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
    
    intervals, data = construct_intervals_data(data_dic, min_value, max_value, min_unit)
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
def chimerge(dataset, json_folder, bins_folder, ip_attrs, port_attrs, sery_attrs, max_seq_len, params_dic):
    
    json_data_dic = {}
    for filename in os.listdir(f'./{json_folder}/{dataset}'):
        with open(f'./{json_folder}/{dataset}/{filename}', 'r') as f:
            json_data = json.load(f)
            json_data_dic[filename.split('.')[0]] = json_data

    attr_list = sery_attrs + port_attrs + ip_attrs
    max_seq_len = 16

    data_dic = construct_data_dic(json_data_dic, attr_list, port_attrs, ip_attrs, sery_attrs, max_seq_len)

    np.set_printoptions(linewidth=2000)
    
    result_dic = {}
    
    # params_dic = {'time':{'min_unit':0.01,'min_value':0,'max_value':10000},
    #               'pkt_len':{'min_unit':1,'min_value':-1500,'max_value':1500},
    #               'flags':{'min_unit':1,'min_value':0,'max_value':255},
    #               'ttl':{'min_unit':1,'min_value':0,'max_value':255},
    #               'src_port':{'min_unit':1,'min_value':0,'max_value':65535},
    #               'dst_port':{'min_unit':1,'min_value':0,'max_value':65535},
    #               'src_ip':{'min_unit':1,'min_value':0,'max_value':4294967295},
    #               'dst_ip':{'min_unit':1,'min_value':0,'max_value':4294967295}}
                  

    for label in params_dic.keys():
        merged_intervals,merged_data = cal_bins(data_dic[label],params_dic[label]['min_unit'],params_dic[label]['min_value'],params_dic[label]['max_value'])
        print(label,':', len(merged_intervals))
        for i in range(len(merged_intervals)):
            merged_intervals[i] = [round(merged_intervals[i][0],2),round(merged_intervals[i][1],2)]
        # print(merged_intervals)
        # print(merged_data.astype(int))
        result_dic[label] = {'intervals':merged_intervals,'data':merged_data.tolist()}
    
    json_str = json.dumps(result_dic)
    with open(f'./{bins_folder}/bins_{dataset}.json', 'w') as file:
        file.write(json_str)

    