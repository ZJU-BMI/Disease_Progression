# coding=utf-8
import numpy as np

x = np.load('20180815133741_110_x.npy')
t = np.load('20180815133741_110_t.npy')

# 平均每次入院事件数
count_list = []
for case in x:
    for visit in case:
        count = 0
        for event in visit:
            if event == 1:
                count += 1
        count_list.append(count)
count_sum = 0
for item in count_list:
    count_sum += item
print('average event per visit ' + str(count_sum / len(count_list)))
# 5.74

# 两次入院之间的平均时间长度差异
time_interval_list = []
for case in t:
    for i in range(len(case)):
        if i == 0:
            continue
        if case[i][0] != 0:
            time_interval = case[i][0] - case[i - 1][0]
            time_interval_list.append(time_interval)
time_interval_sum = 0
for item in time_interval_list:
    time_interval_sum += item
print('time interval between two visit ' + str(time_interval_sum / len(time_interval_list)))
# 441 day
