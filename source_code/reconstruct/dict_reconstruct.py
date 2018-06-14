import jieba
import csv

jieba.load_userdict("..\\resource\\std_name_utf8.txt")

diagnosis_count_map = {}
diagnosis_compare = []
with open("..\\reconstruct_data\\reconstruct_data_unstructured.csv", 'r',
          encoding="utf-8-sig", newline="") as source_data:
    csv_reader = csv.reader(source_data)
    for line in csv_reader:
        diagnosis_compare_tuple = []
        diagnosis = line[4]
        diagnosis_compare_tuple.append(diagnosis)

        seg_list = jieba.cut(diagnosis)  # 默认是精确模式
        for item in seg_list:
            if len(item) >1:
                diagnosis_compare_tuple.append(item)
            if diagnosis_count_map.__contains__(item):
                diagnosis_count_map[item] += 1
            else:
                diagnosis_count_map[item] = 1
        diagnosis_compare.append(diagnosis_compare_tuple)

with open("..\\reconstruct_data\\diagnosis_token_stat.csv", 'w',
          encoding="utf-8-sig", newline="") as diagnosis_stat:
    csv_writer = csv.writer(diagnosis_stat)
    content = []
    for item in diagnosis_count_map:
        content.append([item, diagnosis_count_map[item]])
    csv_writer.writerows(content)

with open("..\\reconstruct_data\\diagnosis_token_compare.csv", 'w',
          encoding="utf-8-sig", newline="") as diagnosis_compare_file:
    csv_writer = csv.writer(diagnosis_compare_file)
    csv_writer.writerows(diagnosis_compare)

