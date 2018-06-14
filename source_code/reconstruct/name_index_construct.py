import csv
import jieba

jieba.load_userdict("..\\reconstruct_data\\medical_name_dict.txt")


# 构建诊断名称和序号的排列
token_count_map = {}
with open("..\\reconstruct_data\\reconstruct_data_unstructured.csv", 'r',
          encoding="utf-8-sig", newline="") as source_data:
    csv_reader = csv.reader(source_data)
    for line in csv_reader:
        diagnosis = line[4]
        diagnosis_list = jieba.cut(diagnosis)
        for item in diagnosis_list:
            if token_count_map.__contains__(item):
                token_count_map[item] += 1
            else:
                token_count_map[item] = 1
sorted_token_list = sorted(token_count_map.items(), key=lambda x: x[1],
                           reverse=True)
token_index_map = {}
index = 1
for item in sorted_token_list:
    item_name = item[0]
    if len(item_name) > 1:
        token_index_map[item_name] = index
        index += 1
print(token_index_map)

with open("..\\reconstruct_data\\token_index.csv", 'w',
          encoding="utf-8-sig", newline="") as token_file:
    csv_writer = csv.writer(token_file)
    csv_writer.writerows([["item_name", "index", "count"]])
    content = []
    for item in token_index_map:
        content.append([item, str(token_index_map[item]), str(token_count_map[
                                                                 item])])
    csv_writer.writerows(content)
