from itertools import islice
import csv
import jieba

valid_index_set = set()
with open('..\\reconstruct_data\\valid_index.txt', 'r', encoding='utf-8-sig',
          newline="") as valid_index_file:
    for line in valid_index_file:
        valid_index_set.add(int(line))
print(valid_index_set)

name_index_map = {}
index_name_map = {}
with open("..\\reconstruct_data\\token_index.csv", 'r', encoding="utf-8-sig",
          newline="") as index_name_file:
    csv_reader = csv.reader(index_name_file)
    for line in islice(csv_reader, 1, None):
        name_index_map[line[0]] = int(line[1])
        index_name_map[int(line[1])] = line[0]

data_with_structured_diagnosis = []
with open("..\\reconstruct_data\\reconstruct_data_unstructured.csv", 'r',
          encoding="utf-8-sig",newline="") as unstructured_data:
    csv_reader = csv.reader(unstructured_data)
    jieba.load_userdict("..\\reconstruct_data\\medical_name_dict.txt")
    for line in islice(csv_reader, 1, None):
        pat_id, visit_id, encounter_id, admission_time, diagnosis, dept, \
         operation, death_record = line
        diagnosis_list = jieba.cut(diagnosis)
        structured_diagnosis_list = []
        for item in diagnosis_list:
            if len(item) > 1 and name_index_map.__contains__(item):
                index = name_index_map[item]
                if valid_index_set.__contains__(index):
                    structured_diagnosis_list.append(index)
        code_str = ""
        for item in structured_diagnosis_list:
            code_str += "$"+str(item)
        data_tuple = [pat_id, visit_id, encounter_id, admission_time,
                      diagnosis, dept, operation, death_record, code_str]
        data_with_structured_diagnosis.append(data_tuple)

with open("..\\reconstruct_data\\reconstruct_data_structured_diagnosis.csv",
          'w', encoding="utf-8-sig", newline="") as write_file:
    csv_writer = csv.writer(write_file)
    csv_writer.writerows([['patient_id', "visit_id'", "encounter_id",
                           "admission_time", "diagnosis", "department",
                           "operation", "death_record", "diagnosis_code_set"]])
    csv_writer.writerows(data_with_structured_diagnosis)
