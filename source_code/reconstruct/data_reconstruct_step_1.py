import csv
import os

# 原始数据库中，无论是EMR还是结构化的Diagnosis表中都存在大量丢失数据
# 但是经由step 1和step 2整理出来的诊断，我们可以确定Patient id和Encounter id所
# 包含的所有数据必然有相应的诊断
# 因此要以Step 1和Step 2整理出来的结果，整理一个没有丢失的Patient ID和
# Encounter ID的对应表

folder_path = 'D:\\Ningxia\\resource\\first_page_record_xml\\'
folder_index = ['01\\', '02\\', '03\\', '04\\', '05\\', '06\\', '07\\',
                '08\\', '09\\', '10\\', '11\\', '12\\', '13\\', '14\\']

encounter_time_map = {}
index_count = 0
encounter_map = {}

with open('..\\resource\\patient_inhospital_record.csv', 'r',
          encoding='utf-8-sig',
          newline="") as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        patient_id = line[0]
        encounter_id = line[1]
        encounter_map[encounter_id] = patient_id

with open('..\\resource\\patient_diagnosis_table_record.csv', 'r', encoding='utf-8-sig',
          newline="") as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        patient_id = line[0]
        encounter_id = line[1]
        encounter_map[encounter_id] = patient_id

with open('..\\reconstruct_data\\encounter_patient_table.csv', 'w', encoding='utf-8-sig',
          newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["encounter_id", "patient_id"])
    for item in encounter_map:
        csv_writer.writerow([item, encounter_map[item]])
