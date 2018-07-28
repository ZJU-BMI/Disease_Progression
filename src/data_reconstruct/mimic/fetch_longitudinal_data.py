# coding=utf-8
import csv
import os
from itertools import islice


def fetch_subject_id(root_path):
    longitudinal_name = set()
    visit_count = dict()
    with open(root_path + 'ADMISSIONS.csv', 'r', encoding='utf-8', newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            subject_id = line[1]
            if not visit_count.__contains__(subject_id):
                visit_count[subject_id] = 1
            else:
                visit_count[subject_id] += 1
        for key in visit_count:
            if visit_count[key] > 1.5:
                longitudinal_name.add(key)
    return longitudinal_name


def main():
    source_data_path = 'F:\\MIMIC\\MIMIC3\\'
    target_data_path = os.path.abspath('..\\..\\..') + '\\reconstruct_data\\mimic_3\\source_data\\'
    id_list = fetch_subject_id(source_data_path)
    name_list = ['ADMISSIONS', 'CALLOUT', 'PROCEDURES_ICD', 'PROCEDUREEVENTS_MV', 'PRESCRIPTIONS', 'PATIENTS',
                 'OUTPUTEVENTS', 'NOTEEVENTS', 'MICROBIOLOGYEVENTS', 'LABEVENTS', 'INPUTEVENTS_MV', 'INPUTEVENTS_CV',
                 'ICUSTAYS', 'DRGCODES', 'DIAGNOSES_ICD', 'DATETIMEEVENTS', 'CPTEVENTS', 'CHARTEVENTS']
    for item in name_list:
        read_and_write_data(source_data_path, item, id_list, target_data_path)


def read_and_write_data(source_data_path, file_name, id_list, target_data_path):
    read_file = open(source_data_path + file_name + '.csv', 'r', newline="")
    write_file = open(target_data_path + file_name + '.csv', 'w', newline="")
    csv_writer = csv.writer(write_file)

    fetch_data = []
    csv_reader = csv.reader(read_file)
    for line in islice(csv_reader, 0, None):
        if id_list.__contains__(line[1]):
            fetch_data.append(line)
        if len(fetch_data) > 100000:
            csv_writer.writerows(fetch_data)
            fetch_data = []
    csv_writer.writerows(fetch_data)

    read_file.close()
    write_file.close()
    print(file_name + ' process finish')
    return fetch_data


if __name__ == '__main__':
    main()
