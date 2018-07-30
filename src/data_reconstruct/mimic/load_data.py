# coding=utf-8
import csv
import os

"""
'ADMISSIONS' finish
'CALLOUT' not need
'PROCEDURES_ICD'
'PROCEDUREEVENTS_MV'
'PRESCRIPTIONS' not need
'PATIENTS' finish
'OUTPUTEVENTS' not need
'NOTEEVENTS' not need
'MICROBIOLOGYEVENTS' not need
'LABEVENTS' not needed
'INPUTEVENTS_MV' not need
'INPUTEVENTS_CV' not need
'ICUSTAYS' not need
'DRGCODES' not need
'DIAGNOSES_ICD' finish
'DATETIMEEVENTS' not need
'CPTEVENTS' finish
'CHARTEVENTS'
"""


def read_admissions(root, file_name):
    admission_map = {}
    with open(root + file_name + ".csv", 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            patient_id, visit_id, admit_time, discharge_time, death_time = line[1], line[2], line[3], line[4], line[5]
            single_admission_map = {'visit_id': visit_id, 'admit_time': admit_time, 'discharge_time': discharge_time,
                                    'death_time': death_time}
            if not admission_map.__contains__(patient_id):
                admission_map[patient_id] = []

            admission_map[patient_id].append(single_admission_map)
    for key in admission_map:
        admission_list = admission_map[key]
        sorted(admission_list, key=lambda x: x['admit_time'])

    return admission_map


def read_patients(root, file_name):
    patients_map = {}
    with open(root + file_name + '.csv', 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            patient_id, gender, birthday = line[1], line[2], line[3]
            patients_map[patient_id] = {'gender': gender, 'birthday': birthday}
    return patients_map


def read_diagnosis(root, file_name):
    diagnosis_map = {}
    with open(root + file_name + '.csv', 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            _, patient_id, visit_id, _, icd_9_code = line

            if not diagnosis_map.__contains__(patient_id):
                diagnosis_map[patient_id] = dict()
            single_diagnosis_map = diagnosis_map[patient_id]

            if not single_diagnosis_map.__contains__(visit_id):
                single_diagnosis_map[visit_id] = []
            single_diagnosis_map[visit_id].append(icd_9_code)
    return diagnosis_map


def read_procedures_icd(root, file_name):
    """
    visit 的序号可能是乱的
    :param root:
    :param file_name:
    :return:
    """
    cpt_map = {}
    with open(root + file_name + '.csv', 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            _, patient_id, visit_id, order, icd_9 = line

            if not cpt_map.__contains__(patient_id):
                cpt_map[patient_id] = dict()
            single_visit_map = cpt_map[patient_id]

            if not single_visit_map.__contains__(visit_id):
                single_visit_map[visit_id] = []
            single_event = {'seq_no': order, 'icd_9_code': icd_9}
            single_visit_map[visit_id].append(single_event)

    for patient_id in cpt_map:
        single_patient_map = cpt_map[patient_id]
        for visit_id in single_patient_map:
            visit_event_list = single_patient_map[visit_id]
            visit_event_list = sorted(visit_event_list, key=lambda x: x['seq_no'])
            single_patient_map[visit_id] = visit_event_list
        cpt_map[patient_id] = single_patient_map
    return cpt_map


def main():
    root = os.path.abspath('..\\..\\..') + '\\reconstruct_data\\mimic_3\\source_data\\'
    cpt_events = read_procedures_icd(root, 'cptevents')


if __name__ == "__main__":
    main()
