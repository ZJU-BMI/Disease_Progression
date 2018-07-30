# coding=utf-8
import csv
import os
from itertools import islice
from xml.dom import minidom
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement

import load_data


def construct_xml(admissions, patients, diagnosis, cpt_events, diagnosis_dict, procedure_dict):
    # XML生成
    root = Element('Patient_List_Event')
    for patient_id in patients:
        patient_node = append_patient_node(root, patients[patient_id], patient_id)

        patient_visit = admissions[patient_id]
        patient_visit = visit_sort(patient_visit)
        for visit_id in range(0, len(patient_visit)):
            # 生成Visit_Node节点及其相关内容
            visit_node = append_visit_node(patient_node, patient_visit, visit_id)

            # 生成住院Diagnosis节点及其相关内容
            diagnosis_node = SubElement(visit_node, "diagnosis")
            if diagnosis.__contains__(patient_id) and diagnosis[patient_id].__contains__(patient_visit[visit_id][0]):
                diagnosis_node.set('contain_diagnosis', "true")
                diagnosis_info = diagnosis[patient_id][patient_visit[visit_id][0]]
                append_diagnosis_node(diagnosis_info, diagnosis_node, diagnosis_dict)
            else:
                diagnosis_node.set('contain_diagnosis', "false")

            # 生成CPT节点及其相关内容
            cpt_events_node = SubElement(visit_node, "procedures")
            if cpt_events.__contains__(patient_id) and cpt_events[patient_id].__contains__(patient_visit[visit_id][0]):
                cpt_events_node.set('contain_cpt', "true")
                cpt_info = cpt_events[patient_id][patient_visit[visit_id][0]]
                append_cpt_node(cpt_info, cpt_events_node, procedure_dict)
            else:
                cpt_events_node.set('contain_cpt', "false")

    return root


def append_patient_node(root, patient_info, patient_id):
    # Patient_Node 属性
    patient_node_attr = {'patient_id': str(patient_id), 'sex': patient_info['gender'], 'birthday': patient_info[
        'birthday']}
    patient_node = SubElement(root, 'patient_node', patient_node_attr)
    return patient_node


def append_visit_node(patient_node, visit_list, visit_id):
    single_visit = visit_list[visit_id]
    for i in range(0, len(single_visit)):
        if len(single_visit[i]) < 2:
            single_visit[i] = 'NoRecord'

    # define the first visit has visit_id = 1
    visit_node_attr = {'visit_id': str(visit_id + 1), 'visit_index': single_visit[0], 'admission_date': single_visit[1],
                       'discharge_date': single_visit[2], 'death_time': single_visit[3]}

    visit_node = SubElement(patient_node, "visit", visit_node_attr)
    return visit_node


def visit_sort(patient_visit_map):
    visit_list = []
    for single_visit in patient_visit_map:
        visit_convert = [single_visit['visit_id'], single_visit['admit_time'], single_visit['discharge_time'],
                         single_visit['death_time']]
        visit_list.append(visit_convert)
    visit_list = sorted(visit_list, key=lambda x: x[1])

    return visit_list


def append_diagnosis_node(visit_diagnosis, diagnosis_node, diagnosis_dict):
    for diagnosis_no in range(0, len(visit_diagnosis)):
        if diagnosis_dict.__contains__(visit_diagnosis[diagnosis_no]):
            disease_name = diagnosis_dict[visit_diagnosis[diagnosis_no]]
        else:
            disease_name = 'NoName'
        diagnosis_node_attr = {'diagnosis_no': str(diagnosis_no + 1), 'icd_9_code': visit_diagnosis[diagnosis_no],
                               'disease_name': disease_name,
                               'normalized_code': diagnosis_normalize(visit_diagnosis[diagnosis_no])}
        SubElement(diagnosis_node, "diagnosis_item", diagnosis_node_attr)


def read_diagnosis_code(root_path, file_name):
    diagnosis_code_map = dict()
    with open(root_path + file_name + '.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            icd_code, short_name = line[1], line[2]
            diagnosis_code_map[icd_code] = short_name
    return diagnosis_code_map


def read_procedure_code(root_path, file_name):
    procedure_code_map = dict()
    with open(root_path + file_name + '.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            icd_code, short_name = line[1], line[2]
            procedure_code_map[icd_code] = short_name
    return procedure_code_map


def append_cpt_node(visit_cpt, cpt_node, procedure_dict):
    cpt_list = []
    for cpt_map in visit_cpt:
        single_cpt = [cpt_map['seq_no'], cpt_map['icd_9_code']]
        cpt_list.append(single_cpt)
    cpt_list = sorted(cpt_list, key=lambda x: x[0])

    for cpt_no in range(0, len(cpt_list)):
        single_cpt_list = cpt_list[cpt_no]
        if procedure_dict.__contains__(single_cpt_list[1]):
            name = procedure_dict[single_cpt_list[1]]
        else:
            name = 'NoRecord'
        cpt_code_attr = {'procedure_no': str(cpt_no + 1), 'icd_9_code': single_cpt_list[1],
                         'procedure_name': name, 'normalized_code': procedure_normalize(single_cpt_list[1])}
        SubElement(cpt_node, "procedure_item", cpt_code_attr)


def prettify(element):
    # 此处只能用utf-8，不然会有错误
    rough_string = ElementTree.tostring(element, 'utf-8')
    recreate_xml = minidom.parseString(rough_string)
    return recreate_xml.toprettyxml(indent="  ")


# TODO 更为细致的清洗策略留待之后做
def diagnosis_normalize(icd_code):
    return icd_code[0:3]


# TODO 更为细致的清洗策略留待之后做
def procedure_normalize(procedure_code):
    return procedure_code[0:2]


def main():
    root = os.path.abspath('..\\..\\..') + '\\reconstruct_data\\mimic_3\\'
    source_root = root + 'source_data\\'
    save_path = root + "reconstruct\\"
    admissions = load_data.read_admissions(source_root, 'ADMISSIONS')
    patients = load_data.read_patients(source_root, 'PATIENTS')
    diagnosis = load_data.read_diagnosis(source_root, 'DIAGNOSES_ICD')
    cpt_events = load_data.read_procedures_icd(source_root, 'PROCEDURES_ICD')
    diagnosis_dict = read_diagnosis_code(source_root, 'D_ICD_DIAGNOSES')
    procedure_dict = read_procedure_code(source_root, 'D_ICD_PROCEDURES')

    xml = construct_xml(admissions, patients, diagnosis, cpt_events, diagnosis_dict, procedure_dict)
    xml_content = prettify(xml)
    with open(save_path + 'reconstructed.xml', 'w', encoding='utf-8-sig', newline="") as file:
        file.write(xml_content)


if __name__ == "__main__":
    main()
