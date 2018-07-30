# coding=utf-8
import random
from xml.dom import minidom
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement

from plagh import data_cleansing, load_data


def data_reconstruction(data_source):
    # XML生成
    root = Element('Patient_List_Event')

    append_meta_data(root, data_source)

    patient_info_map = data_source['patient_info_map']
    patient_visit_map = data_source['patient_visit_map']
    patient_diagnosis = data_source['patient_diagnosis']
    operation = data_source['operation']
    for patient_id in patient_info_map:
        # 如果某个Patient_ID没有对应的入院记录，则直接跳过不做记录
        if not patient_visit_map.__contains__(patient_id):
            continue

        # 生成Patient_Node节点及其相关内容
        patient_node = append_patient_node(root, patient_info_map[patient_id], patient_id)
        patient_visit = patient_visit_map[patient_id]
        for visit_id in patient_visit:
            # 生成Visit_Node节点及其相关内容
            visit_node = append_visit_node(patient_node, patient_visit, visit_id)

            # 生成住院Diagnosis节点及其相关内容
            diagnosis_node = SubElement(visit_node, "diagnosis")
            if patient_diagnosis.__contains__(patient_id) and patient_diagnosis[patient_id].__contains__(visit_id):
                diagnosis_node.set('contain_diagnosis', "true")
                diagnosis_info = patient_diagnosis[patient_id][visit_id]
                append_diagnosis_node(diagnosis_info, diagnosis_node)
            else:
                diagnosis_node.set('contain_diagnosis', "false")

            # 生成Operation节点及其相关内容
            operation_node = SubElement(visit_node, "operation")
            if operation.__contains__(patient_id) and operation[patient_id].__contains__(visit_id):
                operation_node.set('contain_operation', "true")
                operation_info = operation[patient_id][visit_id]
                append_operation_node(operation_info, operation_node)
            else:
                operation_node.set('contain_operation', "false")

            # 使用住院数据，留待以后填充
            if data_source['use_outpatient_data']:
                pass
    return root


def append_patient_node(root, patient_info, patient_id):
    # Patient_Node 属性
    patient_node_attr = {'patient_id': patient_id, 'sex': patient_info['sex'], 'birthday': patient_info[
        'birthday'].strftime("%Y-%m-%d %H:%M:%S"), 'ethnic_group': patient_info['ethnic_group']}
    patient_node = SubElement(root, 'patient_node', patient_node_attr)
    return patient_node


def append_visit_node(patient_node, patient_visit, visit_id):
    visit_map = patient_visit[visit_id]
    visit_node_attr = {'visit_id': visit_id, 'admission_date': visit_map[0].strftime("%Y-%m-%d %H:%M:%S"),
                       'discharge_date': visit_map[1].strftime("%Y-%m-%d %H:%M:%S"),
                       'identity': visit_map[2]}
    visit_node = SubElement(patient_node, "visit", visit_node_attr)
    return visit_node


def append_diagnosis_node(visit_diagnosis, diagnosis_node):

    for diagnosis_no in visit_diagnosis:
        diagnosis_map = visit_diagnosis[diagnosis_no]
        diagnosis_node_attr = {'diagnosis_no': diagnosis_no, 'diagnosis_type': diagnosis_map['diagnosis_type'],
                               'diagnosis_desc': diagnosis_map['diagnosis_desc'], 'icd_code': diagnosis_map['icd_code'],
                               'icd_version': diagnosis_map['icd_version'], 'normalized_icd': str(diagnosis_map[
                                                                                       'normalized_icd_code'])}
        SubElement(diagnosis_node, "diagnosis_item", diagnosis_node_attr)


def append_operation_node(visit_operation, operation_node):
    for operation_no in visit_operation:
        ope_map = visit_operation[operation_no]
        operation_node_attr = {'operation_no': operation_no, 'operation_date': ope_map['operation_date'].strftime(
            "%Y-%m-%d %H:%M:%S"), 'icd_code': ope_map['icd_code'],  'operation_description': ope_map[
            'operation_description'], 'heal': ope_map['heal'], 'normalized_icd': str(ope_map['normalized_icd_code'])}
        SubElement(operation_node, "operation_item", operation_node_attr)


def append_meta_data(root, data_source):
    patient_count, max_visit, min_visit, diagnosis_code_count_map, operation_code_count_map = meta_data_analysis(
        data_source)
    diagnosis_code_count_map['丢弃'] = 0
    operation_code_count_map['丢弃'] = 0
    value_sorted_diagnosis = sorted(diagnosis_code_count_map.items(), key=lambda d: d[1], reverse=True)
    value_sorted_operation = sorted(operation_code_count_map.items(), key=lambda d: d[1], reverse=True)

    metadata_node = SubElement(root, 'mata_data', {'patient_count': str(patient_count), 'max_visit': str(max_visit),
                                                   'min_visit': str(min_visit)})
    diagnosis_node = SubElement(metadata_node, 'diagnosis_coding', {'icd_code_size': str(len(value_sorted_diagnosis))})
    rank = 1
    for item in value_sorted_diagnosis:
        SubElement(diagnosis_node, 'diagnosis_icd', {'code': str(item[0]), 'count': str(item[1]), 'rank': str(rank)})
        rank += 1
    rank = 1
    operation_node = SubElement(metadata_node, 'operation_coding', {'icd_code_size': str(len(value_sorted_operation))})
    for item in value_sorted_operation:
        SubElement(operation_node, 'operation_icd', {'code': str(item[0]), 'count': str(item[1]), 'rank': str(rank)})
        rank += 1


def meta_data_analysis(data_source):
    patient_info_map = data_source['patient_info_map']
    patient_visit_map = data_source['patient_visit_map']
    patient_diagnosis = data_source['patient_diagnosis']
    operation = data_source['operation']
    if data_source['use_outpatient_data']:
        print("使用门诊数据的元数据分析代码尚未完成")

    patient_count = len(patient_info_map)
    max_visit = 0
    min_visit = 100
    for patient_id in patient_visit_map:
        visit_count = len(patient_visit_map[patient_id])
        if visit_count > max_visit:
            max_visit = visit_count
        if visit_count < min_visit:
            min_visit = visit_count

    # 最频繁下的诊断的统计
    diagnosis_code_count_map = {}
    for patient_id in patient_diagnosis:
        for visit_id in patient_diagnosis[patient_id]:
            for diagnosis_no in patient_diagnosis[patient_id][visit_id]:
                diagnosis_info = patient_diagnosis[patient_id][visit_id][diagnosis_no]
                normalized_icd = diagnosis_info['normalized_icd_code']
                if diagnosis_code_count_map.__contains__(normalized_icd):
                    diagnosis_code_count_map[normalized_icd] += 1
                else:
                    diagnosis_code_count_map[normalized_icd] = 1

    # 最频繁下的手术的统计
    operation_code_count_map = {}
    for patient_id in operation:
        for visit_id in operation[patient_id]:
            for operation_no in operation[patient_id][visit_id]:
                operation_info = operation[patient_id][visit_id][operation_no]
                normalized_icd = operation_info['normalized_icd_code']
                if operation_code_count_map.__contains__(normalized_icd):
                    operation_code_count_map[normalized_icd] += 1
                else:
                    operation_code_count_map[normalized_icd] = 1

    return patient_count, max_visit, min_visit, diagnosis_code_count_map, operation_code_count_map


def five_fold_split(data_source):
    patient_info_map = data_source['patient_info_map']
    patient_visit_map = data_source['patient_visit_map']
    patient_diagnosis = data_source['patient_diagnosis']
    operation = data_source['operation']
    use_outpatient_flag = data_source['use_outpatient_data']

    sub_data_0 = {'patient_info_map': {}, 'patient_visit_map': {}, 'patient_diagnosis': {}, 'operation': {},
                  'use_outpatient_data': use_outpatient_flag}
    sub_data_1 = {'patient_info_map': {}, 'patient_visit_map': {}, 'patient_diagnosis': {}, 'operation': {},
                  'use_outpatient_data': use_outpatient_flag}
    sub_data_2 = {'patient_info_map': {}, 'patient_visit_map': {}, 'patient_diagnosis': {}, 'operation': {},
                  'use_outpatient_data': use_outpatient_flag}
    sub_data_3 = {'patient_info_map': {}, 'patient_visit_map': {}, 'patient_diagnosis': {}, 'operation': {},
                  'use_outpatient_data': use_outpatient_flag}
    sub_data_4 = {'patient_info_map': {}, 'patient_visit_map': {}, 'patient_diagnosis': {}, 'operation': {},
                  'use_outpatient_data': use_outpatient_flag}

    sequence_id_list = []
    for sequence_id in patient_info_map:
        sequence_id_list.append(sequence_id)
    random.shuffle(sequence_id_list)

    for i in range(0, len(sequence_id_list)):
        sequence_id = sequence_id_list[i]
        if i % 5 == 0:
            if patient_info_map.__contains__(sequence_id):
                sub_data_0['patient_info_map'][sequence_id] = patient_info_map[sequence_id]
            if patient_visit_map.__contains__(sequence_id):
                sub_data_0['patient_visit_map'][sequence_id] = patient_visit_map[sequence_id]
            if patient_diagnosis.__contains__(sequence_id):
                sub_data_0['patient_diagnosis'][sequence_id] = patient_diagnosis[sequence_id]
            if operation.__contains__(sequence_id):
                sub_data_0['operation'][sequence_id] = operation[sequence_id]
        if i % 5 == 1:
            if patient_info_map.__contains__(sequence_id):
                sub_data_1['patient_info_map'][sequence_id] = patient_info_map[sequence_id]
            if patient_visit_map.__contains__(sequence_id):
                sub_data_1['patient_visit_map'][sequence_id] = patient_visit_map[sequence_id]
            if patient_diagnosis.__contains__(sequence_id):
                sub_data_1['patient_diagnosis'][sequence_id] = patient_diagnosis[sequence_id]
            if operation.__contains__(sequence_id):
                sub_data_1['operation'][sequence_id] = operation[sequence_id]
        if i % 5 == 2:
            if patient_info_map.__contains__(sequence_id):
                sub_data_2['patient_info_map'][sequence_id] = patient_info_map[sequence_id]
            if patient_visit_map.__contains__(sequence_id):
                sub_data_2['patient_visit_map'][sequence_id] = patient_visit_map[sequence_id]
            if patient_diagnosis.__contains__(sequence_id):
                sub_data_2['patient_diagnosis'][sequence_id] = patient_diagnosis[sequence_id]
            if operation.__contains__(sequence_id):
                sub_data_2['operation'][sequence_id] = operation[sequence_id]
        if i % 5 == 3:
            if patient_info_map.__contains__(sequence_id):
                sub_data_3['patient_info_map'][sequence_id] = patient_info_map[sequence_id]
            if patient_visit_map.__contains__(sequence_id):
                sub_data_3['patient_visit_map'][sequence_id] = patient_visit_map[sequence_id]
            if patient_diagnosis.__contains__(sequence_id):
                sub_data_3['patient_diagnosis'][sequence_id] = patient_diagnosis[sequence_id]
            if operation.__contains__(sequence_id):
                sub_data_3['operation'][sequence_id] = operation[sequence_id]
        if i % 5 == 4:
            if patient_info_map.__contains__(sequence_id):
                sub_data_4['patient_info_map'][sequence_id] = patient_info_map[sequence_id]
            if patient_visit_map.__contains__(sequence_id):
                sub_data_4['patient_visit_map'][sequence_id] = patient_visit_map[sequence_id]
            if patient_diagnosis.__contains__(sequence_id):
                sub_data_4['patient_diagnosis'][sequence_id] = patient_diagnosis[sequence_id]
            if operation.__contains__(sequence_id):
                sub_data_4['operation'][sequence_id] = operation[sequence_id]

    return sub_data_0, sub_data_1, sub_data_2, sub_data_3, sub_data_4


# 缩进优化，使文件易于阅读
def prettify(element):
    # 此处只能用utf-8，不然会有错误
    rough_string = ElementTree.tostring(element, 'utf-8')
    recreate_xml = minidom.parseString(rough_string)
    return recreate_xml.toprettyxml(indent="  ")


def unit_test():
    patient_info_file_path = "..\\..\\resource\\patient_info.csv"
    outpatient_diagnosis_file_path = "..\\..\\resource\\outpatient_diagnosis.csv"
    operation_info_file_path = "..\\..\\resource\\operation.csv"
    hospitalized_patient_diagnosis_file_path = "..\\..\\resource\\hospitalized_patient_diagnosis.csv"
    hospitalized_patient_visit_file_path = "..\\..\\resource\\hospitalized_patient_visit_admission_date.csv"

    data_source = load_data.load_all_data(patient_info_file_path, outpatient_diagnosis_file_path,
                                          operation_info_file_path, hospitalized_patient_diagnosis_file_path,
                                          hospitalized_patient_visit_file_path)
    data_source = data_cleansing.cleansing_strategy(data_source, threshold=3, use_outpatient_data=False)
    sub_data_list = five_fold_split(data_source)

    for i in range(0, 5):
        xml_content = data_reconstruction(sub_data_list[i])
        with open("..\\..\\resource\\reconstruct_data\\reconstruction_no_"+str(i)+".xml", 'w', encoding='utf-8-sig',
                  newline="") as xml_file:
            xml_content = prettify(xml_content)
            xml_file.write(xml_content)


if __name__ == "__main__":
    unit_test()
