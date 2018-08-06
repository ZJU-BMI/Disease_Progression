# coding=utf-8
import datetime
import os
import random
from xml.etree import ElementTree

import numpy as np


def parsing_xml(file_path):
    """
    本函数以期从XML中解析出所需要的数据，此处的解析方式比较类似于流式解析，因此函数会写的比较复杂
    :param file_path:
    :return:
    """
    patient_info_map = {}
    patient_visit_date_map = {}
    patient_visit_diagnosis_map = {}
    patient_visit_operation_map = {}

    with open(file_path, 'rt', encoding='utf-8-sig') as f:
        tree = ElementTree.parse(f)

        current_patient = -1
        current_visit = -1
        for node in tree.iter():
            node_tag = node.tag
            node_attrib = node.attrib

            # 调取病人信息
            if node_tag == "patient_node":
                current_patient = node_attrib['patient_id']
                birthday = node_attrib['birthday']
                sex = node_attrib['sex']
                patient_info_map[current_patient] = [birthday, sex]

            if node_tag == 'visit':
                current_visit = node_attrib['visit_id']
                admission_date = node_attrib['admission_date']
                if not patient_visit_date_map.__contains__(current_patient):
                    patient_visit_date_map[current_patient] = {}
                patient_visit_date_map[current_patient][current_visit] = admission_date

            if node_tag == "diagnosis_item":
                diagnosis_icd = node_attrib['normalized_code']
                if not patient_visit_diagnosis_map.__contains__(current_patient):
                    patient_visit_diagnosis_map[current_patient] = {}
                if not patient_visit_diagnosis_map[current_patient].__contains__(current_visit):
                    patient_visit_diagnosis_map[current_patient][current_visit] = []
                patient_visit_diagnosis_map[current_patient][current_visit].append(diagnosis_icd)

            if node_tag == "procedure_item":
                operation_icd = node_attrib['normalized_code']
                if not patient_visit_operation_map.__contains__(current_patient):
                    patient_visit_operation_map[current_patient] = {}
                if not patient_visit_operation_map[current_patient].__contains__(current_visit):
                    patient_visit_operation_map[current_patient][current_visit] = []
                patient_visit_operation_map[current_patient][current_visit].append(operation_icd)

    return [patient_info_map, patient_visit_date_map,
            patient_visit_diagnosis_map, patient_visit_operation_map]


def diagnosis_rank(diagnosis_map):
    count_map = dict()
    for patient_id in diagnosis_map:
        for visit_id in diagnosis_map[patient_id]:
            for item in diagnosis_map[patient_id][visit_id]:
                if not count_map.__contains__(item):
                    count_map[item] = 1
                else:
                    count_map[item] += 1
    count_list = []
    for key in count_map:
        count_list.append([key, count_map[key]])
    count_list = sorted(count_list, key=lambda x: x[1], reverse=True)

    rank = 1
    rank_map = dict()
    for item in count_list:
        rank_map[item[0]] = rank
        rank += 1

    return rank_map


def procedure_rank(procedure_map):
    count_map = dict()
    for patient_id in procedure_map:
        for visit_id in procedure_map[patient_id]:
            for item in procedure_map[patient_id][visit_id]:
                if not count_map.__contains__(item):
                    count_map[item] = 1
                else:
                    count_map[item] += 1
    count_list = []
    for key in count_map:
        count_list.append([key, count_map[key]])
    count_list = sorted(count_list, key=lambda x: x[1], reverse=True)

    rank = 1
    rank_map = dict()
    for item in count_list:
        if len(item) == 0:
            count_map[item] = 100000
            continue
        rank_map[item[0]] = rank
        rank += 1

    return rank_map


def exclude_rare_diagnosis(diagnosis_reserved, diagnosis_code_rank_map, diagnosis_map):
    for patient_id in diagnosis_map:
        for visit_id in diagnosis_map[patient_id]:
            diagnosis_list = diagnosis_map[patient_id][visit_id]
            reserve_list = []
            for item in diagnosis_list:
                rank = diagnosis_code_rank_map[item]
                if int(rank) <= diagnosis_reserved:
                    reserve_list.append(item)
            diagnosis_map[patient_id][visit_id] = reserve_list
    return diagnosis_map


def exclude_rare_procedure(procedure_reserved, procedure_code_rank_map, procedure_map):
    for patient_id in procedure_map:
        for visit_id in procedure_map[patient_id]:
            procedure_list = procedure_map[patient_id][visit_id]
            new_procedure_list = []
            for item in procedure_list:
                rank = procedure_code_rank_map[item]
                if int(rank) <= procedure_reserved:
                    new_procedure_list.append(item)
            procedure_map[patient_id][visit_id] = new_procedure_list
    return procedure_map


def generate_index_name_map(diagnosis_rank_map, procedure_rank_map, diagnosis_reserve, procedure_reserve):
    diagnosis_list = []
    for key in diagnosis_rank_map:
        diagnosis_list.append([key, diagnosis_rank_map[key]])
    procedure_list = []
    for key in procedure_rank_map:
        procedure_list.append([key, procedure_rank_map[key]])
    diagnosis_list = sorted(diagnosis_list, key=lambda x: x[1])
    procedure_list = sorted(procedure_list, key=lambda x: x[1])

    index_name_map = {}
    index = 0
    for item in diagnosis_list:
        diagnosis_item = "D" + str(item[0])
        if not index_name_map.__contains__(diagnosis_item):
            index_name_map[diagnosis_item] = index
            index += 1
            if index >= diagnosis_reserve:
                break
    for item in procedure_list:
        operation_type = "P" + str(item[0])
        if not index_name_map.__contains__(operation_type):
            index_name_map[operation_type] = index
            index += 1
            if index >= diagnosis_reserve + procedure_reserve:
                break
    return index_name_map


def generate_sequence_map(diagnosis_map, procedure_map, visit_date_map, index_name_map):
    # 扫描所有入院记录，确认第一次入院时间，将此时间标记为0
    patient_first_visit_time = {}
    for patient_id in visit_date_map:
        for visit_id in visit_date_map[patient_id]:
            visit_day = visit_date_map[patient_id][visit_id]
            visit_day = datetime.datetime.strptime(visit_day, "%Y-%m-%d %H:%M:%S")
            if not patient_first_visit_time.__contains__(patient_id):
                patient_first_visit_time[patient_id] = visit_day
            if visit_day < patient_first_visit_time[patient_id]:
                patient_first_visit_time[patient_id] = visit_day

    # 重新标定所有数据的时间信息和序号信息
    event_sequence_map = {}
    for patient_id in visit_date_map:
        event_sequence = []
        for visit_id in visit_date_map[patient_id]:
            if diagnosis_map.__contains__(patient_id) and diagnosis_map[patient_id].__contains__(visit_id):
                for diagnosis_item in diagnosis_map[patient_id][visit_id]:
                    # 时间重标
                    visit_day = visit_date_map[patient_id][visit_id]
                    visit_day = datetime.datetime.strptime(visit_day, "%Y-%m-%d %H:%M:%S")
                    first_day = patient_first_visit_time[patient_id]
                    time_interval = (visit_day - first_day).days

                    event_index = index_name_map['D' + diagnosis_item]
                    event_time = time_interval
                    event_sequence.append((event_index, event_time))

            if procedure_map.__contains__(patient_id) and procedure_map[patient_id].__contains__(visit_id):
                procedure_list = procedure_map[patient_id][visit_id]
                for procedure_item in procedure_list:
                    visit_day = visit_date_map[patient_id][visit_id]
                    visit_day = datetime.datetime.strptime(visit_day, "%Y-%m-%d %H:%M:%S")
                    first_day = patient_first_visit_time[patient_id]
                    time_interval = (visit_day - first_day).days

                    event_index = index_name_map['P' + procedure_item]
                    event_time = time_interval
                    event_sequence.append((event_index, event_time))
        event_sequence = sorted(event_sequence, key=lambda data_tuple: data_tuple[1])
        if len(event_sequence) > 0:
            event_sequence_map[patient_id] = event_sequence
    return event_sequence_map


def derive_hawkes_data(file_path, reserve_diagnosis, reserve_procedure):
    patient_info, visit_date, diagnosis_map, procedure_map = parsing_xml(file_path)

    # 找到高频数据
    diagnosis_rank_map = diagnosis_rank(diagnosis_map)
    procedure_rank_map = procedure_rank(procedure_map)

    # 去除不需要的低频数据
    diagnosis_map = exclude_rare_diagnosis(reserve_diagnosis, diagnosis_rank_map, diagnosis_map)
    procedure_map = exclude_rare_procedure(reserve_procedure, procedure_rank_map, procedure_map)

    # 得到编号-名称map
    index_name_map = generate_index_name_map(diagnosis_rank_map, procedure_rank_map, reserve_diagnosis,
                                             reserve_procedure)
    event_sequence_map = generate_sequence_map(diagnosis_map, procedure_map, visit_date, index_name_map)
    return event_sequence_map, index_name_map


def derive_neural_network_data(file_path, reserve_diagnosis, reserve_procedure, time_stamp):
    patient_info, visit_date, diagnosis_map, procedure_map = parsing_xml(file_path)

    # 找到高频数据
    diagnosis_rank_map = diagnosis_rank(diagnosis_map)
    procedure_rank_map = procedure_rank(procedure_map)

    # 去除不需要的低频数据
    diagnosis_map = exclude_rare_diagnosis(reserve_diagnosis, diagnosis_rank_map, diagnosis_map)
    procedure_map = exclude_rare_procedure(reserve_procedure, procedure_rank_map, procedure_map)

    # 得到编号-名称map
    index_name_map = generate_index_name_map(diagnosis_rank_map, procedure_rank_map, reserve_diagnosis,
                                             reserve_procedure)

    # 病人的时间信息和事件信息
    patient_x = np.zeros([len(visit_date), time_stamp, reserve_diagnosis + reserve_procedure])
    patient_t = np.zeros([len(visit_date), time_stamp, 1])

    patient_index = 0
    for patient_id in visit_date:
        single_patient_x = patient_x[patient_index]
        single_patient_t = patient_t[patient_index]
        for visit_id in visit_date[patient_id]:
            if int(visit_id) > time_stamp:
                continue
            # t 信息
            if int(visit_id) == 1:
                single_patient_t[0][0] = 0
            else:
                first_day = datetime.datetime.strptime(visit_date[patient_id]['1'], "%Y-%m-%d %H:%M:%S")
                current_day = datetime.datetime.strptime(visit_date[patient_id][visit_id], "%Y-%m-%d %H:%M:%S")
                time_interval = (current_day - first_day).days
                single_patient_t[int(visit_id) - 1][0] = time_interval

            # x 信息
            if diagnosis_map.__contains__(patient_id) and diagnosis_map[patient_id].__contains__(visit_id):
                diagnosis = diagnosis_map[patient_id][visit_id]
                for item in diagnosis:
                    d_item = 'D' + str(item)
                    index = index_name_map[d_item] - 1
                    single_patient_x[int(visit_id) - 1, index] = 1

            if procedure_map.__contains__(patient_id) and procedure_map[patient_id].__contains__(visit_id):
                procedure = procedure_map[patient_id][visit_id]
                for item in procedure:
                    d_item = 'P' + str(item)
                    index = index_name_map[d_item] - 1
                    single_patient_x[int(visit_id) - 1, index] = 1

        patient_index += 1

    return patient_x, patient_t


def hawkes_random_split(event_sequence_map, fold=5):
    event_list = []
    for patient_id in event_sequence_map:
        event_list.append([patient_id, event_sequence_map[patient_id]])
    random.shuffle(event_list)

    batch_size = len(event_list) // 5

    batch_map = dict()
    for i in range(0, fold):
        batch_event = event_list[i * batch_size: (i + 1) * batch_size]
        single_batch_map = dict()

        for item in batch_event:
            single_batch_map[item[0]] = item[1]
        batch_map[i + 1] = single_batch_map
    return batch_map


def hawkes(reserve_diagnosis, reserve_procedure, file_path, file_name):
    event_sequence_map, index_name_map = derive_hawkes_data(file_path + file_name,
                                                            reserve_diagnosis=reserve_diagnosis,
                                                            reserve_procedure=reserve_procedure)
    batch_map = hawkes_random_split(event_sequence_map, fold=5)
    return batch_map, index_name_map


def neural_nets(reserve_diagnosis, reserve_procedure, time_stamp, file_path, file_name):
    """
    输出经过随机排序的BTD的数据，经过核验，没有发现问题
    :param reserve_diagnosis:
    :param reserve_procedure:
    :param time_stamp:
    :param file_path:
    :param file_name:
    :return:
    """
    # 返回Time Major的数据
    # index 一样的x, t，对应一样的人
    patient_x, patient_t = derive_neural_network_data(os.path.join(file_path, file_name),
                                                      reserve_diagnosis=reserve_diagnosis,
                                                      reserve_procedure=reserve_procedure,
                                                      time_stamp=time_stamp)

    shuffle_list = []
    for i in range(0, len(patient_t)):
        shuffle_list.append(np.concatenate([patient_x[i], patient_t[i]], axis=1))
    random.shuffle(shuffle_list)

    shuffle_list = np.array(shuffle_list)
    x = shuffle_list[:, :, 0:-1]
    t = shuffle_list[:, :, -1]
    t = t[:, :, np.newaxis]

    save_path = file_path + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    np.save(save_path + '_x.npy', x)
    np.save(save_path + '_t.npy', t)


def main():
    file_path = os.path.abspath('..\\..\\..') + '\\reconstruct_data\\mimic_3\\reconstruct\\'
    file_name = 'reconstructed.xml'
    reserve_diagnosis = 80
    reserve_procedure = 30
    time_stamp = 5
    # hawkes(reserve_diagnosis=reserve_diagnosis,
    #        reserve_procedure=reserve_procedure, file_path=file_path, file_name=file_name)

    neural_nets(reserve_diagnosis, reserve_procedure, time_stamp, file_path, file_name)


if __name__ == '__main__':
    main()
