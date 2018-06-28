from xml.etree import ElementTree
import datetime


def load_need_data_5_fold(xml_file_path, xml_file_name_list, diagnosis_reserve, operation_reserve):
    data_sequence_info_list = []
    operation_rank_sum_map = {}
    diagnosis_rank_sum_map = {}
    # 排出统一的rank
    for i in range(0, len(xml_file_name_list)):
        data_source = parsing_xml(xml_file_path+xml_file_name_list[i])
        diagnosis_code_rank_map, operation_code_rank_map, _, _, _, _ = data_source
        for item in diagnosis_code_rank_map:
            if not diagnosis_rank_sum_map.__contains__(item):
                diagnosis_rank_sum_map[item] = int(diagnosis_code_rank_map[item])
            else:
                diagnosis_rank_sum_map[item] += int(diagnosis_code_rank_map[item])
        for item in operation_code_rank_map:
            if not operation_rank_sum_map.__contains__(item):
                operation_rank_sum_map[item] = int(operation_code_rank_map[item])
            else:
                operation_rank_sum_map[item] += int(operation_code_rank_map[item])

    diagnosis_rank_sum_list = sorted([[item, diagnosis_rank_sum_map[item]] for item in diagnosis_rank_sum_map],
                                     key=lambda k: k[1])
    operation_rank_sum_list = sorted([[item, operation_rank_sum_map[item]] for item in operation_rank_sum_map],
                                     key=lambda k: k[1])
    index = 0
    diagnosis_code_rank_map = {}
    for item in diagnosis_rank_sum_list:
        diagnosis_code_rank_map[item[0]] = index
        index += 1
    operation_code_rank_map = {}
    index = 0
    for item in operation_rank_sum_list:
        operation_code_rank_map[item[0]] = index
        index += 1

    index_name_map = generate_index_name_map(diagnosis_rank_sum_list, operation_rank_sum_list, diagnosis_reserve,
                                             operation_reserve)

    target_data = []
    for i in range(0, len(xml_file_name_list)):
        data_source = parsing_xml(xml_file_path+xml_file_name_list[i])
        _, _, patient_info_map, patient_visit_date_map, visit_diagnosis_map, visit_operation_map = data_source

        # exclude rare event
        diagnosis_map = exclude_rare_diagnosis(diagnosis_reserve, diagnosis_code_rank_map, visit_diagnosis_map,)
        operation_map = exclude_rare_operation(operation_reserve, operation_code_rank_map, visit_operation_map,)
        data_sequence_info_list.append([diagnosis_map, operation_map, patient_visit_date_map, None])
        data_sequence_info = generate_sequence_map(diagnosis_map, operation_map, patient_visit_date_map, None,
                                                   index_name_map)
        target_data.append(data_sequence_info)

    return target_data, index_name_map


def generate_index_name_map(diagnosis_rank_sum_list, operation_rank_sum_list, diagnosis_reserve, operation_reserve):
    index_name_map = {}
    index = 0
    for item in diagnosis_rank_sum_list:
        diagnosis_item = "D" + item[0]
        if not index_name_map.__contains__(diagnosis_item):
            index_name_map[diagnosis_item] = index
            index += 1
            if index >= diagnosis_reserve:
                break
    for item in operation_rank_sum_list:
        operation_type = "O" + item[0]
        if not index_name_map.__contains__(operation_type):
            index_name_map[operation_type] = index
            index += 1
            if index >= diagnosis_reserve+operation_reserve:
                    break

    return index_name_map


def generate_sequence_map(visit_diagnosis_map, visit_operation_map, patient_visit_date_map, outpatient_info,
                          index_name_map):
    # 扫描所有入院记录，确认第一次入院时间，将此时间标记为0
    patient_first_visit_time = {}
    for patient_id in patient_visit_date_map:
        for visit_id in patient_visit_date_map[patient_id]:
            visit_day = patient_visit_date_map[patient_id][visit_id]
            visit_day = datetime.datetime.strptime(visit_day, "%Y-%m-%d %H:%M:%S")
            if not patient_first_visit_time.__contains__(patient_id):
                patient_first_visit_time[patient_id] = visit_day
            if visit_day < patient_first_visit_time[patient_id]:
                patient_first_visit_time[patient_id] = visit_day
    if outpatient_info is not None:
        # TODO
        # 扫描以获得门诊数据的相关内容
        pass

    # 重新标定所有数据的时间信息和序号信息
    event_sequence_map = {}
    for patient_id in patient_visit_date_map:
        event_sequence = []
        for visit_id in patient_visit_date_map[patient_id]:
            if visit_diagnosis_map.__contains__(patient_id) and visit_diagnosis_map[patient_id].__contains__(visit_id):
                for diagnosis_item in visit_diagnosis_map[patient_id][visit_id]:
                    # 时间重标
                    visit_day = patient_visit_date_map[patient_id][visit_id]
                    visit_day = datetime.datetime.strptime(visit_day, "%Y-%m-%d %H:%M:%S")
                    first_day = patient_first_visit_time[patient_id]
                    time_interval = (visit_day - first_day).days
                    # 考虑由于计时尺度（部分数据精确到分，部分数据精确到天）造成的少量偏差
                    if -2 < time_interval < 0:
                        time_interval = 0
                    elif time_interval >= 0:
                        pass
                    else:
                        continue

                    event_index = index_name_map['D'+diagnosis_item]
                    event_time = time_interval
                    event_sequence.append((event_index, event_time))
            if visit_operation_map.__contains__(patient_id) and visit_operation_map[patient_id].__contains__(visit_id):
                operation_list = visit_operation_map[patient_id][visit_id]
                for operation_item in operation_list:
                    operation_type, operation_time = operation_item
                    operation_type = "O"+operation_type

                    visit_day = datetime.datetime.strptime(operation_time, "%Y-%m-%d %H:%M:%S")
                    first_day = patient_first_visit_time[patient_id]
                    time_interval = (visit_day - first_day).days
                    # 考虑由于计时尺度（部分数据精确到分，部分数据精确到天）造成的少量偏差
                    if -2 < time_interval < 0:
                        time_interval = 0
                    elif time_interval >= 0:
                        pass
                    else:
                        continue

                    event_index = index_name_map[operation_type]
                    event_time = time_interval
                    event_sequence.append((event_index, event_time))
        event_sequence = sorted(event_sequence, key=lambda data_tuple: data_tuple[1])
        if len(event_sequence) > 0:
            event_sequence_map[patient_id] = event_sequence
    if outpatient_info is not None:
        # TODO
        # 扫描以获得门诊数据的相关内容
        pass

    return event_sequence_map


def parsing_xml(file_path):
    """
    本函数以期从XML中解析出所需要的数据，此处的解析方式比较类似于流式解析，因此函数会写的比较复杂
    :param file_path:
    :return:
    """
    diagnosis_code_rank_map = {}
    operation_code_rank_map = {}
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
            if node_tag == "diagnosis_icd":
                diagnosis_code_rank_map[node_attrib['code']] = node_attrib['rank']
            if node_tag == "operation_icd":
                operation_code_rank_map[node_attrib['code']] = node_attrib['rank']

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
                diagnosis_icd = node_attrib['normalized_icd']
                if not patient_visit_diagnosis_map.__contains__(current_patient):
                    patient_visit_diagnosis_map[current_patient] = {}
                if not patient_visit_diagnosis_map[current_patient].__contains__(current_visit):
                    patient_visit_diagnosis_map[current_patient][current_visit] = []
                patient_visit_diagnosis_map[current_patient][current_visit].append(diagnosis_icd)

            if node_tag == "operation_item":
                operation_icd = node_attrib['normalized_icd']
                operation_date = node_attrib['operation_date']
                if not patient_visit_operation_map.__contains__(current_patient):
                    patient_visit_operation_map[current_patient] = {}
                if not patient_visit_operation_map[current_patient].__contains__(current_visit):
                    patient_visit_operation_map[current_patient][current_visit] = []
                patient_visit_operation_map[current_patient][current_visit].append([operation_icd, operation_date])

    return [diagnosis_code_rank_map, operation_code_rank_map, patient_info_map, patient_visit_date_map,
            patient_visit_diagnosis_map, patient_visit_operation_map]


def exclude_rare_diagnosis(top_diagnosis_reserved, diagnosis_code_rank_map, patient_visit_diagnosis_map):
    for patient_id in patient_visit_diagnosis_map:
        for visit_id in patient_visit_diagnosis_map[patient_id]:
            diagnosis_list = patient_visit_diagnosis_map[patient_id][visit_id]
            reserve_list = []
            for item in diagnosis_list:
                rank = diagnosis_code_rank_map[item]
                if int(rank) < top_diagnosis_reserved:
                    reserve_list.append(item)
            patient_visit_diagnosis_map[patient_id][visit_id] = reserve_list
    return patient_visit_diagnosis_map


def exclude_rare_operation(top_operation_reserved, operation_code_rank_map, patient_visit_operation_map):
    for patient_id in patient_visit_operation_map:
        for visit_id in patient_visit_operation_map[patient_id]:
            operation_list = patient_visit_operation_map[patient_id][visit_id]
            new_operation_list = []
            for item in operation_list:
                rank = operation_code_rank_map[item[0]]
                if int(rank) < top_operation_reserved:
                    new_operation_list.append(item)
            patient_visit_operation_map[patient_id][visit_id] = new_operation_list
    return patient_visit_operation_map


def event_stat():
    xml_path = "..\\..\\resource\\reconstruct_data\\"
    file_name_list = ['reconstruction_no_0.xml', 'reconstruction_no_1.xml', 'reconstruction_no_2.xml',
                      'reconstruction_no_3.xml', 'reconstruction_no_4.xml']
    data_sequence_list, index_name_map = load_need_data_5_fold(xml_path, file_name_list, 50, 0)
    # 每种事件发生了多少次
    # 一共发生了多少事件
    # 每个病人的时间跨度
    # 平均每个病人的一个序列发生了多少事件
    sequence_list = {}
    for i in data_sequence_list:
        sequence_list.update(data_sequence_list[i])

    # 每种事件发生了多少次
    event_count = {}
    for patient_id in sequence_list:
        single_sequence_list = sequence_list[patient_id]
        for item in single_sequence_list:
            item_index = item[0]
            if not event_count.__contains__(item_index):

    print(data_source)


if __name__ == "__main__":
    event_stat()
