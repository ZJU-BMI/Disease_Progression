from plagh import load_data

"""
数据清洗（预处理/重建）策略主要完成以下几件事情
1.根据预设的阈值，选择住院+门诊次数大于某一阈值的病人数据，丢弃不满足要求的病人数据
2.住院诊断数据编码存在ICD9-ICD10两个版本，需要进行合并
3.住院诊断数据为6位，需要选择合适的粒度
4.门诊的诊断数据没有编码，需要清洗生成编码
"""


def cleansing_strategy(data_source, threshold, use_outpatient_data=False):
    """
    :param data_source: 原始数据，包括load_data载入的基本信息,入院,住院诊断,手术,门诊诊断5类数据，名称分别为
    patient_info_map, patient_visit_map, patient_diagnosis, operation, outpatient_diagnosis
    :param use_outpatient_data: 是否使用门诊数据（默认为否，若是则需要结构化门诊数据）
    :param threshold: 长期纵向数据阈值
    :return: 返回data_source 字典，包括
    住院诊断：DS:[patient_id][visit_id][diagnosis_no]{diagnosis_type, diagnosis_desc, icd_code, icd_version,
    diagnosis_date, treat_day, treat_result, normalized_icd_code}
    门诊诊断（若处理了）：DS:[patient_id][visit_no]{visit_date, diagnosis_description, icd_code, normalized_icd_code}
    手术数据：DS:[patient_id][visit_id][operation_no]{icd_code, operation_date, operation_description,
    heal， normalized_icd_code}
    病人基本信息： DS:[patient_id]{sex, birthday, ethnic_group}
    住院信息： DS:[patient_id][visit_id][admission_date,discharge_date,identity]
    是否使用住院数据 Flag
    """
    patient_diagnosis = data_source['patient_diagnosis']
    operation = data_source['operation']
    outpatient_diagnosis_map = data_source['outpatient_diagnosis']

    # 如果使用门诊数据，进行必要的清洗
    if use_outpatient_data:
        outpatient_diagnosis_map = cleansing_outpatient_data(outpatient_diagnosis_map)
        outpatient_icd_normalize(outpatient_diagnosis_map=outpatient_diagnosis_map)

    # 合并、归一化诊断ICD9，ICD10编码
    hospitalized_icd_normalize(hospitalized_diagnosis_map=patient_diagnosis)

    # 归一化手术ICD9编码
    operation_icd_normalize(operation)

    # 清除入院次数不满足最低要求的数据
    data_source = exclude_short_data(data_source, threshold)

    # 加上是否使用住院数据的Flag
    data_source['use_outpatient_data'] = use_outpatient_data
    return data_source


# 清洗门诊数据
# 具体清洗策略之后补充
def cleansing_outpatient_data(outpatient_data):
    pass
    return outpatient_data


# 如何添加门诊数据之后补充
def add_outpatient_data(outpatient_diagnosis_map):
    return outpatient_diagnosis_map


# 丢弃所有数据中长期纵向数据不满足要求的数据
def exclude_short_data(data_source, threshold):
    patient_info_map = data_source['patient_info_map']
    patient_visit_map = data_source['patient_visit_map']
    patient_diagnosis = data_source['patient_diagnosis']
    operation = data_source['operation']
    outpatient_diagnosis_map = data_source['outpatient_diagnosis']

    exclude_id_set = set()
    for patient_id in patient_visit_map:
        if len(patient_visit_map[patient_id]) < threshold:
            exclude_id_set.add(patient_id)
    for patient_id in exclude_id_set:
        if patient_info_map.__contains__(patient_id):
            patient_info_map.pop(patient_id)
        if patient_visit_map.__contains__(patient_id):
            patient_visit_map.pop(patient_id)
        if patient_diagnosis.__contains__(patient_id):
            patient_diagnosis.pop(patient_id)
        if operation.__contains__(patient_id):
            operation.pop(patient_id)
        if outpatient_diagnosis_map.__contains__(patient_id):
            outpatient_diagnosis_map.pop(patient_id)

    return data_source


# 对门诊数据中的ICD编码进行归一化
def outpatient_icd_normalize(outpatient_diagnosis_map):
    for patient_id in outpatient_diagnosis_map:
        for visit_no in outpatient_diagnosis_map[patient_id]:
            diagnosis_info = outpatient_diagnosis_map[patient_id][visit_no]
            icd_code = diagnosis_info['icd_code']
            diagnosis_info['normalized_icd_code'] = icd_code[0]
    return outpatient_diagnosis_map


# 对住院数据中的ICD编码进行归一化
def hospitalized_icd_normalize(hospitalized_diagnosis_map):
    for patient_id in hospitalized_diagnosis_map:
        for visit_id in hospitalized_diagnosis_map[patient_id]:
            for diagnosis_no in hospitalized_diagnosis_map[patient_id][visit_id]:
                diagnosis_info = hospitalized_diagnosis_map[patient_id][visit_id][diagnosis_no]
                icd_code = diagnosis_info['icd_code']
                diagnosis_info['normalized_icd_code'] = diagnosis_icd_normalize_strategy(icd_code)
    return hospitalized_diagnosis_map


def diagnosis_icd_normalize_strategy(icd_code):
    if len(icd_code) < 3 or icd_code == '-' or icd_code == '*':
        return '丢弃'

    if icd_code[0:3] == 'I50':
        if icd_code == 'I50.103' or icd_code == 'I50.907' or icd_code == 'I50.911':
            normalized_name = '急性左心衰'
        elif icd_code == 'I50.901':
            normalized_name = '心功能1级'
        elif icd_code == 'I50.902':
            normalized_name = '心功能2级'
        elif icd_code == 'I50.903':
            normalized_name = '心功能3级'
        elif icd_code == 'I50.910':
            normalized_name = '心功能4级'
        else:
            normalized_name = '丢弃'
    elif icd_code[0] == 'C':
        normalized_name = '癌症'
    elif len(icd_code) == 8 and (icd_code[-2:-1] == '/3' or icd_code[-2:-1] == '/6'):
        normalized_name = '癌症'
    elif icd_code[0:3] == 'D50' or icd_code[0:3] == 'D64':
        normalized_name = '贫血'
    elif icd_code[0:3] == 'I05' or icd_code[0:3] == 'I06' or icd_code[0:3] == 'I07' or icd_code[0:3] == 'I08' or  \
            icd_code[0:3] == 'I09':
        normalized_name = '风湿性瓣膜病'
    elif icd_code[0:3] == 'I34' or icd_code[0:3] == 'I35' or icd_code[0:3] == 'I36':
        normalized_name = '非风湿性瓣膜病'
    elif icd_code[0:3] == 'I44' or icd_code[0:3] == 'I45':
        normalized_name = '传导阻滞'
    elif icd_code[0:3] == 'I47' or icd_code[0:3] == 'I48':
        normalized_name = '阵发性心动过速,心房、心室的扑动颤动'
    elif icd_code[0:2] == 'I6' or icd_code[0:3] == 'I6':
        normalized_name = '脑血管疾病（造成或未造成脑梗死）'
    elif icd_code[0:3] == 'J15' or icd_code[0:3] == 'J16' or icd_code[0:3] == 'J17' or icd_code[0:3] == 'J18':
        normalized_name = '肺炎'
    elif icd_code[0] == "Z":
        normalized_name = "丢弃"

    # icd 9 合并归一化
    elif icd_code[0:3] == '394' or icd_code[0:3] == '395' or icd_code[0:3] == '396' or icd_code[0:3] == '397' or \
            icd_code[0:3] == '398':
        normalized_name = '风湿性瓣膜病'
    elif icd_code[0:3] == '401' or icd_code[0:3] == '402':
        normalized_name = 'I10'
    elif icd_code[0:3] == '426':
        normalized_name = '传导阻滞'
    elif icd_code[0:3] == '427':
        normalized_name = '阵发性心动过速,心房、心室的扑动颤动'
    elif icd_code[0:3] == '428':
        if icd_code == '428.006':
            normalized_name = '心功能1级'
        elif icd_code == '428.007':
            normalized_name = '心功能2级'
        elif icd_code == '428.008':
            normalized_name = '心功能3级'
        elif icd_code == '428.006':
            normalized_name = '心功能4级'
        elif icd_code == '428.102':
            normalized_name = '急性左心衰'
        else:
            normalized_name = '丢弃'

    elif icd_code[0:3] == '250':
        normalized_name = 'E11'
    elif icd_code[0:3] == '410':
        normalized_name = "I21"
    elif icd_code[0:3] == '412':
        normalized_name = "I25"
    elif icd_code[0:3] == '413':
        normalized_name = "I20"
    elif icd_code[0:3] == '438':
        normalized_name = '脑血管疾病（造成或未造成脑梗死）'
    elif icd_code[0:3] == '486':
        normalized_name = '肺炎'
    elif icd_code[0:3] == '585':
        normalized_name = 'N18'

    else:
        normalized_name = icd_code[0:3]

    return normalized_name


# 对手术数据的ICD编码进行归一化
def operation_icd_normalize(operation_map):
    for patient_id in operation_map:
        for visit_id in operation_map[patient_id]:
            for operation_id in operation_map[patient_id][visit_id]:
                operation_info = operation_map[patient_id][visit_id][operation_id]
                icd_code = operation_info['icd_code']
                operation_info['normalized_icd_code'] = icd_code
    return operation_map


# 是否进行数据padding
def padding_data(source_data):
    return source_data


def unit_test():
    patient_info_file_path = "..\\..\\resource\\patient_info.csv"
    outpatient_diagnosis_file_path = "..\\..\\resource\\outpatient_diagnosis.csv"
    operation_info_file_path = "..\\..\\resource\\operation.csv"
    hospitalized_patient_diagnosis_file_path = "..\\..\\resource\\hospitalized_patient_diagnosis.csv"
    hospitalized_patient_visit_file_path = "..\\..\\resource\\hospitalized_patient_visit_admission_date.csv"
    data_source = load_data.load_all_data(patient_info_file_path, outpatient_diagnosis_file_path,
                                          operation_info_file_path, hospitalized_patient_diagnosis_file_path,
                                          hospitalized_patient_visit_file_path)
    data_source = cleansing_strategy(data_source, 3)
    return data_source


if __name__ == "__main__":
    unit_test()
    print("test success")
