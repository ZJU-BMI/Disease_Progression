import csv
import datetime
from itertools import islice


# 载入病人性别，民族，出生日期信息
def load_patient_info(file_path, encoding="gbk", newline=""):
    patient_info_map = {}
    with open(file_path, "r", encoding=encoding, newline=newline) as patient_info_file:
        csv_reader = csv.reader(patient_info_file)
        for line in islice(csv_reader, 1, None):
            patient_id, sex, birthday, ethnic_group = line
            # 部分数据有错，没有Birthday信息，直接丢弃
            if len(birthday) < 2:
                continue
            birthday = datetime.datetime.strptime(birthday, "%Y/%m/%d")
            patient_info_map[patient_id] = dict({"sex": sex, "birthday": birthday, "ethnic_group": ethnic_group})
    return patient_info_map


# 载入门诊病人诊断信息
def load_outpatient_diagnosis(file_path, encoding="gbk", newline=""):
    outpatient_diagnosis_map = {}
    with open(file_path, 'r', encoding=encoding, newline=newline) as outpatient_file:
        csv_reader = csv.reader(outpatient_file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_date, visit_no, diagnosis_description = line
            visit_date = datetime.datetime.strptime(visit_date, "%Y/%m/%d")
            if outpatient_diagnosis_map.__contains__(patient_id):
                outpatient_diagnosis_map[patient_id][visit_no] = {"visit_date": visit_date,
                                                                  "diagnosis_description": diagnosis_description}
            else:
                outpatient_diagnosis_map[patient_id] = {}
                outpatient_diagnosis_map[patient_id][visit_no] = {"visit_date": visit_date,
                                                                  "diagnosis_description": diagnosis_description}
    return outpatient_diagnosis_map


# 载入手术相关信息
def load_operation(file_path, encoding="gbk", newline=""):
    operation_map = {}
    with open(file_path, 'r', encoding=encoding, newline=newline) as operation_file:
        csv_reader = csv.reader(operation_file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, operation_no, operation_description, icd_9, heal, operation_date = line

            # 原始数据日期格式不一致，一种是年月日时分，一种是年月日
            # 若日期格式为年月日时分，长度必然大于10；若为年月日，长度必然小于等于10
            if len(operation_date) < 2:
                continue
            elif len(operation_date) > 10:
                operation_date = datetime.datetime.strptime(operation_date, "%Y/%m/%d %H:%M:%S")
            else:
                operation_date = datetime.datetime.strptime(operation_date, "%Y/%m/%d")

            content = {"icd_code": icd_9, "operation_date": operation_date,  "operation_description":
                       operation_description,  "heal": heal, }
            if operation_map.__contains__(patient_id) and operation_map[patient_id].__contains__(visit_id):
                operation_map[patient_id][visit_id][operation_no] = content
            if not operation_map.__contains__(patient_id):
                operation_map[patient_id] = {}
                operation_map[patient_id][visit_id] = {}
                operation_map[patient_id][visit_id][operation_no] = content
            if operation_map.__contains__(patient_id) and not operation_map[patient_id].__contains__(visit_id):
                operation_map[patient_id][visit_id] = {}
                operation_map[patient_id][visit_id][operation_no] = content
    return operation_map


# 载入住院病人诊断信息
def load_hospitalized_patient_diagnosis(file_path, encoding="gbk", newline=""):
    diagnosis_map = {}
    with open(file_path, 'r', encoding=encoding, newline=newline) as hospitalized_patient_diagnosis_file:
        csv_reader = csv.reader(hospitalized_patient_diagnosis_file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, diagnosis_type, diagnosis_no, diagnosis_desc, icd_code, icd_version, \
             diagnosis_date, treat_day, treat_result = line
            # 原始数据日期格式不一致，一种是年月日时分，一种是年月日
            # 若日期格式为年月日时分，长度必然大于10；若为年月日，长度必然小于等于10
            if len(diagnosis_date) < 2:
                continue
            elif len(diagnosis_date) > 10:
                diagnosis_date = datetime.datetime.strptime(diagnosis_date, "%Y/%m/%d %H:%M:%S")
            else:
                diagnosis_date = datetime.datetime.strptime(diagnosis_date, "%Y/%m/%d")

            if diagnosis_type == "3":
                diagnosis_type = "出院主要诊断"
            elif diagnosis_type == "A":
                diagnosis_type = "出院其他诊断"

            content = {"diagnosis_type": diagnosis_type, "diagnosis_desc": diagnosis_desc, "icd_code": icd_code,
                       "icd_version": icd_version, "diagnosis_date": diagnosis_date, "treat_day": treat_day,
                       "treat_result": treat_result}
            if diagnosis_map.__contains__(patient_id) and diagnosis_map[patient_id].__contains__(visit_id):
                diagnosis_map[patient_id][visit_id][diagnosis_no] = content
            if not diagnosis_map.__contains__(patient_id):
                diagnosis_map[patient_id] = {}
                diagnosis_map[patient_id][visit_id] = {}
                diagnosis_map[patient_id][visit_id][diagnosis_no] = content
            if diagnosis_map.__contains__(patient_id) and not diagnosis_map[patient_id].__contains__(visit_id):
                diagnosis_map[patient_id][visit_id] = {}
                diagnosis_map[patient_id][visit_id][diagnosis_no] = content
    return diagnosis_map


# 载入住院病人入院时间信息
def load_hospitalized_patient_visit_admission_date(file_path, encoding="gbk", newline=""):
    admission_map = {}
    with open(file_path, 'r', encoding=encoding, newline=newline) as hospitalized_patient_admission_file:
        csv_reader = csv.reader(hospitalized_patient_admission_file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, admission_date, discharge_date, military_flag = line

            # 原始数据日期格式不一致，一种是年月日时分，一种是年月日
            # 若日期格式为年月日时分，长度必然大于10；若为年月日，长度必然小于等于10
            if len(admission_date) < 2:
                continue
            elif len(admission_date) > 10:
                admission_date = datetime.datetime.strptime(admission_date, "%Y/%m/%d %H:%M:%S")
            else:
                admission_date = datetime.datetime.strptime(admission_date, "%Y/%m/%d")
            if len(discharge_date) < 2:
                continue
            elif len(discharge_date) > 10:
                discharge_date = datetime.datetime.strptime(discharge_date, "%Y/%m/%d %H:%M:%S")
            else:
                discharge_date = datetime.datetime.strptime(discharge_date, "%Y/%m/%d")

            content = [admission_date, discharge_date, military_flag]
            if admission_map.__contains__(patient_id):
                admission_map[patient_id][visit_id] = content
            if not admission_map.__contains__(patient_id):
                admission_map[patient_id] = {}
                admission_map[patient_id][visit_id] = content
    return admission_map


def load_all_data(patient_info_file_path, outpatient_diagnosis_file_path, operation_info_file_path,
                  hospitalized_patient_diagnosis_file_path, hospitalized_patient_visit_file_path):
    """
    :param patient_info_file_path:
    :param outpatient_diagnosis_file_path:
    :param operation_info_file_path:
    :param hospitalized_patient_diagnosis_file_path:
    :param hospitalized_patient_visit_file_path:
    :return: data_source,包含五类数据
    """
    patient_info_map = load_patient_info(patient_info_file_path)
    outpatient_diagnosis_map = load_outpatient_diagnosis(outpatient_diagnosis_file_path)
    operation_map = load_operation(operation_info_file_path)
    hospitalized_patient_diagnosis_map = load_hospitalized_patient_diagnosis(hospitalized_patient_diagnosis_file_path)
    hospitalized_patient_admission_map = load_hospitalized_patient_visit_admission_date(
        hospitalized_patient_visit_file_path)
    data_source = {'patient_info_map': patient_info_map, 'patient_visit_map': hospitalized_patient_admission_map,
                   'patient_diagnosis': hospitalized_patient_diagnosis_map, 'operation': operation_map,
                   'outpatient_diagnosis': outpatient_diagnosis_map}
    return data_source


def unit_test():
    patient_info_file_path = "..\\..\\resource\\patient_info.csv"
    outpatient_diagnosis_file_path = "..\\..\\resource\\outpatient_diagnosis.csv"
    operation_info_file_path = "..\\..\\resource\\operation.csv"
    hospitalized_patient_diagnosis_file_path = "..\\..\\resource\\hospitalized_patient_diagnosis.csv"
    hospitalized_patient_visit_file_path = "..\\..\\resource\\hospitalized_patient_visit_admission_date.csv"
    data_source = load_all_data(patient_info_file_path, outpatient_diagnosis_file_path, operation_info_file_path,
                                hospitalized_patient_diagnosis_file_path, hospitalized_patient_visit_file_path)
    print(data_source)
    print("Data Load Success")


if __name__ == "__main__":
    unit_test()
