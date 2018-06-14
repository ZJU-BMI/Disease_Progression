import csv
from itertools import islice
# 根据先前的所有结果，重整出含有诊断和Visit信息的病人病情发展序列
# 此处考虑过门诊是否存在日康手术的可能性，最终决定还是不考虑日康手术


def main():
    encounter_patient_map = {}
    encounter_visit_map = {}
    encounter_other_info_map = {}
    encounter_diagnosis = {}
    patient_visit_count = {}
    general_map = {}

    # 加载数据
    with open('..\\resource\\patient_inhospital_record.csv', 'r', encoding='utf-8-sig',
              newline="") as patient_emr:
        csv_reader = csv.reader(patient_emr)
        for line in islice(csv_reader, 1, None):
            pat_id, encounter_id, cate, admission, discharge, main_diag, \
             other_diag, department, operation, death = line
            if other_diag != 'No_Other_diagnosis':
                diagnosis = main_diag+"$"+other_diag
            else:
                diagnosis = main_diag

            encounter_diagnosis[encounter_id] = diagnosis
            encounter_other_info_map[encounter_id] = [department, operation, death]

    with open('..\\resource\\patient_diagnosis_table_record.csv', 'r', encoding='utf-8-sig',
              newline="") as patient_diagnosis:
        csv_reader = csv.reader(patient_diagnosis)
        for line in islice(csv_reader, 1, None):
            _, encounter_id, _, diagnosis, _, code, dept_name = line
            if encounter_diagnosis.__contains__(encounter_id):
                origin_diagnosis = encounter_diagnosis[encounter_id]
                encounter_diagnosis[encounter_id] = origin_diagnosis+"$"+diagnosis
            else:
                encounter_diagnosis[encounter_id] = diagnosis
            if not encounter_other_info_map.__contains__(encounter_id):
                encounter_other_info_map[encounter_id] = [dept_name,
                                                          "no_operation_record",
                                                          "no_discharge_record"]

    # 计算每个病人的入院总次数
    with open('..\\reconstruct_data\\visit.csv', 'r', encoding='utf-8-sig', newline="") as \
            visit_csv:
        csv_reader = csv.reader(visit_csv)
        for line in islice(csv_reader, 1, None):
            patient_id, time, _, encounter_id, visit_id = line
            if patient_visit_count.__contains__(patient_id):
                if int(visit_id) > patient_visit_count[patient_id]:
                    patient_visit_count[patient_id] = int(visit_id)
            else:
                patient_visit_count[patient_id] = 0

    with open('..\\reconstruct_data\\visit.csv', 'r', encoding='utf-8-sig', newline="") as \
            visit_csv:
        csv_reader = csv.reader(visit_csv)
        for line in islice(csv_reader, 1, None):
            patient_id, time, _, encounter_id, visit_id = line
            encounter_visit_map[encounter_id] = visit_id
            encounter_patient_map[encounter_id] = patient_id
            info_map = encounter_other_info_map[encounter_id]

            if not general_map.__contains__(patient_id):
                general_map[patient_id] = {}

            general_map[patient_id][visit_id] = [time, encounter_id,
                                                 info_map[0], info_map[1],
                                                 info_map[2]]

    for encounter_id in encounter_diagnosis:
        diagnosis = encounter_diagnosis[encounter_id]
        correspond_pat_id = encounter_patient_map[encounter_id]
        correspond_visit_id = encounter_visit_map[encounter_id]
        visit_info = general_map[correspond_pat_id][correspond_visit_id]

        if len(visit_info) == 5:
            visit_info.append(diagnosis)
        elif len(visit_info) == 6:
            new_diagnosis = visit_info[5]+"$"+diagnosis
            visit_info[5] = new_diagnosis
        else:
            print("illegal length")

    with open('..\\reconstruct_data\\reconstruct_data_unstructured.csv', 'w',
              encoding='utf-8-sig', newline="") as final:
        csv_writer = csv.writer(final)
        csv_writer.writerow(["patient_id", "visit_id", "encounter_id", "time",
                             "diagnosis", "department", "operation",
                             "discharge_type"])
        for patient_id in general_map:
            visit_map = general_map[patient_id]
            for visit_id in visit_map:
                if len(visit_map[visit_id]) == 6:
                    time, encounter_id, department, operation, death, diagnosis\
                        = visit_map[visit_id]
                    if patient_visit_count[patient_id] > 4:
                        csv_writer.writerow([patient_id, visit_id,
                                             encounter_id, time, diagnosis,
                                             department, operation, death])
                else:
                    print(patient_id+"_"+visit_id+"_"+visit_map[visit_id][0],
                          visit_map[visit_id][1])

    print("Finish")


if __name__ == "__main__":
    main()
