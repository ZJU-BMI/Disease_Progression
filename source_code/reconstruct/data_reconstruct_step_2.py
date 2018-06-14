import csv
from dateutil import parser
from itertools import islice

# 将所有人的诊断组合，按照顺序排列，形成Encounter相应的Visit表
# 需要指出，Encounter和Visit并非一一对应，在数据清洗过程中，发现有些病人存在
# 一次疾病，重复入院的现象。因此做两条规则
# 1.两次入院时间相邻在30天以内的，视为同一入院，赋予一样的Visit_ID
# 2.两次入院时间在30天以上的，但是共享入院号的，赋予同样的ID
# 3.在住院过程中，若发生新的Encounter ID，则新的Encounter赋予同样的Visit_ID
# 经过如上处理，大约还会有20多个非法数据，主要表征是，在点A以encounter id a就诊，


# 之后在点B用ID号b就诊，之后再在点C用ID号a就诊。这种数据直接抛弃
def main():
    patient_encounter_map = {}
    encounter_patient_map = {}

    # 也就是Step 3的结果，整合EMR和结构化数据后的Encounter Patient映射
    with open('..\\resource\\encounter_patient_table.csv', 'r', encoding='utf-8-sig',
              newline="") as encounter_patient_table:
        map_table = csv.reader(encounter_patient_table)
        for line in islice(map_table, 1, None):
            encounter_id, patient_id = line
            encounter_patient_map[encounter_id] = patient_id
            if not patient_encounter_map.__contains__(patient_id):
                patient_encounter_map[patient_id] = []

    table_counter = 0
    with open('..\\resource\\patient_diagnosis_table_record.csv', 'r', encoding='utf-8-sig',
              newline="") as patient_table:
        table_reader = csv.reader(patient_table)

        for line in islice(table_reader, 1, None):
            patient_id = line[0]
            encounter_id = line[1]
            encounter_date = line[2]
            # 将字符串类型的时间转换为Date类型
            encounter_date = encounter_date.split(" ")[0]
            encounter_date = parser.parse(encounter_date)
            patient_encounter_map[patient_id].append([encounter_date, "T", encounter_id])

            # 测试用
            # table_counter += 1
            # if table_counter > 10000:
            #    break

    emr_counter = 0
    with open('..\\resource\\patient_inhospital_record.csv', 'r', encoding='utf-8-sig',
              newline="") as patient_emr:
        table_reader = csv.reader(patient_emr)

        for line in islice(table_reader, 1, None):
            encounter_id = line[1]
            admission = line[3]
            discharge = line[4]

            admission = admission.split(" ")[0]
            admission = admission.replace('年', '-')
            admission = admission.replace('月', '-')
            admission = admission.replace('日', '')
            admission = parser.parse(admission)
            discharge = discharge.split(" ")[0]
            discharge = discharge.replace('年', '-')
            discharge = discharge.replace('月', '-')
            discharge = discharge.replace('日', '')
            discharge = parser.parse(discharge)

            patient_id = encounter_patient_map[encounter_id]
            patient_encounter_map[patient_id].append([admission, "A", encounter_id])
            patient_encounter_map[patient_id].append([discharge, "D", encounter_id])

            # 测试用
            # emr_counter += 1
            # if emr_counter > 10000:
            #     break

    print("Data Loaded")

    visit_reconstruct = {}
    # 把各个Encounter按照顺序重排
    for patient_id in patient_encounter_map:
        patient_list = patient_encounter_map[patient_id]
        patient_list = sorted(patient_list, key=lambda x: x[0])
        visit_reconstruct[patient_id] = patient_list

    # 给定每个病人入院的Visit ID号
    # 规则，如果两次入院的注册时间间隔少于10天，归为一次入院
    # 如果某次入院时间的落在EMR标记的住院期间内，归为同一次入院
    for patient_id in visit_reconstruct:
        encounter_list = visit_reconstruct[patient_id]
        visit_id = 0
        last_time = parser.parse('2000-01-01')
        last_encounter_id = "YunCaiYue"
        # current attr有两种状态，第一种代指在住院中发生新的Encounter
        # 用A表示，第二种代表正常，用B表示。在A时，Visit ID锁死
        # 之前的预处理使得每次住院一定有入院时间和出院时间
        current_attr = "B"
        for visit_tuple in encounter_list:
            date, attr, encounter_id = visit_tuple

            # 给定访问号码
            if (date-last_time).days > 30 and current_attr != "A" and \
                    last_encounter_id != encounter_id:
                visit_id += 1
                visit_tuple.append(visit_id)
            else:
                visit_tuple.append(visit_id)
            last_time = date
            last_encounter_id = encounter_id

            # 若为住院锁死Visit_ID
            if attr == 'A':
                current_attr = 'A'
            if attr == 'D':
                current_attr = 'B'

    with open('..\\reconstruct_data\\visit.csv', 'w', encoding='utf-8-sig', newline="") as visit:
        visit_writer = csv.writer(visit)
        visit_writer.writerow(["patient_id", "create_date", "admission_type",
                               "encounter_id", "visit_id"])
        for pat_id in visit_reconstruct:
            for item in visit_reconstruct[pat_id]:
                visit_writer.writerow([pat_id, item[0], item[1], item[2],
                                       item[3]])


if __name__ == "__main__":
    main()

