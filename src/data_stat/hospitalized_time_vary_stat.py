import csv
from itertools import islice
import datetime


# 得到病人第一次入院和最后一次入院的时间，入院的总次数
def load_first_and_last_admission(file_path):
    patient_first_admission_map = {}
    patient_last_admission_map = {}
    with open(file_path, 'r', encoding="gbk", newline="") as source_data:
        csv_reader = csv.reader(source_data)
        for line in islice(csv_reader, 1, None):
            pat_id, visit_id, admission_date, discharge_date, military_flag = line
            content = [visit_id, admission_date]
            if not patient_first_admission_map.__contains__(pat_id):
                patient_first_admission_map[pat_id] = content
                patient_last_admission_map[pat_id] = content
            else:
                if int(visit_id) < int(patient_first_admission_map[pat_id][0]):
                    patient_first_admission_map[pat_id] = content
                if int(visit_id) > int(patient_last_admission_map[pat_id][0]):
                    patient_last_admission_map[pat_id] = content
    return patient_first_admission_map, patient_last_admission_map


# 得到病人入院次数
def load_visit_count(file_path):
    visit = {}
    with open(file_path, 'r', encoding="gbk", newline="") as source_data:
        csv_reader = csv.reader(source_data)
        for line in islice(csv_reader, 1, None):
            pat_id, visit_id, admission_date, discharge_date, military_flag = line
            if not visit.__contains__(pat_id):
                visit[pat_id] = 1
            else:
                visit[pat_id] += 1
    return visit


# 输出住院次数大于某个阈值的病人的名单
def patient_list_output(threshold, data_source, file_name, save_path="..\\..\\resource\\stat\\patient_list\\"):
    patient_list = []
    for patient_id in data_source:
        if data_source[patient_id] >= threshold:
            patient_list.append([patient_id])
    with open(save_path+file_name, 'w', encoding='utf-8-sig', newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(patient_list)


# 输出住院次数大于某个阈值的病人的纵向时间跨度
def patient_time_vary(threshold, count, time_vary, file_name,  save_path=
                      "..\\..\\resource\\stat\\time_vary_distribution\\"):
    patient_time_vary_list = []
    for patient_id in count:
        if count[patient_id] >= threshold:
            patient_time_vary_list.append([patient_id, time_vary[patient_id]])
    with open(save_path + file_name, 'w', encoding='utf-8-sig', newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(patient_time_vary_list)


# 统计病人住院纵向数据时间跨度和住院次数
def main():
    # 载入数据
    source_data_path = "..\\..\\resource\\hospitalized_patient_visit_admission_date.csv"
    first_admission, last_admission = load_first_and_last_admission(source_data_path)
    patient_visit_count = load_visit_count(source_data_path)

    visit_distribution = {}
    patient_duration = {}

    # 统计住院次数分布
    for patient_id in first_admission:
        visit_count = int(patient_visit_count[patient_id])
        if not visit_distribution.__contains__(visit_count):
            visit_distribution[visit_count] = 1
        else:
            visit_distribution[visit_count] += 1
    # 统计病人住院长期纵向数据时间跨度
    for patient_id in first_admission:
        first_admission_date = datetime.datetime.strptime(first_admission[patient_id][1], "%Y/%m/%d %H:%M:%S")
        last_admission_date = datetime.datetime.strptime(last_admission[patient_id][1], "%Y/%m/%d %H:%M:%S")
        patient_duration[patient_id] = (last_admission_date - first_admission_date).days

    # 输出住院次数为某个数量的统计结果
    count_path = "..\\..\\resource\\stat\\hospitalized_visit_count_stat.csv"
    with open(count_path, 'w', encoding="utf-8-sig", newline="") as visit_count_stat:
        csv_writer = csv.writer(visit_count_stat)
        for item in visit_distribution:
            csv_writer.writerows([[item, visit_distribution[item]]])

    # 输出不同阈值下，病人长期纵向数据的时间分布和清单
    # 阈值指：住院时间至少大于n次，才能称为有长期纵向数据
    for item in range(3, 10):
        patient_list_output(item, patient_visit_count, "patient_list_count_greater_than_" + str(item) + ".csv")
        patient_time_vary(item, patient_visit_count, patient_duration, "patient_time_vary_count_greater_than_" +
                          str(item) + ".csv")


if __name__ == "__main__":
    main()

