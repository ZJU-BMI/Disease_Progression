import numpy as np
import datetime
import csv
import math

"""
Toy Hawkes
Toy在两方面
1.数据并不完整，只使用了ICD10编码的，没有门诊数据的，未做任何清洗的，分割粒度取前三位的，入院至少三次的病人的纵向数据
2.Hawkes过程建模本身较为简单，使用的是最基础的，conditional intensity: lambda_u = mu_u + sum[alpha*exp(t-t_i)]的多元
  Hawkes过程
"""


def main():
    # Load Data
    xml_file_path = "..\\..\\resource\\reconstruct_data\\zero_level_reconstruction.xml"
    diagnosis_size = 10
    operation_size = 5
    data_source = load_need_data(xml_file_path, diagnosis_reserve=diagnosis_size, operation_reserve=operation_size)
    event_data = event_list_reconstruction(data_source)
    diagnosis_reserve = data_source['diagnosis_size']
    operation_reserve = data_source['operation_size']
    # hyper parameter setting

    # parameter initialize
    omega = 1
    base_intensity, mutual_exciting_coefficient = initialize_parameter(diagnosis_reserve+operation_reserve)

    # EM Algorithm
    for times in range(0, 50):
        print("iter " + str(times))
        # E Step
        auxiliary_variable = update_auxiliary_variable(base_intensity, mutual_exciting_coefficient, omega,
                                                       event_data,  diagnosis_size)
        # M Step
        base_intensity = update_parameter_mu(base_intensity, event_data, auxiliary_variable, diagnosis_size)
        mutual_exciting_coefficient = update_parameter_alpha(mutual_exciting_coefficient, omega, event_data,
                                                             auxiliary_variable, diagnosis_size)

    # output_result
    base_intensity = base_intensity.tolist()
    mutual_exciting_coefficient = mutual_exciting_coefficient.tolist()
    with open("base_intensity.csv", 'w', encoding='utf-8-sig', newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(base_intensity)
    with open("mutual_exciting_coefficient.csv", 'w', encoding='utf-8-sig', newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(mutual_exciting_coefficient)





def event_list_reconstruction(data_source):
    diagnosis_code_rank_map = data_source['diagnosis_rank']
    operation_code_rank_map = data_source['operation_rank']
    patient_visit_date_map = data_source['visit_date']
    visit_diagnosis_map = data_source['visit_diagnosis']
    visit_operation_map = data_source['visit_operation']

    patient_event_list = {}
    for patient in patient_visit_date_map:
        patient_event_list[patient] = {}
        event_list = []

        for visit in patient_visit_date_map[patient]:
            visit_date = datetime.datetime.strptime(patient_visit_date_map[patient][visit], "%Y-%m-%d %H:%M:%S")

            if visit_diagnosis_map.__contains__(patient) and visit_diagnosis_map[patient].__contains__(visit):
                diagnosis_list = visit_diagnosis_map[patient][visit]
                for item in diagnosis_list:
                    event_list.append((diagnosis_code_rank_map[item]+"_D", visit_date))
            if visit_operation_map.__contains__(patient) and visit_operation_map[patient].__contains__(visit):
                # Data Structure [operation_code, operation_time]
                operation_list = visit_operation_map[patient][visit]
                for item in operation_list:
                    item_rank = operation_code_rank_map[item[0]]+"_O"
                    operation_date = datetime.datetime.strptime(item[1], "%Y-%m-%d %H:%M:%S")
                    event_list.append((item_rank, operation_date))

        event_list = sorted(event_list, key=lambda data_tuple: data_tuple[1])
        patient_event_list[patient] = event_list
    return patient_event_list





def initialize_parameter(event_size, initial_strategy='default'):
    """
    参数初始化策略，当前只支持default一种，不同的初始化策略留待以后再加
    :param event_size:
    :param initial_strategy:
    :return: base intensity 和 核函数的系数
    """
    base_intensity = 0
    mutual_exciting_coefficient = 0

    if initial_strategy == 'default':
        base_intensity = np.random.normal(0, 1, [event_size, 1])
        mutual_exciting_coefficient = np.random.normal(0, 1, [event_size, event_size])

    return base_intensity, mutual_exciting_coefficient


def update_auxiliary_variable(base_intensity, mutual_exciting_coefficient, omega, full_event_list, diagnosis_size):
    auxiliary_variable = {}
    for patient in full_event_list:
        auxiliary_variable[patient] = []
        event_list = full_event_list[patient]
        for c_index in range(len(event_list)):
            index_list = []
            for p_index in range(0, c_index+1):
                # 计算q_ii
                # 当c_index == p_index时轮到，因此这个数据会在index_list的最后
                if p_index == c_index:
                    q_ii = calculate_q_ii(event_list, c_index, base_intensity, mutual_exciting_coefficient, omega,
                                          diagnosis_size)
                    index_list.append(q_ii)
                # 计算q_ij
                else:
                    q_ij = calculate_q_ij(event_list, c_index, p_index, base_intensity, mutual_exciting_coefficient,
                                          omega, diagnosis_size)
                    index_list.append(q_ij)
            auxiliary_variable[patient].append(index_list)
    # 由于Hawkes的特点，每个auxiliary_variable的数据结构直观上会呈现一个类似三角形的样子
    return auxiliary_variable


def calculate_q_ii(event_list, c_index, base_intensity, mutual_exciting_coefficient, omega, diagnosis_size):
    # current event info
    c_event_id_tuple, c_event_time = event_list[c_index]
    c_event_rank, c_event_type = c_event_id_tuple.split("_")
    # rank是从1起算的，因此这一部分需要把额外的1减掉，才能对应上base_intensity数组中的index
    c_event_index = int(c_event_rank)-1
    if c_event_type == "O":
        c_event_index = c_event_index + diagnosis_size
    c_mu = base_intensity[c_event_index][0]

    p_ii = c_mu / calculate_q_denominator(event_list, c_index, base_intensity, mutual_exciting_coefficient, omega,
                                          diagnosis_size)
    return p_ii


def calculate_q_ij(event_list, c_index, p_index, base_intensity, mutual_exciting_coefficient, omega, diagnosis_size):
    # current event info
    c_event_id_tuple, c_event_time = event_list[c_index]
    c_event_rank, c_event_type = c_event_id_tuple.split("_")
    c_event_index = int(c_event_rank)-1
    # last event info
    p_event_id_tuple, p_event_time = event_list[p_index]
    p_event_rank, p_event_type = p_event_id_tuple.split("_")
    p_event_index = int(p_event_rank)-1

    if c_event_type == "O":
        c_event_index = c_event_index + diagnosis_size
    if p_event_type == "O":
        p_event_index = p_event_index + diagnosis_size
    c_alpha = mutual_exciting_coefficient[c_event_index][p_event_index]

    time_differ = (c_event_time-p_event_time).days
    if time_differ != 0:
        nominator = c_alpha * math.exp(-1 * omega * time_differ)
        q_ij = nominator / calculate_q_denominator(event_list, c_index, base_intensity, mutual_exciting_coefficient,
                                                   omega,
                                                   diagnosis_size)
    else:
        q_ij = 0

    return q_ij


def calculate_q_denominator(event_list, c_index, base_intensity, mutual_exciting_coefficient, omega, diagnosis_size):
    # current event info i.e. mu
    c_event_id_tuple, c_event_time = event_list[c_index]
    c_event_rank, c_event_type = c_event_id_tuple.split("_")
    c_event_index = int(c_event_rank)-1
    if c_event_type == "O":
        c_event_index = c_event_index + diagnosis_size
    c_mu = base_intensity[c_event_index]

    # previous event info
    p_sum = 0
    for p_index in range(0, c_index):
        p_event_id_tuple, p_event_time = event_list[p_index]
        p_event_rank, p_event_type = p_event_id_tuple.split("_")
        p_event_index = int(p_event_rank) - 1
        if p_event_type == "O":
            p_event_index = p_event_index + diagnosis_size

        # 我们的数据和标准Hawkes过程不太一样，标准的点过程是没有同一时刻发生多个事件的，但是我们的有，
        # 前一个事件和后一个事件可能同时发生，因此要做检验
        if (c_event_time-p_event_time).days > 0:
            p_alpha = mutual_exciting_coefficient[c_event_index][p_event_index]
            time_differ = (c_event_time - p_event_time).days
            p_sum += p_alpha * math.exp(-1 * omega * time_differ)
    denominator = p_sum + c_mu
    return denominator


def update_parameter_mu(base_intensity, full_event_list, auxiliary_variable, diagnosis_size):
    base_intensity_length = base_intensity.shape[0]
    for index in range(0, base_intensity_length):
        nominator = 0
        denominator = 0

        # calculate nominator
        for patient in full_event_list:
            single_event_list = full_event_list[patient]
            event_list_length = len(single_event_list)
            for event_index in range(0, event_list_length):
                # 确认第event_index的event的类型
                event_id = single_event_list[event_index][0]
                event_type_rank, event_type = event_id.split("_")
                event_type_index = int(event_type_rank)-1
                if event_type == "O":
                    event_type_index = event_type_index + diagnosis_size

                if index == event_type_index:
                    nominator += auxiliary_variable[patient][event_index][-1]

        # calculate denominator
        for patient in full_event_list:
            single_event_list = full_event_list[patient]
            if len(single_event_list) == 0:
                continue
            denominator += (single_event_list[-1][1] - single_event_list[0][1]).days

        mu = nominator/denominator
        base_intensity[index][0] = mu
    return base_intensity


def update_parameter_alpha(mutual_exciting_coefficient, omega, full_event_list, auxiliary_variable, diagnosis_size):
    length = mutual_exciting_coefficient.shape[0]
    for row_index in range(0, length):
        for col_index in range(0, length):
            nominator = 0
            denominator = 0

            # nominator calculate
            for patient in full_event_list:
                event_list = full_event_list[patient]
                event_list_length = len(event_list)
                for c_event_index in range(0, event_list_length):
                    c_event_id_rank, c_event_type = event_list[c_event_index][0].split("_")
                    c_event_id_index = int(c_event_id_rank)-1
                    if c_event_type == "O":
                        c_event_id_index += diagnosis_size

                    for p_event_index in range(0, c_event_index):
                        p_event_id_rank, p_event_type = event_list[p_event_index][0].split("_")
                        p_event_id_index = int(p_event_id_rank) - 1
                        if p_event_type == "O":
                            p_event_id_index += diagnosis_size

                        if row_index == c_event_id_index and col_index == p_event_id_index:
                            nominator += auxiliary_variable[patient][c_event_index][p_event_index]

            # denominator calculate
            for patient in full_event_list:
                event_list = full_event_list[patient]
                event_list_length = len(event_list)
                if event_list_length == 0:
                    continue
                time_vary = (event_list[-1][1]-event_list[0][1]).days
                for u_index in range(0, length):
                    for index in range(0, event_list_length):
                        event_id_rank, event_type = event_list[index][0].split("_")
                        event_id_index = int(event_id_rank) - 1
                        event_time = (event_list[index][1]-event_list[0][1]).days
                        if event_type == "O":
                            event_id_index += diagnosis_size
                        if row_index == u_index and col_index == event_id_index:
                            term = (1-math.exp(-1*omega*(time_vary-event_time)))/omega
                            denominator += term
            alpha = nominator/denominator
            mutual_exciting_coefficient[row_index][col_index] = alpha
    return mutual_exciting_coefficient


# 尚未完成
def calculate_log_likelihood():
    pass


if __name__ == "__main__":
    main()
