# coding=utf-8
import csv
import os

import mimic.derive_training_data as dtd
from hawkes.hawkes_process import Hawkes


def hawkes_load_data(file_path, file_name, diagnosis_reserve, procedure_reserve):
    # load data
    data_sequence_info, name_index_map = dtd.hawkes(reserve_diagnosis=diagnosis_reserve,
                                                    reserve_procedure=procedure_reserve,
                                                    file_path=file_path,
                                                    file_name=file_name)
    # derive training data and test data
    index = 0
    test_event_sequence_map = None
    train_event_sequence_map = dict()
    for key in data_sequence_info:
        if index == 0:
            test_event_sequence_map = data_sequence_info[key]
        else:
            train_event_sequence_map.update(data_sequence_info[key])
        index += 1

    return train_event_sequence_map, test_event_sequence_map, name_index_map


def hawkes_save_name_index_map(file_path, file_name, name_index_map):
    with open(file_path + file_name, 'w', encoding='utf-8-sig', newline="") as f:
        csv_writer = csv.writer(f)
        for index in name_index_map:
            csv_writer.writerows([[index, name_index_map[index]]])


def save_result(return_data_map, file_path, name_prefix):
    train = name_prefix + 'train_log_likelihood.csv'
    test = name_prefix + 'test_log_likelihood.csv'
    mutual_intensity = name_prefix + 'mutual_intensity.csv'
    base_intensity = name_prefix + 'base_intensity.csv'
    auxiliary_variable = name_prefix + 'auxiliary_variable.csv'
    k_omega = name_prefix + 'k_omega.csv'
    y_omega = name_prefix + 'y_omega.csv'
    event_number_each_slot = name_prefix + 'event_number_each_slot.csv'
    event_count_list = name_prefix + 'count_of_each_event.csv'

    with open(os.path.join(file_path, train), 'w', encoding='utf-8-sig', newline="") as f:
        csv_writer = csv.writer(f)
        train_log_likelihood_tendency = return_data_map['train_log_likelihood_tendency']
        for i in range(0, len(train_log_likelihood_tendency)):
            csv_writer.writerows([[i, train_log_likelihood_tendency[i]]])

    with open(os.path.join(file_path, test), 'w', encoding='utf-8-sig', newline="") as f:
        csv_writer = csv.writer(f)
        test_log_likelihood_tendency = return_data_map['test_log_likelihood_tendency']
        for i in range(0, len(test_log_likelihood_tendency)):
            csv_writer.writerows([[i, test_log_likelihood_tendency[i]]])

    with open(os.path.join(file_path, mutual_intensity), 'w', encoding='utf-8-sig', newline="") as f:
        csv_writer = csv.writer(f)
        mutual_intensity_matrix = []
        mutual_intensity_data = return_data_map['mutual_intensity']
        for i in range(0, len(mutual_intensity_data)):
            row = []
            for j in range(0, len(mutual_intensity_data[i])):
                row.append(mutual_intensity_data[i][j])
            mutual_intensity_matrix.append(row)
        csv_writer.writerows(mutual_intensity_matrix)

    with open(os.path.join(file_path, base_intensity), 'w', encoding='utf-8-sig', newline="") as f:
        csv_writer = csv.writer(f)
        base_intensity_vector = []
        base_intensity_data = return_data_map['base_intensity']
        for i in range(0, len(base_intensity_data)):
            base_intensity_vector.append(base_intensity_data[i][0])
        csv_writer.writerows([base_intensity_vector])

    with open(os.path.join(file_path, auxiliary_variable), 'w', encoding='utf-8-sig', newline="") as f:
        csv_writer = csv.writer(f)
        auxiliary_variable_data = return_data_map['auxiliary_variable']
        for sequence_id in auxiliary_variable_data:
            for event_no in auxiliary_variable_data[sequence_id]:
                sequence_trigger_list = [sequence_id, event_no]
                for item in auxiliary_variable_data[sequence_id][event_no]:
                    sequence_trigger_list.append(item)
                csv_writer.writerows([sequence_trigger_list])

    if return_data_map['kernel'] == 'Fourier' or return_data_map['kernel'] == 'fourier':
        with open(os.path.join(file_path, k_omega), 'w', encoding='utf-8-sig', newline="") as f:
            csv_writer = csv.writer(f)
            k_omega_list = []
            k_omega_data = return_data_map['k_omega']
            for slot in range(0, len(k_omega_data)):
                k_omega_list.append(k_omega_data[slot])
            csv_writer.writerows([k_omega_list])

        with open(os.path.join(file_path, y_omega), 'w', encoding='utf-8-sig', newline="") as f:
            csv_writer = csv.writer(f)
            y_omega_list = []
            y_omega_data = return_data_map['y_omega']
            for slot in range(0, len(y_omega_data)):
                y_omega_list.append(y_omega_data[slot])
            csv_writer.writerows([y_omega_list])

        with open(os.path.join(file_path, event_count_list), 'w', encoding='utf-8-sig', newline="") as f:
            csv_writer = csv.writer(f)
            event_count_list = return_data_map['count_of_each_event']
            csv_writer.writerows([event_count_list])

        with open(os.path.join(file_path, event_number_each_slot), 'w', encoding='utf-8-sig', newline="") as f:
            csv_writer = csv.writer(f)
            count_of_each_slot = return_data_map['count_of_each_slot']
            csv_writer.writerows([count_of_each_slot])


def hawkes_optimization(train_data, test_data, iteration, diagnosis_reserve, procedure_reserve, kernel,
                        time_slot, ):
    """
    :param train_data:
    :param test_data:
    :param iteration:
    :param diagnosis_reserve:
    :param procedure_reserve:
    :param kernel:
    :param time_slot:
    :return:
    """

    event_sum = diagnosis_reserve + procedure_reserve

    hawkes_process = Hawkes(training_data=train_data, test_data=test_data, event_count=event_sum, kernel=kernel,
                            init_strategy='default', time_slot=time_slot)
    hawkes_process.optimization(iteration)

    return_data_map = dict()
    return_data_map['train_log_likelihood_tendency'] = hawkes_process.train_log_likelihood_tendency
    return_data_map['test_log_likelihood_tendency'] = hawkes_process.test_log_likelihood_tendency
    return_data_map['mutual_intensity'] = hawkes_process.mutual_intensity
    return_data_map['base_intensity'] = hawkes_process.base_intensity
    return_data_map['auxiliary_variable'] = hawkes_process.auxiliary_variable
    return_data_map['kernel'] = kernel
    if kernel == 'fourier' or kernel == 'Fourier':
        return_data_map['k_omega'] = hawkes_process.k_omega
        return_data_map['y_omega'] = hawkes_process.y_omega
        return_data_map['count_of_each_slot'] = hawkes_process.count_of_each_slot
        return_data_map['count_of_each_event'] = hawkes_process.count_of_each_event

    return return_data_map


def hawkes_train():
    # the data location of local environment is different from remote server, we need select appropriate path at first
    # source data server: os.path.abspath('/mnt/datashare/diseaseprogression/reconstruct_data/mimic_3/reconstruct/')
    # source data local: os.path.abspath('..\\..\\..') + '\\reconstruct_data\\mimic_3\\reconstruct\\'
    # save data server: os.path.abspath('/mnt/datashare/diseaseprogression/reconstruct_data/mimic_3/reconstruct/')
    # save data local os.path.abspath('..\\..\\..') + '\\reconstruct_data\\mimic_3\\reconstruct\\'
    source_file_path = os.path.abspath('..\\..\\..') + '\\reconstruct_data\\mimic_3\\reconstruct\\'
    save_file_path = os.path.abspath('..\\..\\..') + '\\reconstruct_data\\mimic_3\\reconstruct\\'
    source_file_name = 'reconstructed.xml'
    name_prefix_temp = '{}_diagnosis_{}_procedure_{}_iteration_{}_slot_{}_'

    # Experiment
    iteration = 5
    for diagnosis_reserve in [3]:
        for procedure_reserve in [3]:
            train_data, test_data, name_index_map = \
                hawkes_load_data(source_file_path, source_file_name, diagnosis_reserve, procedure_reserve)

            # save corresponding name index map
            map_name = 'index_name_map_diagnosis_' + str(diagnosis_reserve) + '_procedure_' + str(
                procedure_reserve) + '.csv'
            hawkes_save_name_index_map(save_file_path, map_name, name_index_map)

            for kernel in ['exp', 'fourier']:
                if kernel == 'exp':
                    time_slot = None
                    parameter_map = hawkes_optimization(train_data, test_data, iteration, diagnosis_reserve,
                                                        procedure_reserve, kernel, time_slot)
                    name_prefix = name_prefix_temp.format(kernel, str(diagnosis_reserve), str(procedure_reserve),
                                                          str(iteration), 'none')
                    save_result(parameter_map, save_file_path, name_prefix)
                else:
                    for time_slot in [2, 3, 4]:
                        parameter_map = hawkes_optimization(train_data, test_data, iteration, diagnosis_reserve,
                                                            procedure_reserve, kernel, time_slot)
                        name_prefix = name_prefix_temp.format(kernel, str(diagnosis_reserve), str(procedure_reserve),
                                                              str(iteration), str(time_slot))
                        save_result(parameter_map, save_file_path, name_prefix)


if __name__ == '__main__':
    hawkes_train()
