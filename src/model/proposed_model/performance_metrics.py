# coding=utf-8
import csv
import os

import numpy as np
import sklearn.metrics as sk_metric


def performance_measure(c_pred, c_label, input_depth, threshold):
    # performance metrics are obtained based on A Review on Multi-Label Learning Algorithms,
    # Zhang et al, TKDE, 2014
    """
    :param c_pred: with size [time_stamp, batch_size, data_length]
    :param c_label: with size [time_stamp, batch_size, data_length]
    :param input_depth:
    :param threshold:
    :return:
    """
    c_auxiliary_one = np.ones(c_pred.shape)
    c_auxiliary_zero = np.zeros(c_pred.shape)
    c_pred_label = np.where(c_pred > threshold, c_auxiliary_one, c_auxiliary_zero)

    _, coverage_ = coverage(c_pred, c_label)
    _, one_error = top_k_coverage(c_pred, c_label, 1)
    _, five_coverage = top_k_coverage(c_pred, c_label, 5)
    _, ten_coverage = top_k_coverage(c_pred, c_label, 10)
    _, fifteen_coverage_rate = top_k_coverage(c_pred, c_label, 15)

    tp = np.sum(np.logical_and(c_pred_label, c_label))
    tn = np.sum(np.logical_not(np.logical_or(c_pred_label, c_label)))
    fp = np.sum(np.logical_xor(np.logical_or(c_pred_label, c_label), c_label))
    fn = np.sum(np.logical_not(c_pred_label)) - tn

    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_1 = 2 * precision * recall / (precision + recall)

    # hamming loss
    denominator = c_label.shape[0] * c_label.shape[1] * c_label.shape[2]
    difference = np.logical_xor(c_pred_label, c_label)
    hamming_loss = np.sum(difference) / denominator

    c_label = np.reshape(c_label, [-1, input_depth])
    c_pred_label = np.reshape(c_pred_label, [-1, input_depth])
    coverage_error = sk_metric.coverage_error(c_label, c_pred_label)
    rank_loss = sk_metric.label_ranking_loss(c_label, c_pred_label)
    average_precision = sk_metric.average_precision_score(c_label, c_pred_label)

    macro_auc = sk_metric.roc_auc_score(c_label, c_pred_label, average='macro')
    micro_auc = sk_metric.roc_auc_score(c_label, c_pred_label, average='micro')

    metrics_map = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f_1, 'hamming_loss': hamming_loss,
                   'coverage_error': coverage_error, 'ranking_loss': rank_loss, 'average_precision': average_precision,
                   'micro_auc': micro_auc, 'macro_auc': macro_auc,
                   'coverage': coverage_, '5_coverage': five_coverage, '10_coverage': ten_coverage,
                   'one_error': one_error, '15_coverage': fifteen_coverage_rate}
    return metrics_map


def coverage(c_pred, c_label):
    coverage_list = []
    coverage_sum = 0
    for i in range(len(c_label)):
        single_coverage = coverage_day(c_pred, c_label, i)
        coverage_list.append(single_coverage)
        coverage_sum += single_coverage
    return coverage_list, coverage_sum / len(c_label)


def coverage_day(c_pred, c_label, time_stamp):
    pred = c_pred[time_stamp]
    label = c_label[time_stamp]

    coverage_sum = 0
    effective_data = 0
    for i in range(len(pred)):
        # i 0-batch_size:
        single_pred = pred[i]
        single_label = label[i]
        pair_list = []
        for j in range(len(single_pred)):
            pair_list.append([single_pred[j], single_label[j]])
        pair_list = sorted(pair_list, key=lambda x: x[0])
        last_index = 0
        for j in range(len(pair_list)):
            if pair_list[j][1] == 1:
                last_index = j
        if last_index > 0:
            effective_data += 1
        coverage_sum += last_index
    return coverage_sum / effective_data


def top_k_coverage(c_pred, c_label, k):
    coverage_list = []
    coverage_sum = 0
    for i in range(len(c_label)):
        single_coverage = top_k_coverage_day(c_pred, c_label, i, k)
        coverage_list.append(single_coverage)
        coverage_sum += single_coverage
    return coverage_list, coverage_sum / len(c_label)


def top_k_coverage_day(c_pred, c_label, time_stamp, k):
    pred = c_pred[time_stamp]
    label = c_label[time_stamp]
    coverage_sum = 0
    effective_data = 0
    for i in range(len(pred)):
        # i 0-batch_size:
        single_pred = pred[i]
        single_label = label[i]
        pair_list = []
        for j in range(len(single_pred)):
            pair_list.append([single_pred[j], single_label[j]])
        pair_list = sorted(pair_list, key=lambda x: x[0])
        for j in range(k):
            if pair_list[j][1] == 1:
                coverage_sum += 1

        effective_flag = False
        for j in range(len(pair_list)):
            if pair_list[j][1] == 1:
                effective_flag = True
        if effective_flag:
            effective_data += 1
    return coverage_sum / effective_data


def save_result(path, file_name, data):
    matrix_to_write = []
    head = ['epoch', 'batch', 'acc', 'precision', 'recall', 'f1', 'hamming_loss', 'coverage', 'ranking_loss',
            'average_precision', 'macro_auc', 'micro_auc', 'coverage', 'one_error'
                                                                       '5_coverage', '10_coverage', '15_coverage']
    matrix_to_write.append(head)

    for item in data:
        epoch = item[0]
        batch = item[1]
        acc = item[2]['acc']
        precision = item[2]['precision']
        recall = item[2]['recall']
        f1 = item[2]['f1']
        hamming_loss = item[2]['hamming_loss']
        coverage_error = item[2]['coverage_error']
        ranking_loss = item[2]['ranking_loss']
        average_precision = item[2]['average_precision']
        macro_auc = item[2]['macro_auc']
        micro_auc = item[2]['micro_auc']
        coverage_ = item[2]['coverage']
        one_error = item[2]['one_error']
        five_coverage = item[2]['5_coverage']
        ten_coverage = item[2]['10_coverage']
        fifteen_coverage = item[2]['15_coverage']
        single_result = [epoch, batch, acc, precision, recall, f1, hamming_loss, coverage_error, ranking_loss,
                         average_precision, macro_auc, micro_auc, coverage_, one_error,
                         five_coverage, ten_coverage, fifteen_coverage]
        matrix_to_write.append(single_result)

    with open(os.path.join(path, file_name), 'w', encoding='utf-8-sig', newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(matrix_to_write)


def save_roc(path, file_name, data):
    fpr, tpr, threshold = data
    matrix_to_write = list()
    matrix_to_write.append(['fpr', 'tpr', 'threshold'])
    for i, _ in enumerate(fpr):
        matrix_to_write.append([fpr, tpr, threshold])
    with open(os.path.join(path, file_name), 'w', encoding='utf-8-sig', newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(matrix_to_write)
