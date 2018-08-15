# coding=utf-8
import csv
import os

import numpy as np
import sklearn.metrics as sk_metric


def performance_measure(c_pred, r_pred, c_label, r_label, input_depth, threshold):
    # performance metrics are obtained based on A Review on Multi-Label Learning Algorithms,
    # Zhang et al, TKDE, 2014
    """
    :param c_pred: with size [time_stamp, batch_size, data_length]
    :param r_pred: with size [time_stamp, batch_size, data_length]
    :param c_label: with size [time_stamp, batch_size, data_length]
    :param r_label: with size [time_stamp, batch_size, data_length]
    :param input_depth:
    :param threshold:
    :return:
    """
    c_auxiliary_one = np.ones(c_pred.shape)
    c_auxiliary_zero = np.zeros(c_pred.shape)
    c_pred_label = np.where(c_pred > threshold, c_auxiliary_one, c_auxiliary_zero)

    acc = np.sum(np.logical_and(c_pred_label, c_label)) / np.sum(np.logical_or(c_pred_label, c_label))
    precision = np.sum(np.logical_and(c_pred_label, c_label)) / np.sum(c_pred_label)
    recall = np.sum(np.logical_and(c_pred_label, c_label)) / np.sum(c_label)
    f_1 = precision * recall / (precision + recall)

    # hamming loss
    denominator = c_label.shape[0] * c_label.shape[1] * c_label.shape[2]
    difference = np.logical_xor(c_pred_label, c_label)
    hamming_loss = np.sum(difference) / denominator

    c_label = np.reshape(c_label, [-1, input_depth])
    c_pred_label = np.reshape(c_pred_label, [-1, input_depth])
    coverage = sk_metric.coverage_error(c_label, c_pred_label)
    rank_loss = sk_metric.label_ranking_loss(c_label, c_pred_label)
    average_precision = sk_metric.average_precision_score(c_label, c_pred_label)

    macro_auc = sk_metric.roc_auc_score(c_label, c_pred_label, average='macro')
    micro_auc = sk_metric.roc_auc_score(c_label, c_pred_label, average='micro')
    time_dev = np.sum(np.abs(r_pred - r_label))

    metrics_map = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f_1, 'hamming_loss': hamming_loss,
                   'coverage': coverage, 'ranking_loss': rank_loss, 'average_precision': average_precision,
                   'absolute_time_deviation': time_dev, 'micro_auc': micro_auc, 'macro_auc': macro_auc}
    return metrics_map


def save_result(path, file_name, data):
    matrix_to_write = []
    head = ['epoch', 'batch', 'acc', 'precision', 'recall', 'f1', 'hamming_loss', 'coverage', 'ranking_loss',
            'average_precision', 'macro_auc', 'micro_auc', 'absolute_time_deviation']
    matrix_to_write.append(head)

    for item in data:
        epoch = item[0]
        batch = item[1]
        acc = item[2]['acc']
        precision = item[2]['precision']
        recall = item[2]['recall']
        f1 = item[2]['f1']
        hamming_loss = item[2]['hamming_loss']
        coverage = item[2]['coverage']
        ranking_loss = item[2]['ranking_loss']
        average_precision = item[2]['average_precision']
        macro_auc = item[2]['macro_auc']
        micro_auc = item[2]['micro_auc']
        absolute_time_deviation = item[2]['absolute_time_deviation']
        single_result = [epoch, batch, acc, precision, recall, f1, hamming_loss, coverage, ranking_loss,
                         average_precision, macro_auc, micro_auc, absolute_time_deviation]
        matrix_to_write.append(single_result)

    with open(os.path.join(path, file_name), 'w', encoding='utf-8-sig', newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(matrix_to_write)
