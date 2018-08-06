# coding=utf-8
import csv

import numpy as np
import tensorflow as tf


class Intensity(object):
    def __init__(self, model_config):
        self.__event_number = model_config.input_x_depth
        self.__mi_holder = tf.placeholder('float64',
                                          [self.__event_number, self.__event_number], name='mutual_intensity')
        self.__bi_holder = tf.placeholder('float64', [1, self.__event_number], name='base_intensity')
        print('initialize rnn and build mutual intensity component accomplished')

    @property
    def mutual_intensity_placeholder(self):
        return self.__mi_holder

    @property
    def base_intensity_placeholder(self):
        return self.__bi_holder

    @staticmethod
    def read_mutual_intensity(mutual_intensity_path, size, encoding):
        """
        Mutual intensity is a m by m square matrix, Entry x_ij means the mutual intensity the i triggers j
        :return:
        """

        mutual_intensity = np.zeros([size, size])
        with open(mutual_intensity_path, 'r', encoding=encoding, newline="") as file:
            csv_reader = csv.reader(file)
            row_index = 0
            for line in csv_reader:
                if len(line) != size:
                    raise ValueError('mutual intensity incompatible')
                for col_index in range(0, size):
                    mutual_intensity[row_index][col_index] = line[col_index]
                row_index += 1
            if row_index != size:
                raise ValueError('mutual intensity incompatible')
        return mutual_intensity

    @staticmethod
    def read_base_intensity(base_intensity_path, size, encoding):
        base_intensity = np.zeros([1, size])
        with open(base_intensity_path, 'r', encoding=encoding, newline="") as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                if len(line) != size:
                    raise ValueError('mutual intensity incompatible')
                for col_index in range(0, size):
                    base_intensity[0][col_index] = line[col_index]
                break
        return base_intensity
