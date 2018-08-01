# coding=utf-8
import csv

import numpy as np


class Intensity(object):
    def __init__(self, model_configuration):
        self.__file_encoding = model_configuration.file_encoding
        self.__event_number = model_configuration.input_x_depth

        # get_intensity
        self.__mutual_intensity_path = model_configuration.mutual_intensity_path
        self.__base_intensity_path = model_configuration.base_intensity_path
        self.__base_intensity = self.__read_base_intensity()
        self.__mutual_intensity = self.__read_mutual_intensity()

        print('initialize rnn and build mutual intensity component accomplished')

    def __read_mutual_intensity(self):
        """
        Mutual intensity is a m by m square matrix, Entry x_ij means the mutual intensity the i triggers j
        :return:
        """
        mutual_file_path = self.__mutual_intensity_path
        encoding = self.__file_encoding
        size = self.__event_number

        mutual_intensity = np.zeros([size, size])
        with open(mutual_file_path, 'r', encoding=encoding, newline="") as file:
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

    def __read_base_intensity(self):
        base_file_path = self.__base_intensity_path
        encoding = self.__file_encoding
        size = self.__event_number

        base_intensity = np.zeros([1, size])
        with open(base_file_path, 'r', encoding=encoding, newline="") as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                if len(line) != size:
                    raise ValueError('mutual intensity incompatible')
                for col_index in range(0, size):
                    base_intensity[0][col_index] = line[col_index]
                break
        return base_intensity

    @property
    def mutual_intensity(self):
        return self.__mutual_intensity

    @property
    def base_intensity(self):
        return self.__base_intensity
