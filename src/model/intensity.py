# coding=utf-8
import csv

import numpy as np


class Intensity(object):
    def __init__(self, event_number, mutual_intensity_path, base_intensity_path, file_encoding):
        self.__file_encoding = file_encoding
        self.__event_number = event_number

        # get_intensity
        self.__mutual_intensity_path = mutual_intensity_path
        self.__base_intensity_path = base_intensity_path
        self.__base_intensity = self.__read_base_intensity()
        self.__mutual_intensity = self.__read_mutual_intensity()

        print('initialize rnn and build mutual intensity component accomplished')

    def __read_mutual_intensity(self):
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
