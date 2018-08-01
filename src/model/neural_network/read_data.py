# coding=utf-8
import numpy as np


class LoadData(object):
    def __init__(self, batch_size, data_path, time_length):
        """
        encapsulation the load data process
        :param batch_size:
        :param data_path: 注意，要求原始数据是BTD形式，输出则是TBD形式
        :param time_length:
        """
        self.__batch_size = batch_size
        self.__data_path = data_path
        self.__time_length = time_length
        self.__max_batch_index = None
        self.__global_batch_index = 0

        self.__origin_train_x, self.__origin_train_t, self.__origin_test_x, self.__origin_test_t = self.__read_data()
        # BTD TO TBD
        self.__origin_test_x = np.transpose(self.__origin_test_x, [1, 0, 2])
        self.__origin_test_t = np.transpose(self.__origin_test_t, [1, 0, 2])

        self.__batch_train_x, self.__batch_train_t = self.__pre_process(self.__origin_train_x, self.__origin_train_t)

    def get_max_batch_index(self):
        return self.__max_batch_index

    def __read_data(self):
        """
        the data structure of origin_data
        :return:
        train_x: [train_sample_size, time_length, x_depth]
        train_t: [train_sample_size, time_length, t_depth]
        test_x: [test_sample_size, time_length, x_depth]
        test_t: [test_sample_size, time_length, t_depth]
        """
        # TODO 如何读取数据还要根据具体数据集的实际情况补充相关代码，此处仅为测试样例
        # 此处直接读取numpy二进制数组
        path = self.__data_path
        train_x = np.random.random_integers(0, 1, [self.__batch_size * 10 + 1, self.__time_length, 10])
        train_t = np.random.random_integers(0, 1, [self.__batch_size * 10 + 1, self.__time_length, 1])
        test_x = np.random.random_integers(0, 1, [self.__batch_size * 3 + 1, self.__time_length, 10])
        test_t = np.random.random_integers(0, 1, [self.__batch_size * 3 + 1, self.__time_length, 1])
        print('data path: ' + path)
        return train_x, train_t, test_x, test_t

    def get_train_next_batch(self):
        """
        return next batch.
        :return:
        """
        if self.__global_batch_index == self.__max_batch_index:
            batch_train_x, batch_train_t = self.__pre_process(self.__origin_train_x, self.__origin_train_t)
            self.__batch_train_x, self.__batch_train_t = batch_train_x, batch_train_t
            self.__global_batch_index = 0

        x, t = self.__batch_train_x[self.__global_batch_index], self.__batch_train_t[self.__global_batch_index]
        self.__global_batch_index += 1
        return x, t

    def get_test_data(self):
        return self.__origin_test_x, self.__origin_test_t

    def __pre_process(self, train_x, train_t):
        """
        shuffle and reshape the data
        :param train_x: [train_sample_size, time_length, x_depth]
        :param train_t: [train_sample_size, time_length, t_depth]
        :return:
        shuffled, and convert data from batch_major to time_major
        train_x: [batch_count, batch_size, time_length, x_depth]
        train_t: [batch_count, batch_size, time_length, t_depth]
        """
        batch_size = self.__batch_size

        # 数据随机化
        train_data = np.concatenate((train_x, train_t), axis=2)
        np.random.shuffle(train_data)

        train_x = train_data[:, :, 0:len(train_data[0][0] - 1)]
        train_t = train_data[:, :, -1]

        data_length = len(train_x)
        self.__max_batch_index = data_length // batch_size
        discard = data_length % batch_size
        train_x = train_x[0:data_length - discard, :, :]
        train_t = train_t[0:data_length - discard, :, :]

        # BTD TO TBD
        train_x = np.transpose(train_x, [0, 2, 1, 3])
        train_t = np.transpose(train_t, [0, 2, 1, 3])

        return train_x, train_t


def unit_test():
    data_path = ""
    batch_size = 6
    time_length = 7
    load_data = LoadData(batch_size, data_path, time_length)
    for i in range(0, 100):
        load_data.get_train_next_batch()


if __name__ == "__main__":
    unit_test()
