# coding=utf-8
import os

import numpy as np

save_path = os.path.join(os.path.abspath('..\\'), 'model_evaluate\\validationTest')
train_input_x = np.random.randint(0, 1, [235, 5, 20])
train_input_t = np.random.randint(0, 1, [235, 5, 1])
test_input_x = np.random.randint(0, 1, [87, 5, 20])
test_input_t = np.random.randint(0, 1, [87, 5, 1])
np.save(os.path.join(save_path, 'train_input_x.npy'), train_input_x)
np.save(os.path.join(save_path, 'train_input_t.npy'), train_input_t)
np.save(os.path.join(save_path, 'test_input_x.npy'), test_input_x)
np.save(os.path.join(save_path, 'test_input_t.npy'), test_input_t)
