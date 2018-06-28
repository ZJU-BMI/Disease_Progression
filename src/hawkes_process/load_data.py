import numpy as np
import csv
from itertools import islice


def load_data(file_path):
    data_map = {}
    with open(file_path, 'r', encoding='utf-8-sig', newline="") as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            if not data_map.__contains__(patient_id):
                data_map[patient_id] = {}
            visit_id = line[1]
            data_tuple = np.array(line[2:])
            data_map[patient_id][visit_id] = data_tuple
    return data_map


def unit_test():
    data_file_path = "..\\..\\resource\\reconstruct_data\\toy_data__level.csv"
    data_map = load_data(data_file_path)
    print(data_map)


if __name__ == "__main__":
    unit_test()
