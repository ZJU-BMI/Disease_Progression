import csv
from itertools import islice

operation_map = {}
token_index = 1
with open("..\\reconstruct_data\\reconstruct_data_structured_diagnosis.csv",
          "r", encoding="utf-8-sig", newline="") as file:
    csv_reader = csv.reader(file)
    for line in islice(csv_reader, 1, None):
        pat_id, visit_id, encounter_id, admission_time, diagnosis, dept, \
         operation, death_record, code_str = line
        if not operation_map.__contains__(operation):
            operation_map[operation] = token_index
            token_index += 1

write_content = []
with open("..\\reconstruct_data\\reconstruct_data_structured_diagnosis.csv",
          "r", encoding="utf-8-sig", newline="") as file:
    csv_reader = csv.reader(file)
    for line in islice(csv_reader, 1, None):
        pat_id, visit_id, encounter_id, admission_time, diagnosis, dept, \
         operation, death_record, code_str = line
        write_content.append([pat_id, visit_id, encounter_id, admission_time,
                              diagnosis, dept, operation, death_record,
                              code_str, operation_map[operation]])
with open("..\\reconstruct_data\\reconstruct_data_structured.csv",
          "w", encoding="utf-8-sig", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows([["patient_id", "visit_id", "encounter_id",
                           "admission_time", "diagnosis", "department",
                           "operation", "discharge_type", "diagnosis_code_set",
                           "operation_code"]])
    csv_writer.writerows(write_content)

with open("..\\reconstruct_data\\operation_dict.csv",
          "w", encoding="utf-8-sig", newline="") as file:
    csv_writer = csv.writer(file)
    for item in operation_map:
        csv_writer.writerows([[item, operation_map[item]]])
