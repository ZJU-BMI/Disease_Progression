import csv

# 从数据库的结构化数据中抽取诊断信息（包括住院与门诊），形成CSV

# connect database
cursor = database_connector.get_cursor()
sql = "SELECT s.PatientIdentifier_Id, s.EncounterIdentifier, DateOfDiagnosis, " \
      "s.Description, DiagnosisIdentifier_Id, ResultOfTreatment," \
      "AssignedPatientLocation_LocationDescription_Code," \
      "AssignedPatientLocation_LocationDescription_Name from " \
      "eval_problemDiagnosis s,Admin_PatientAdmission t where " \
      "s.PatientIdentifier_Id = t.PatientIdentifier_Id and " \
      "s.EncounterIdentifier = t.EncounterIdentifier_Id " \
      "and s.Description is not null "

encounter_diagnosis_map = {}
fetch_unit = 100000
download_record = 0
with cursor.execute(sql):
    while True:
        data_rows = cursor.fetchmany(fetch_unit)
        current_row = 0
        for line in data_rows:
            current_row += 1
            patient_id, encounter_id, date, des, diag_id, treat_result, \
                location_code, location_name = line
            # 确保时间，病情描述，病人ID和入院ID存在
            if date is None:
                continue
            if des is None:
                continue
            if patient_id is None:
                continue
            if encounter_id is None:
                continue

            if treat_result is None:
                treat_result = 'no_Result'
            if location_code is None:
                location_code = 'no_location_code'
            if location_name is None:
                location_name = 'location_name'

            # 逗号替换，防止写CSV出现问题
            des = des.replace(',', ';')
            des = des.replace('，', ';')

            # 每个Encounter都被写入一条记录中
            if encounter_diagnosis_map.__contains__(encounter_id):
                origin_description = encounter_diagnosis_map[encounter_id][3]
                new_description = origin_description+"$"+des
                encounter_diagnosis_map[encounter_id][3] = new_description
            else:
                encounter_diagnosis_map[encounter_id] = [patient_id,
                                                         encounter_id, date,
                                                         des, treat_result,
                                                         location_code,
                                                         location_name]
            download_record += 1
            if download_record % fetch_unit == 0:
                print('download record ' + str(download_record))

        if current_row < 10:
            break


writer = csv.writer(open('..\\resource\\patient_diagnosis_table_record.csv', 'w',
                         encoding='utf-8-sig', newline=""))
head = ["patient_id", "encounter_id", "create_date", "description",
        "treat_result", "location_code", "location_name"]
writer.writerow(head)
for item in encounter_diagnosis_map:
    row = [encounter_diagnosis_map[item][0],
           encounter_diagnosis_map[item][1],
           encounter_diagnosis_map[item][2],
           encounter_diagnosis_map[item][3],
           encounter_diagnosis_map[item][4],
           encounter_diagnosis_map[item][5],
           encounter_diagnosis_map[item][6]
           ]
    writer.writerow(row)

