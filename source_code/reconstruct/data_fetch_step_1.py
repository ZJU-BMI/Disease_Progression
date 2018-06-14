# coding=utf-8

# 重新整理病案首页记录，抽取其中的入院，出院，诊断，手术，死亡信息，整理成CSV

import csv
from xml.dom.minidom import parseString
from reconstruct import database_connector

depart_interest = {}

cursor = database_connector.get_cursor()
sql = "select contentText, EncounterIdentifier, PatientIdentifier_Id " \
      "from Obsr_EmrDocument where emrTypeName = '住院病案首页基本数据'"

encounter_time_map = {}
index_count = 0
fetch_unit = 10000
download_record = 0
with cursor.execute(sql):
    while True:
        data_rows = cursor.fetchmany(fetch_unit)
        current_row = 0
        for line in data_rows:
            # 读取行数计数
            current_row += 1
            download_record += 1
            if download_record % fetch_unit == 0:
                print('download record ' + str(download_record))

            # 读取目标数据
            context, encounter_id, patient_id = line
            with parseString(context) as dom_tree:
                root = dom_tree.documentElement
                item_set = root.getElementsByTagName('DItem')
                if len(item_set) > 0:
                    encounter_info = dict(patient_id=patient_id,
                                          cate='inHospital')
                    for item in item_set:
                        if item.hasAttribute('ItemID'):
                            item_id = item.getAttribute("ItemID")
                            if item_id == 'HDSD00.11.085':
                                admission_date = item.getAttribute("ItemValue")
                                encounter_info['admission'] = admission_date
                            if item_id == 'HDSD00.11.019':
                                discharge_date = item.getAttribute('ItemValue')
                                encounter_info['discharge'] = discharge_date
                            if item_id == 'HDSD00.11.024':
                                main_diagnosis = item.getAttribute('ItemValue')
                                if main_diagnosis is not None and \
                                        main_diagnosis != "":
                                    # 防止逗号分隔符出错
                                    main_diagnosis = main_diagnosis.replace(
                                        ',', ';')
                                    main_diagnosis = main_diagnosis.replace(
                                        '，', ';')
                                    encounter_info[
                                        'main_diagnosis'] = main_diagnosis
                            if item_id == 'HDSD00.11.021':
                                other_diagnosis = item.getAttribute('ItemValue')
                                # 防止逗号分隔符出错
                                other_diagnosis = other_diagnosis.replace(
                                    ',', ';')
                                other_diagnosis = other_diagnosis.replace(
                                    '，', ';')
                                encounter_info[
                                    'other_diagnosis'] = other_diagnosis
                            if item_id == 'HDSD00.11.018':
                                department = item.getAttribute('ItemValue')
                                encounter_info['department'] = department
                            if item_id == 'HDSD00.11.090':
                                operation = item.getAttribute("ItemValue")
                                encounter_info['operation'] = operation
                            if item_id == 'HDSD00.11.603':
                                discharge_way = item.getAttribute("ItemValue")
                                encounter_info['discharge_way'] = discharge_way

                    # 数据必须满足拥有入院，出院时间和主诊断，如果缺少直接丢弃
                    flag = encounter_info.__contains__('admission') and \
                        encounter_info.__contains__('discharge') and \
                        encounter_info.__contains__('main_diagnosis')
                    if flag:
                        encounter_time_map[encounter_id] = encounter_info
                    else:
                        print('xml size error')

        if current_row < 10:
            break

writer = csv.writer(open('..\\resource\\patient_inhospital_record.csv', 'w',
                         encoding='utf-8-sig', newline=""))
head = ["patient_id", "encounter_id", "cate", "admission_date",
        "discharge_time", "main_diagnosis", "other_diagnosis", "department",
        "operation", "discharge_way"]
writer.writerow(head)
for item in encounter_time_map:
    row = [encounter_time_map[item]['patient_id'],
           item,
           encounter_time_map[item]['cate'],
           encounter_time_map[item]['admission'],
           encounter_time_map[item]['discharge'],
           encounter_time_map[item]['main_diagnosis']]
    if encounter_time_map[item].__contains__('other_diagnosis'):
        row.append(encounter_time_map[item]['other_diagnosis'])
    else:
        row.append('No_Other_diagnosis')
    if encounter_time_map[item].__contains__('department'):
        row.append(encounter_time_map[item]['department'])
    else:
        row.append('No_Depart_Record')
    if encounter_time_map[item].__contains__('operation'):
        row.append(encounter_time_map[item]['operation'])
    else:
        row.append('no_operation_record')
    if encounter_time_map[item].__contains__('discharge_way'):
        row.append(encounter_time_map[item]['discharge_way'])
    else:
        row.append('no_discharge_record')
    writer.writerow(row)
