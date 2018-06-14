import pyodbc

# 从数据库中把所有病案首页记录，出院记录和入院记录下载到本地


def main():
    # connect database
    server = '172.16.99.248'
    database = 'clever-cdr-nxzyy'
    username = 'ZJU_SZJ'
    password = 'nxzyy@szj0808'
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=' +
                          server + ';PORT=1443;DATABASE=' + database +
                          ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    print('database connected')

    # fetch data
    sql_emr = "select patientIdentifier_id, encounterIdentifier,createDate, " \
              "emrTypeName, contentText from Obsr_EmrDocument where  " \
              "emrTypeName = '出院记录' or emrTypeName =  '入院记录' or  " \
              "emrTypeName ='住院病案首页基本数据'"

    first_page_path = 'D:\\Ningxia\\first_page_record_xml\\'
    admission_path = 'D:\\Ningxia\\admission_record_xml\\'
    discharge_path = 'D:\\Ningxia\\discharge_record_xml\\'
    folder_index_list = ['01\\', '02\\', '03\\', '04\\', '05\\', '06\\',
                         '07\\', '08\\', '09\\', '10\\', '11\\', '12\\', '13\\',
                         '14\\']
    fetch_unit = 50000

    with cursor.execute(sql_emr):
        download_record = 0
        first_page_count = 0
        admission_count = 0
        discharge_count = 0
        while True:
            data_rows = cursor.fetchmany(fetch_unit)
            current_row = 0
            for line in data_rows:
                patient_id, encounter_id, create_date, emr_type, emr_content \
                    = line

                if create_date is None:
                    create_date = 'Null'
                else:
                    create_date = create_date.split(" ")[0]

                if emr_type == '住院病案首页基本数据':
                    create_xml(patient_id+'_'+encounter_id+"_"+create_date +
                               ".xml", first_page_path, folder_index_list[first_page_count//fetch_unit],
                               emr_content)
                    first_page_count += 1
                elif emr_type == '入院记录':
                    create_xml(patient_id+'_'+encounter_id+"_"+create_date +
                               ".xml", admission_path, folder_index_list[admission_count//fetch_unit],
                               emr_content)
                    admission_count += 1
                elif emr_type == '出院记录':
                    create_xml(patient_id+'_'+encounter_id+"_"+create_date +
                               ".xml", discharge_path, folder_index_list[discharge_count//fetch_unit],
                               emr_content)
                    discharge_count += 1
                else:
                    print('error')

                download_record += 1
                current_row += 1
                if download_record % fetch_unit == 0:
                    print('download record ' + str(download_record))
            if current_row < 10:
                break


def create_xml(file_name, file_path, folder_index, file_content):
    with open(file_path + folder_index + file_name, 'w', encoding='utf-8-sig',
              newline="") as w:
        w.write(file_content)


if __name__ == "__main__":
    main()
