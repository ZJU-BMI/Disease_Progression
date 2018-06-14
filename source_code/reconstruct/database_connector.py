import pyodbc


def get_cursor():
    server = '172.16.99.248'
    database = 'clever-cdr-nxzyy'
    username = 'ZJU_SZJ'
    password = 'nxzyy@szj0808'
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=' +
                          server + ';PORT=1443;DATABASE=' + database +
                          ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    print('database connected')
    return cursor
