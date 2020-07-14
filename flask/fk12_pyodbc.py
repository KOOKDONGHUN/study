import pyodbc as pyo
# print(pyo.__version__)
server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
password = '1234'

conn = pyo.connect('DRIVER={ODBC Driver 17 for SQL Server};' + f'SERVER={server};' +
                    'PORT=1433;' +  f'DATABASE={database};' + f'UID={username};' + f'PWD={password};')

cursor = conn.cursor()

sql = 'SELECT * from iris2;'

with cursor.execute(sql):
    row = cursor.fetchone()

    while row :
        for i in range(len(row)):
            print(row[i],end=' ')
        print()
        row = cursor.fetchone()

conn.close()