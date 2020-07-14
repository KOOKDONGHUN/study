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

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/sqltable')
def showsql():
    cursor.execute(sql)
    return render_template('myweb.html', rows=cursor.fetchall())

if __name__ == '__main__':
    app.run(host=server, port=8080, debug=False)
conn.close()