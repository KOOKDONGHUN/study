import pymssql as ms
print(ms.__version__)

conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')

cursor = conn.cursor()

cursor.execute('SELECT * from iris2;')

row = cursor.fetchone()

while row :
    print(f'첫 컬럼 : {row[0]}\t 둘컬럼 : {row[1]}')
    row = cursor.fetchone()

conn.close()