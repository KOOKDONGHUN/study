import pymssql as ms
print(ms.__version__)

conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')

cursor = conn.cursor()

cursor.execute('SELECT * from credit_card_data;')

row = cursor.fetchone()

while row :
    print(f'첫 컬럼 : {row[0]}\t 둘컬럼 : {row[1]}')
    print(len(row))
    row = cursor.fetchone()
print(len(row))
conn.close()
