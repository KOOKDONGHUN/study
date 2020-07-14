import numpy as np
import pymssql as ms

server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
pwd = '1234'

conn = ms.connect(server=server, user=username, password=pwd, database=database)

cursor = conn.cursor()

cursor.execute('SELECT * from iris2;')

row = cursor.fetchall()
print(row)
conn.close()

print('-'*44)
aaa = np.array(row)
print(aaa)
print(type(aaa))
print(aaa.shape)

np.save('./Data/test_flask_iris2.npy', arr=aaa)