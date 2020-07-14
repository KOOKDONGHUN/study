conn = ms.connect(server='127.0.0.1:49683', user='bit2', password='3411', database='bitdb')

cursor = conn.cursor()

cursor.execute('SELECT * FROM iris2;')

# 150행 중 1줄을 가져온다
row = cursor.fetchone()
# row = cursor.fetchchone()
# row = cursor.fetchchone()

while row :
    print('첫 컬럼 : %s, 둘 컬럼 : %s'%(row[0], row[1]))
    row = cursor.fetchone()

conn.close()    # connect 했으니 close