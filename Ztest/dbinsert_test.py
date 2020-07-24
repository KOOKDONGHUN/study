import pymssql as ms
import collections

# try:
#     collectionsAbc = collections.abc
# except AttributeError:
#     collectionsAbc = collections

# insert_data = ['a', 'aasss', 've', 'aaaa']
# tablename = 'test'

# conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
# cursor = conn.cursor()
# sql = f'INSERT INTO {tablename} VALUES (%s, %s, %s, %s)'
# # sql = 'SELECT * FROM iris2;'
# cursor.execute(sql, (insert_data[0], insert_data[1], insert_data[2], insert_data[3]))
# conn.commit()
# conn.close()
def dbconn(tablename, insert_data):
    conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'INSERT INTO {tablename} values(%s ,%s ,%s ,%s)'
    cursor.execute(sql, (insert_data[0], insert_data[1], insert_data[2], insert_data[3]))
    conn.commit()
    conn.close()
ls = ['', '', '', '']

dbconn('iris2', ls)