import pymssql as ms

def create_table(tablename):
    # conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f"IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{tablename}' AND xtype='U')\
         CREATE TABLE {tablename} (id int identity, que varchar(1200) null, que_detail varchar(1200) null,\
             ans_writer char(50) null, ans_detail varchar(1200) null)"
    cursor.execute(sql)
    conn.commit()
    conn.close()

create_table('test5')