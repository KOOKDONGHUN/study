import sqlite3

conn = sqlite3.connect('test.db')

cursor = conn.cursor()

sql = 'CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT, FoodName TEXT, Company TEXT, Price INTEGER)'
cursor.execute(sql)

sql = 'DELETE from supermarket' # 시작하기 전에 ? 뭐라는지 못들음 
cursor.execute(sql)

sql = 'INSERT into supermarket(Itemno , Category , FoodName , Company , Price) values(?,?,?,?,?)'
cursor.execute(sql, (1, '과일', '자몽', '마트', 1500))

sql = 'INSERT into supermarket(Itemno , Category , FoodName , Company , Price) values(?,?,?,?,?)'
cursor.execute(sql, (2, '음료수', '망고주스', '편의점', 1000))

sql = 'INSERT into supermarket(Itemno , Category , FoodName , Company , Price) values(?,?,?,?,?)'
cursor.execute(sql, (3, '고기', '소고기', '하나로마트', 15000))

sql = 'INSERT into supermarket(Itemno , Category , FoodName , Company , Price) values(?,?,?,?,?)'
cursor.execute(sql, (4, '박카스', '약', '약국', 500))

sql = 'SELECT * from supermarket'
cursor.execute(sql)

conn.commit()

rows = cursor.fetchall()

for row in rows:
    for i in range(len(row)):
        print(row[i],end=' ')
    print()

conn.close()