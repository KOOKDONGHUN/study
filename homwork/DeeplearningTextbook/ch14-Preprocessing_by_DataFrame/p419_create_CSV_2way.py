import csv

with open("./Data/csv/csv0.csv", "w") as csvfile:
    writer = csv.writer(csvfile, lineterminator='\n')

    writer.writerow(["city", 'year', 'season'])
    writer.writerow(["Nagano", '1998', 'winter'])
    writer.writerow(["Sydney", '2000', 'summer'])
    writer.writerow(["Salt Lake City", '2000', 'winter'])
    writer.writerow(["Athens", '2004', 'summer'])
    writer.writerow(["Torino", '2006', 'winter'])
    writer.writerow(["Beijing", '2008', 'summer'])
    writer.writerow(["Vencouver", '2010', 'winter'])
    writer.writerow(["London", '2012', 'summer'])
    writer.writerow(["Sochi", '2014', 'winter'])
    writer.writerow(["Rio de Janeiro", '2016', 'summer'])


import pandas as pd

data = {'city' : ["Nagano", "Sydney", "Salt Lake City", "Athens", "Torino", "Beijing", "Vencouver", "London", "Sochi", "Rio de Janeiro"],
        'year' : ['1998', '2000', '2000', '2004', '2006', '2008', '2010', '2012', '2014', '2016'],
        'season' : ['winter', 'summer', 'winter', 'summer', 'winter', 'summer', 'winter', 'summer', 'winter', 'summer']}

df = pd.DataFrame(data)
df.to_csv('./Data/csv/csv1.csv',index=None)