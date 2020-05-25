from matplotlib import pyplot as plt
'''  '''
test_1_grades = [ 99, 90, 85, 97, 80 ]
test_2_grades = [ 100, 85, 60, 90, 70 ]

plt.scatter(test_1_grades, test_2_grades)

plt.title("Axes Aren`t Comparable")
plt.xlabel("test_1_grades")
plt.ylabel("test_2_grades")
plt.show()