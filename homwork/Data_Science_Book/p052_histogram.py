from matplotlib import pyplot as plt
from collections import Counter
''' 히스토그램이란 정해진 구간에 대한 항목의 개수를 보여줌으로써 값의 분포를 관찰 할 수 있는 그래프 '''
grades = [83, 95, 91, 97, 70, 0, 85, 83, 100, 67, 73, 77, 0]

# 점수는 10점 단위로 그룹화한다. 100점은 90점대로 속한다.
histogram = Counter(min(grade // 10*10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()], # 각 막대를 오른쪽으로 5만큼 옮기고 
        histogram.values(),                # 각 막대의 높이를 정해주고
        10,                                # 너비는 10으로 한다.
        edgecolor = (0, 0, 0))             # 각 막대의 테두리는 검정색으로 설정한다.

plt.axis([-5, 105, 0, 5])

plt.xticks([ 10*i for i in range(11)]) # x 축의 레이블은 0, 10, ... , 100

plt.title("Distribution of Exam 1 Grades")

plt.xlabel("Decile")
plt.ylabel("# of Students")

plt.show()