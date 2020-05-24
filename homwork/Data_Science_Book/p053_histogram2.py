from matplotlib import pyplot as plt
from collections import Counter

''' 히스토그램이란 정해진 구간에 대한 항목의 개수를 보여줌으로써 값의 분포를 관찰 할 수 있는 그래프 '''

mentions = [500, 505]
years = [2017,2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")

# # 이렇게 하지 않으면 matplotlib이 x축에 0, 1 레이블을 달고
# #  주변부 어딘가에 +2.013... 이라고 표가해 둘 것이다.
# plt.ticklabel_format(useOffset=False)
''' ?? 차이가 없는데 뭔 말이야... '''

# # 오해를 불러일으킬 수 있는 y축의 tick이 500이상의 부분만 보여줄 것이다.
# plt.axis([2016.5, 2018.5, 499, 506])
# plt.title("Look at the 'Huge' Increase!")
# plt.show()

''' 실제 값의 차이는 6인데 위에 처럼 축을 지정하면 엄청 차이가 많아 보인다. '''
plt.axis([2016.5, 2018.5, 0, 550])
plt.title("Not So Huge Anymore")
plt.show()