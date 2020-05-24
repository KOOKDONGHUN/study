from matplotlib import pyplot as plt

movies = ["Annie Hall", "Ben-Hur", " Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# 막대의 x 좌표는 [0, 1, 2, 3, 4], y 좌표는 [num_oscars]
plt.bar(range(len(movies)),num_oscars)

plt.title("My Favorite Movies")
plt.ylabel("# of Academy Awards")

# x축 각 막대의 중앙에 영화 제목을 레이블로 추가하자.
plt.xticks(range(len(movies)),movies)

plt.show()