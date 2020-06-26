# x, y, w, h = map(int, input().split())

# min = 1000

# for i in range(1,w):
#     for j in range(1,h):
#         res = ((x-i)**2 + (y-j)**2)**0.5
#         if res < min :
#             min = res
# print(min)

x,y,w,h=map(int,input().split())

print(min(x,y,(w-x),(h-y)))