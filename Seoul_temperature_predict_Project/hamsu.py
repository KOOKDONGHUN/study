
# DataFrame의 널값 열별로 확인하기
def view_nan(data,print_view=True):
    for i in data.columns:
        if print_view :
            print(i , "\t", len(data[data[i].isnull()]))