import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({"col" : [1, 1, 2, 3, 4, 4, 6, 6],
                        "col2" : ['a', 'b', 'b', 'b', 'c', 'c', 'b', 'b']})

print(f"dupli_data : \n{dupli_data}")

# 중복된 행을 True로 표시한다.
print(f"dupli_data : \n{dupli_data.duplicated()}")

# 중복된 행을 제거해준다. 모든 칼럼이 동일한 중복을 말함 칼럼당 중복이 아니라
print(f"dupli_data : \n{dupli_data.drop_duplicates()}")

