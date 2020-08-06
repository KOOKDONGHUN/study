import urllib.parse

# encode = '%2BiGFfCq9V7ttCQMaerGI3Uki53svVwaY1w%2BB6%2BDsu6A%3D'
        # +iGFfCq9V7ttCQMaerGI3Uki53svVwaY1w+B6+Dsu6A=

encode = 'https://kin.naver.com/userinfo/answerList.nhn?u=%2BiGFfCq9V7ttCQMaerGI3Uki53svVwaY1w%2BB6%2BDsu6A%3D&isSearch=true&query=%EC%9E%90%EA%B2%A9%EC%A6%9D&sd=answer&y=0&section=qna&isWorry=false&x=0&page=1'
decode = urllib.parse.unquote(encode)
print(decode)
