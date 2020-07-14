# 20-07-14_35
# img 그래프 웹으로 표현하기

from flask import Flask, render_template, send_file, make_response

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

@app.route('/mypic')
def mypic():
    return render_template('mypic.html')

@app.route('/plot')
def plot():

    fig, axis = plt.subplots(1)

    # 데이터를 준비
    x = [1, 2, 3, 4, 5]
    y = [0, 2, 1, 3, 4]

    # 데이터를 캔버스에 그린다
    axis.plot(x, y)
    canvas = FigureCanvas(fig)

    from io import BytesIO
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    port = 5050
    app.debug = False
    app.run(port = port)

# web 상으로 plot을 구현 가능하다
# 어쨌든 데이터베이스에 파일들을 관리하고 큰 파일의 경우에는 더 빠르게 작동할 것이다.