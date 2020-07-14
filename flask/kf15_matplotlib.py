from flask import Flask, render_template, send_file, make_response

from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
password = '1234'

app = Flask(__name__)

@app.route('/mypic')
def mypic():
    return render_template('mypic.html')

@app.route('/plot')
def plot():

    fig, axis = plt.subplots(1)

    # data
    x = [1, 2, 3, 4, 5]
    y = [0, 2, 1, 3, 4]

    # draw on canvas bt data
    axis.plot(x,y)
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(host=server, port=8080, debug=False)

