from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello1():
    return "<h1>hello donghoon</h1>"

@app.route('/ping', methods=['GET'])
def hello2():
    return "<h1>pong</h1>"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)