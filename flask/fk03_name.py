from flask import Flask

app = Flask(__name__)

@app.route('/<name>')
def user(name):
    return '<h1>hello %s !!!</h1>' %name

@app.route('/user/<name>')
def user2(name):
    return f'<h1>hello user {name} !!!</h1>'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)