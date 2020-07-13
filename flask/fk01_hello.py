from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello33():
    return "<h1> hello donghoon world</h1>"

@app.route('/bit')
def hello3():
    return "<h2> hello fucking flask</h2>"

@app.route('/bit/hello')
def hello2():
    return "<h2> hello fucking bit flask</h2>"

@app.route('/gemma')
def hellogemma():
    return "<h2> hello gemma</h2>"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

