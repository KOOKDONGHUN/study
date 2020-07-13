from flask import Flask, Response, make_response

app = Flask(__name__)

@app.route('/')
def response():
    custom_response = Response("[★] Custom Response", 300, {'Program' : 'Flask web Fucking App kkk'})
    print('0. ')
    return make_response(custom_response)

@app.before_first_request
def before(): # route 실행되기 이전에 처음 한번만 실행
    print("1. 앱이 가동되고 나서 첫번쨰 HTTP 요청에만 응답합니다")

@app.before_request
def before_request(): # route 이전에 실행 ??
    print("2. 매 HTTP 요청 이전에 응답합니다")

@app.after_request
def after_request(response): # route 다음에 실행
    print("3. 매 HTTP 요청 이후에 응답합니다")
    return response

@app.teardown_request
def teardown_request(exception): # route 다음에 실행
    print("4. 매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다.") 
    # return exception

@app.teardown_appcontext
def teardown_appcontext(exception): # route 다음에 실행
    print("5. 매 HTTP 요청의 애플리게이션 컨텍스트가 종료될 때 실행된다.") 
    # return exception

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)