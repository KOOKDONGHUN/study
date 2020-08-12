import numpy as np

np.random.seed(0)

# n은 배치 크기이며, D_in은 입력의 차원이다.
# H는 은닉층의 차원이며, D_out은 출력 차원이다. 여기서 말하는 은닉층의 차원은 무엇?
N, D_in, H, D_out = 64, 1000, 100, 10

# 무작위의 입력과 출력 데이터를 생성한다. 
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 무작위로 가중치를 초기화 한다.
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-06
for time_step in range(500):
# for time_step in range(5):

    # 순전파 단계 : 예측값 y를 계산한다.???
    h = x.dot(w1) # x와 w1의 행렬곱 
    # print(f'h : {h}')

    h_relu = np.maximum(h, 0) # h에 0이하인 값도 있네 그래서 최소가 0이 되는데 그게 relu취해주는거랑 같네 ...
    # print(f'h_relu : {h_relu}')

    y_pred = h_relu.dot(w2) # 활성화 함수 relu를 거친 이전의 h에 다시한번 w2를 곱한다?? 두번째 레이어 이기 때문인가 

    # loss를 계산하고 출력합니다.
    loss = np.square(y_pred - y).sum() # 
    print(time_step, loss)

    # 손실에 따른 w1, w2의 변화도를 계산하고 역전파합니다.
    grad_y_pred = 2.0 * (y_pred - y) # 2.0을 곱하는 이유??
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0 # 음수에 0 relu 적용
    grad_w1 = x.T.dot(grad_h)

    # 가중치를 갱신합니다.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2