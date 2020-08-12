import torch

class MyRelu(torch.autograd.Function):
    """ Tensor연산을 하는 순전파와 역전파 단계를 구현하자. """

    @staticmethod # 클래스와 스태틱 개념을 잘 모르겠네 ...
    def forward(ctx, input):
        """순전파 단계에서는 입력을 갖는 Tensor를 받아 출력을 갖는 Tensor를 반환합니다.
           ctx는 컨텍스트 객체로 역전파 연산을 위한 정보 저장에 사용합니다.
           ctx.save_for_backward method를 사용하여 역전파 단계에서 사용할 어떠한
           객체도 저장해 둘 수 있습니다. """

        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """역전파 단계에서는 출력에 대한 손실의 변화도를 갖는 Tensor를 받고,
           입력에 대한 손실의 변화도를 계산합니다."""
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[ input < 0 ] = 0
        return grad_input
        
dtype = torch.float
device = torch.device('cpu')

# n은 배치 크기이며, D_in은 입력의 차원이다.
# H는 은닉층의 차원이며, D_out은 출력 차원이다. 여기서 말하는 은닉층의 차원은 무엇? 내가아는 노드의 갯수를 말하는건가 ㅎㅎ
N, D_in, H, D_out = 64, 1000, 100, 10

# 무작위의 입력과 출력 데이터를 생성한다. 
x = torch.randn(N, D_in,device=device, dtype=dtype)
y = torch.randn(N, D_out,device=device, dtype=dtype)

# 무작위로 가중치를 초기화 한다.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True) # requires_grad=True 역전파시 가중치에 대한 변화도를 계산할 필요가 있음을 나타낸다.
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-06
for time_step in range(500):
# for time_step in range(5):

    # 순전파 단계 : 예측값 y를 계산한다.???
    # h = x.mm(w1) # x와 w1의 행렬곱 matmul
    # print(f'h : {h}')
    # 역전파 단계를 하드코딩하지 않아도되고 중간값들(각각의 레이어들의 가중치를 말하는듯?)에 대한 참조를 갖고 있을 필요가 없다.
    # 참조를 가질 필요없다는게 뭔말인지 모르겠다. 그냥 하드코딩 하지않아도 연결된다 그런의미 같음
    y_pred = x.mm(w1).clamp(min=0).mm(w2) # x와 w1의 행렬곱 matmul


    # h_relu = h.clamp(min=0) # h에 0이하인 값도 있네 그래서 최소가 0이 되는데 그게 relu취해주는거랑 같네 ...
    # print(f'h_relu : {h_relu}')

    # y_pred = h_relu.mm(w2) # 활성화 함수 relu를 거친 이전의 h에 다시한번 w2를 곱한다?? 두번째 레이어 이기 때문인가 

    # loss를 계산하고 출력합니다.
    loss = (y_pred - y).pow(2).sum() # 
    print(time_step, loss.item()) # item() == loss의 스칼라 값

    # 손실에 따른 w1, w2의 변화도를 계산하고 역전파합니다.
    # grad_y_pred = 2.0 * (y_pred - y) # 2.0을 곱하는 이유??
    # grad_w2 = h_relu.t().mm(grad_y_pred)
    # grad_h_relu = grad_y_pred.mm(w2.T)
    # grad_h = grad_h_relu.clone()
    # grad_h[h < 0] = 0 # 음수에 0 relu 적용
    # grad_w1 = x.t().mm(grad_h)
    loss.backward() 
    # 이게 바로 autograd인데 역전파를 손쉽게 해주는 메소드
    # requires_grad = True의 Tensor에 대해 손실 변화도(loss함수의 미분값?의 변화)를 계산한다.
    # w1.grad와 w2.grad는 w1과 w2 각각에 대한 손실의 변화도를 갖는 Tensor가 됩니다.

    # 경사 하강법 (SGD)를 사용하여 가중치를 수동으로 갱신한다.
    # torch.no_grad()로 감싸는 이유는 가중치들이 requires_grad=True이지만
    # autograd에서는 이를 추적할 필요가 없기 때문이다...... 왜??
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 가중치 갱신 후에는 수동으로 변화도를 0으로 만든다.
        w1.grad.zero_() # 왜 하는지 정확한 이유는 모르겠으나 변화도에 대한 누적?을 방지하기 위한 것 처럼 보임
        w2.grad.zero_()