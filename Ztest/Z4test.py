#Fast Fourier Transform(고속 푸리에 변환)
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft, fftshift
from IPython.display import (Audio, display)

#주파수 = 주기의 역수
#주기 = 주파수의 역수
def beat(A, B, fc, df, dur, fs=11025):
    #0 ~ 3까지 1 / fs의 단위로 t값을 쪼갠다.
    #np.arange, np.linspace 두 개가 서로 비슷해보임
    #그러나 linspace는 쪼개는 개수를 선택
    #arange는 쪼갠 길이값을 선택
    #linspace(0, 3, 3001) == arange(0, 3, 0.001)
    t = np.arange(0, dur, 1 / fs)
    print(t)
    #fc = 440, df = 5 -> 435 Hz
    x1 = A * np.cos(2 * np.pi * (fc - df) * t)
    #445 Hz
    x2 = B * np.cos(2 * np.pi * (fc + df) * t)
    #두 주파수의 차이가 많이 나지 않으면 맥놀이가 발생
    #그래서 우우웅 우웅 하는 소리가 나게 된다.
    x = x1 + x2
    return x, t

fc, delf, fs, dur = 440, 5, 11025, 3
#맥놀이가 발생하는 소리를 만든다.
x, t = beat(1, 0.5, fc, delf, dur)
#해당 파형(시간)을 그래프로 그린다.
#x축(시간), y축(진폭)
fig, axes = plt.subplots(2, 1, figsize=(6, 3.5))
axes[0].plot(t * 1000, x)
axes[0].set_xlim(0, 200)
axes[0].set_ylim(-2, 2)

display(Audio(data=x, rate=fs, autoplay=True))

#푸리에 변환을 수행한 결과를 보여준다.
#푸리에 변환은 x축(주파수), y축(진폭)
Nfft = 2048
#푸리에 변환을 하는 구간
X = fftshift(fft(x)) / fs
f = np.linspace(-fs / 2, fs / 2, len(X), endpoint=False)
#np.abs()는 절대값을 만들어줌
axes[1].plot(f, np.abs(X))
axes[1].set_xlim(-500, 500)
fig.show()
plt.show()