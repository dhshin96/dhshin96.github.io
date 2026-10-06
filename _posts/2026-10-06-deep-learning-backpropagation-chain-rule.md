---
title: "딥러닝 입문 ③: 역전파와 연쇄법칙 손계산·Python 검산"
date: 2026-10-06 20:00:00 +0900
categories: [학습과 모델링]
tags: [딥러닝, 역전파, 연쇄법칙, 계산 그래프, Python]
description: "작은 계산 그래프에서 순전파로 손실을 구하고, 연쇄법칙으로 가중치와 편향의 기울기를 뒤에서부터 계산합니다. 손계산 결과를 순수 Python과 수치 미분으로 검산합니다."
reading_time: 10
series: "처음 시작하는 딥러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 순전파가 남긴 계산 경로를 거꾸로 따라가며 연쇄법칙으로 기울기를 구하고, 손계산과 Python 출력이 일치하는지 확인한다.

입력 $x=2$, 가중치 $w=3$, 편향 $b=-1$, 정답 $y=7$인 작은 신경망을 생각해 보자. 순전파 결과는 $z=5$, ReLU 출력은 $a=5$, 손실은 **2.0**이다. 이 계산을 역방향으로 따라가면 손실의 가중치 기울기는 **$-4$**, 편향 기울기는 **$-2$**가 된다. 이번 글은 이 두 숫자가 어디서 나오는지 계산 그래프 한 장으로 확인한다.

앞 글에서 [MSE가 예측과 정답의 차이를 손실로 바꾸는 과정]({{ '/posts/deep-learning-loss-functions-mse-cross-entropy/' | relative_url }})을 계산했다. 이제 손실이 각 매개변수에 얼마나 민감한지 구할 차례다. 역전파는 매개변수를 직접 바꾸는 규칙이 아니라, **손실의 기울기를 효율적으로 계산하는 방법**이다. 실제로 값을 어떻게 바꿀지는 다음 글의 경사하강법에서 다룬다.

## 1. 직관: 계산 영수증을 끝에서부터 읽는다

신경망의 순전파는 입력에서 출력으로 계산한다. 이 예제에서는 `곱하기 → 더하기 → ReLU → 손실` 순서다. 각 연산은 입력값뿐 아니라, 자신의 출력이 입력에 얼마나 민감한지를 나타내는 **국소 미분값**도 가진다.

역전파는 손실에서 출발해 이 연산들을 반대 순서로 방문한다. 뒤에서 받은 기울기에 현재 연산의 국소 미분값을 곱해 앞쪽으로 보낸다. 영수증의 마지막 합계에서 시작해 어느 항목이 합계에 얼마나 영향을 주었는지 거슬러 올라가는 셈이다.

여기서 기울기의 부호와 크기를 구분해야 한다. $partial L/partial w=-4$는 현재 지점에서 $w$를 아주 조금 키우면 손실이 감소하는 방향이라는 뜻이다. 절댓값 4는 그 근처의 민감도다. 실제 이동량은 학습률과 최적화 방법까지 정해야 결정된다.

## 2. 기호와 수식: 국소 미분값을 곱한다

계산 그래프를 다음처럼 정의한다.

$$
z=wx+b, \qquad a=\operatorname{ReLU}(z), \qquad L=\frac{1}{2}(a-y)^2
$$

$L$은 마지막 출력 $a$를 거쳐 $z$, 다시 $w$로 이어진 합성함수다. 따라서 연쇄법칙을 적용하면 다음과 같다.

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial a}
 \frac{\partial a}{\partial z}
 \frac{\partial z}{\partial w}
$$

이 식에서 각 항은 가까이 붙은 한 연산의 국소 미분값이다. $partial L/\partial a=a-y$, $z>0$일 때 ReLU의 미분은 $partial a/\partial z=1$, 그리고 $partial z/\partial w=x$다. 편향은 $partial z/\partial b=1$이므로 같은 방식으로 계산한다.

## 3. 작은 손계산: 손실 2.0에서 기울기 −4까지

먼저 왼쪽에서 오른쪽으로 순전파한다.

| 단계 | 계산 | 결과 |
| --- | --- | ---: |
| 선형 결합 | $z=3\times2-1$ | $5$ |
| ReLU | $a=\max(0,5)$ | $5$ |
| 손실 | $L=\frac{1}{2}(5-7)^2$ | $2$ |

이제 손실에서 시작해 오른쪽에서 왼쪽으로 역전파한다.

$$
\frac{\partial L}{\partial a}=5-7=-2, \qquad
\frac{\partial a}{\partial z}=1
$$

따라서 $partial L/\partial z=(-2)\times1=-2$다. 선형 결합의 국소 미분값 $partial z/\partial w=x=2$, $partial z/\partial b=1$, $partial z/\partial x=w=3$을 차례로 곱하면 다음 결과를 얻는다.

$$
\frac{\partial L}{\partial w}=-2\times2=-4, \qquad
\frac{\partial L}{\partial b}=-2\times1=-2, \qquad
\frac{\partial L}{\partial x}=-2\times3=-6
$$

<figure class="study-figure">
  <img src="{{ '/assets/images/deep-learning/backpropagation-chain-rule.svg' | relative_url }}" alt="입력 2와 가중치 3, 편향 마이너스 1을 순전파해 z 5, ReLU 출력 5, 손실 2를 구한 뒤, 손실 기울기 마이너스 2를 역방향으로 전달해 가중치 기울기 마이너스 4와 편향 기울기 마이너스 2를 얻는 계산 그래프">
  <figcaption>그림 1. 순전파는 값을 오른쪽으로 계산하고, 역전파는 국소 미분값을 곱하며 기울기를 왼쪽으로 전달한다.</figcaption>
</figure>

## 4. 실행 가능한 Python 실습

다음 코드는 외부 라이브러리 없이 순전파와 역전파를 직접 구현한다. 마지막에는 $w$를 아주 조금 흔들어 계산한 수치 미분으로 손계산 기울기를 검산한다.

```python
def forward(x, w, b, target):
    z = w * x + b
    activation = max(0.0, z)
    loss = 0.5 * (activation - target) ** 2
    return z, activation, loss


x, w, b, target = 2.0, 3.0, -1.0, 7.0
z, activation, loss = forward(x, w, b, target)

grad_activation = activation - target
grad_z = grad_activation * (1.0 if z > 0 else 0.0)
grad_w = grad_z * x
grad_b = grad_z
grad_x = grad_z * w

epsilon = 1e-5
loss_plus = forward(x, w + epsilon, b, target)[2]
loss_minus = forward(x, w - epsilon, b, target)[2]
numerical_grad_w = (loss_plus - loss_minus) / (2 * epsilon)

print(f"z={z:.1f}, activation={activation:.1f}, loss={loss:.1f}")
print(f"grad_w={grad_w:.1f}, grad_b={grad_b:.1f}, grad_x={grad_x:.1f}")
print(f"numerical_grad_w={numerical_grad_w:.6f}")
print(f"difference={abs(grad_w - numerical_grad_w):.2e}")
```

실행 출력은 다음과 같다.

```text
z=5.0, activation=5.0, loss=2.0
grad_w=-4.0, grad_b=-2.0, grad_x=-6.0
numerical_grad_w=-4.000000
difference=2.62e-11
```

## 5. 결과 해석: 자동 미분도 같은 경로를 따른다

손계산과 수치 미분의 $w$ 기울기가 소수 여섯 자리까지 같다. 두 값의 차이 $2.62\times10^{-11}$은 유한한 간격과 부동소수점 계산에서 생긴 작은 오차다. 이렇게 해석적 기울기를 수치 미분과 비교하는 검사는 역전파 구현의 실수를 찾는 데 유용하다.

딥러닝 프레임워크의 자동 미분도 핵심 원리는 같다. 순전파 중 연산으로 계산 그래프를 만들고, 스칼라 손실에서 시작해 연쇄법칙으로 그래프를 거꾸로 따라가며 기울기를 계산한다. 다만 실제 모델에서는 텐서 연산과 여러 갈래 경로를 다뤄야 하므로 프레임워크가 필요한 중간값과 미분 규칙을 관리한다.

첫 글의 [퍼셉트론과 ReLU 계산]({{ '/posts/deep-learning-perceptron-relu/' | relative_url }})을 다시 보면 활성화함수가 역전파에서 문처럼 작동하는 이유가 선명해진다. 이 예제는 $z=5>0$이라 ReLU의 국소 미분값이 1이었다. $z<0$이면 그 값은 0이 되어 이 경로의 기울기도 0이 된다. $z=0$에서의 미분값은 구현이 정한 관례를 따른다.

## 6. 직접 바꿔 보는 연습

먼저 정답 `target`을 `4.0`으로 바꿔 보자. $partial L/\partial a$의 부호가 양수로 바뀌므로 `grad_w`도 양수가 될 것이라고 예상한 뒤 실행한다. 다음으로 `w=-1.0`, `b=-1.0`을 넣어 $z<0$인 경우를 확인한다. ReLU가 0을 출력하고 이 경로의 `grad_w`가 0이 되는지 살펴본다.

마지막으로 `epsilon`을 `1e-2`, `1e-5`, `1e-8`로 바꾸어 차이를 비교한다. 간격이 너무 크면 곡선의 국소 기울기를 거칠게 근사하고, 너무 작으면 부동소수점 반올림 오차가 커질 수 있다. 수치 미분은 학습에 쓰기보다 작은 예제의 기울기 검산에 적합하다.

## 핵심 요약

- 순전파는 계산 그래프의 값을 입력에서 손실 방향으로 구한다.
- 역전파는 손실에서 출발해 국소 미분값을 곱하며 매개변수 기울기를 계산한다.
- 예제에서 손실은 2.0이고, 연쇄법칙으로 구한 기울기는 $\partial L/\partial w=-4$, $\partial L/\partial b=-2$다.
- 순수 Python 역전파와 수치 미분이 일치하므로 계산 경로를 검산할 수 있다.

## 다음 글 예고

다음 글에서는 경사하강법과 옵티마이저를 다룬다. 역전파로 얻은 기울기에 학습률을 적용해 가중치를 갱신하고, 확률적 경사하강법과 모멘텀의 차이를 작은 수치 예제로 확인한다.

## 참고 자료

1. [Dive into Deep Learning, Forward Propagation, Backward Propagation, and Computational Graphs](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html) — 계산 그래프를 역순으로 순회하며 연쇄법칙으로 기울기를 구하는 공개 교과서 설명.
2. [PyTorch 공식 튜토리얼, Automatic Differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) — 순전파 중 계산 그래프를 만들고 `backward()`로 기울기를 계산하는 공식 설명.
3. [Deep Learning, Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html) — 역전파를 학습 전체가 아닌 기울기 계산 알고리즘으로 구분하고 계산 절차를 설명한 공개 교과서.
