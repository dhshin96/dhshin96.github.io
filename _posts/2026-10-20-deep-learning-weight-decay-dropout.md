---
title: "딥러닝 입문 ⑤: 가중치 감쇠와 드롭아웃 손계산·Python 실습"
date: 2026-10-20 20:00:00 +0900
categories: [학습과 모델링]
tags: [딥러닝, 정규화, 가중치-감쇠, 드롭아웃, 과적합, Python]
description: "가중치 감쇠가 손실식에 제약을 더하는 과정과 드롭아웃이 학습 중 활성값을 무작위로 끄는 과정을 구분합니다. 순수 Python으로 가중치 변화와 드롭아웃 출력의 기댓값을 검산합니다."
reading_time: 11
series: "처음 시작하는 딥러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 가중치 감쇠와 드롭아웃이 모델에 제약을 주는 위치를 구분하고, 두 방법의 계산을 작은 숫자로 재현한다.

같은 손실 $\frac{1}{2}(w-3)^2$을 10번 줄였더니 정규화가 없을 때는 $w=2.6779$였다. 가중치 감쇠 계수 0.5를 더하면 $w=1.9435$로 더 작아졌다. 한편 활성값 `[1, 2, 3, 4]`에 드롭아웃 확률 0.5를 적용해 가능한 마스크 16개를 모두 계산하니 출력 합의 평균은 원래 합과 같은 **10.0**이었다. 두 방법은 모두 과적합을 줄이려는 정규화지만, 작동 위치가 다르다.

앞 글에서 [경사하강법이 기울기로 가중치를 갱신하는 과정]({{ '/posts/deep-learning-gradient-descent-momentum/' | relative_url }})을 확인했다. 이번에는 그 갱신식에 가중치 크기를 줄이는 항을 넣고, 신경망의 중간 활성값에는 무작위 마스크를 씌운다.

## 1. 직관: 답을 외울 여지를 줄이는 두 장치

매개변수가 많은 모델은 학습 자료의 규칙뿐 아니라 우연한 흔들림까지 맞출 수 있다. 학습 오차는 작지만 처음 보는 자료의 오차가 커지는 상태가 과적합이다. 정규화는 학습 문제에 제약이나 무작위성을 더해 특정한 학습 자료에 지나치게 맞추기 어렵게 만든다.

가중치 감쇠는 큰 가중치에 비용을 부과한다. 예측 오차만 줄이면 큰 값이 유리하더라도, 가중치가 커질수록 추가 손실이 생긴다. 드롭아웃은 훈련할 때 일부 활성값을 확률적으로 0으로 만든다. 한 뉴런의 출력에 계속 의존하던 다음 층은 매번 같은 입력 조합을 받을 수 없다.

둘을 같은 방법으로 보면 안 된다. 가중치 감쇠는 **최적화할 목적함수와 가중치 갱신**을 바꾸고, 드롭아웃은 **훈련 중 순전파의 활성값**을 바꾼다. 드롭아웃은 평가할 때 끄며, 가중치 감쇠로 학습된 가중치는 평가할 때도 그대로 사용한다.

## 2. 기호와 수식: 벌점과 무작위 마스크

데이터 손실을 $L_{data}(w)$, 가중치 감쇠 계수를 $\lambda$라 하면 L2 벌점을 더한 목적함수는 다음과 같다.

$$
L_{total}(w)=L_{data}(w)+\frac{\lambda}{2}w^2
$$

이를 미분하면 기울기에 $\lambda w$가 추가된다. 학습률이 $\eta$일 때 한 번의 갱신은 다음과 같다.

$$
w \leftarrow w-\eta\left(\frac{\partial L_{data}}{\partial w}+\lambda w\right)
$$

데이터 기울기가 0이라면 $w$는 $(1-\eta\lambda)$배가 된다. 그래서 갱신할 때마다 0 쪽으로 감쇠한다. $\lambda=0$이면 원래 경사하강법으로 돌아간다. $\lambda$가 클수록 가중치 제약은 강하지만, 지나치면 필요한 신호까지 약하게 표현할 수 있다.

드롭아웃 확률을 $p$, 각 활성값을 $h_i$, 유지 여부를 나타내는 마스크를 $m_i$라 하자. 훈련 중에는 $m_i$를 유지 확률 $1-p$인 베르누이 분포에서 뽑고 다음처럼 계산한다.

$$
\tilde{h}_i=\frac{m_i h_i}{1-p}, \qquad m_i\in\{0,1\}
$$

살아남은 값을 $1/(1-p)$배 하는 방식을 역 드롭아웃이라 부른다. 이때 $E[\tilde{h}_i]=h_i$이므로 훈련 출력의 기댓값 규모가 유지된다. 평가에서는 마스크와 배율을 모두 없애고 $h_i$를 그대로 통과시킬 수 있다.

## 3. 작은 손계산: 무엇이 실제로 달라질까

$L_{data}(w)=\frac{1}{2}(w-3)^2$, $w=2$, $\lambda=0.5$라고 하자. 데이터 기울기는 $2-3=-1$이고, 감쇠 기울기는 $0.5\times2=1$이다. 둘의 합이 0이므로 이 지점에서는 가중치가 움직이지 않는다. 정규화가 없을 때의 최솟값 3과 달리, 정규화된 목적함수의 최솟값은 2가 된다.

드롭아웃에서는 $h=[1,2,3,4]$, $p=0.5$, 마스크 $m=[1,0,1,0]$을 사용해 보자. 살아남은 값을 2배 하면 출력은 `[2, 0, 6, 0]`, 합은 8이다. 한 번의 출력은 원래 합 10과 다를 수 있다. 그러나 가능한 16개 마스크를 같은 확률로 평균하면 출력 합은 10이 된다.

<figure class="study-figure">
  <img src="{{ '/assets/images/deep-learning/weight-decay-dropout.svg' | relative_url }}" alt="왼쪽은 데이터 손실의 최솟값 w 3이 가중치 감쇠 0.5를 더하면 w 2로 이동하는 계산이다. 오른쪽은 활성값 1, 2, 3, 4에 마스크 1, 0, 1, 0과 배율 2를 적용해 2, 0, 6, 0을 만드는 과정이며, 가능한 16개 마스크의 출력 합 평균은 원래 합 10과 같다.">
  <figcaption>그림 1. 가중치 감쇠는 목적함수의 최적점을 바꾸고, 드롭아웃은 훈련 순전파마다 활성값의 일부를 바꾼다.</figcaption>
</figure>

## 4. 실행 가능한 Python 실습

다음 코드는 외부 패키지나 난수 없이 두 계산을 재현한다. 드롭아웃은 마스크 하나를 확인한 뒤 가능한 마스크를 모두 열거해 기댓값을 정확히 계산한다.

```python
from itertools import product

def optimize(weight_decay, steps=10, learning_rate=0.2):
    w = 0.0
    for _ in range(steps):
        data_grad = w - 3.0
        w -= learning_rate * (data_grad + weight_decay * w)
    data_loss = 0.5 * (w - 3.0) ** 2
    return w, data_loss

for decay in (0.0, 0.5):
    w, data_loss = optimize(decay)
    print(f"weight_decay={decay:.1f}: w={w:.4f}, data_loss={data_loss:.4f}")

activations = [1.0, 2.0, 3.0, 4.0]
p = 0.5
mask = [1, 0, 1, 0]
dropped = [x * m / (1 - p) for x, m in zip(activations, mask)]
print("one mask:", [f"{x:.1f}" for x in dropped], f"sum={sum(dropped):.1f}")

all_sums = []
for bits in product((0, 1), repeat=len(activations)):
    values = [x * m / (1 - p) for x, m in zip(activations, bits)]
    all_sums.append(sum(values))
print(f"mean over all {len(all_sums)} masks: {sum(all_sums) / len(all_sums):.1f}")
print(f"original sum: {sum(activations):.1f}")
```

실행 출력은 다음과 같다.

```text
weight_decay=0.0: w=2.6779, data_loss=0.0519
weight_decay=0.5: w=1.9435, data_loss=0.5581
one mask: ['2.0', '0.0', '6.0', '0.0'] sum=8.0
mean over all 16 masks: 10.0
original sum: 10.0
```

## 5. 결과 해석: 학습 손실만으로 고르지 않는다

가중치 감쇠를 적용한 쪽의 데이터 손실 0.5581은 적용하지 않은 쪽의 0.0519보다 크다. 이것은 실패가 아니다. 학습 중에는 데이터 손실과 벌점을 합친 목적함수를 줄였기 때문이다. 정규화의 목적은 학습 자료를 가장 정확히 외우는 것이 아니라, 보지 않은 자료에서의 오차를 줄일 가능성을 높이는 데 있다.

드롭아웃의 한 마스크에서는 합이 8.0이지만 전체 마스크의 평균은 10.0이다. 기댓값이 유지된다는 말은 매번 같은 출력이 나온다는 뜻이 아니다. 이 무작위성이 학습마다 다른 부분 연결을 사용하게 한다. 실제 학습에서는 무작위 마스크를 반복해서 뽑으며, 평가 모드에서는 드롭아웃을 끈다.

$\lambda$와 $p$는 학습 데이터의 오차만 보고 정하면 안 된다. 후보 설정을 공정하게 고르는 절차는 [학습·검증·테스트 세트의 역할]({{ '/posts/machine-learning-train-validation-test-split/' | relative_url }})에서 확인할 수 있다. 검증 성능이 좋아지는 범위에서 정규화 강도를 선택하고, 테스트 세트는 마지막 평가에 남긴다.

## 6. 직접 바꿔 보는 연습

먼저 `weight_decay`를 `0.1`과 `1.0`으로 바꾸고 최종 가중치와 데이터 손실을 비교한다. 다음으로 `steps=50`으로 늘려 $\lambda=0.5$일 때 $w$가 2에 가까워지는지 확인한다.

드롭아웃에서는 `mask`를 `[0, 1, 0, 1]`로 바꿔 한 번의 합을 계산한다. 마지막으로 활성값 하나를 10으로 바꾸고, 특정 마스크의 출력 합은 크게 흔들려도 모든 마스크의 평균은 원래 합과 같은지 검산한다.

## 핵심 요약

- 가중치 감쇠는 손실에 L2 벌점을 더해 큰 가중치에 비용을 부과한다.
- 드롭아웃은 훈련 중 활성값을 확률적으로 0으로 만들고, 살아남은 값을 확대해 기댓값을 유지한다.
- 가중치 감쇠는 가중치 갱신을, 드롭아웃은 훈련 순전파를 바꾸므로 작동 위치가 다르다.
- 정규화 강도는 학습 손실의 최솟값이 아니라 검증 성능을 기준으로 고른다.

## 다음 글 예고

다음 글에서는 합성곱 신경망(CNN)을 다룬다. 작은 필터가 입력의 일부를 훑으며 특징 맵을 만드는 과정을 손계산하고 Python으로 확인한다.

## 참고 자료

1. [Dive into Deep Learning, Weight Decay](https://classic.d2l.ai/chapter_multilayer-perceptrons/weight-decay.html) — L2 벌점을 손실에 더하는 식과 가중치 감쇠 갱신을 설명한 공개 교과서.
2. [Srivastava 외, Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.jmlr.org/papers/v15/srivastava14a.html) — 훈련 중 뉴런과 연결을 무작위로 제외하는 드롭아웃을 제시한 원 논문.
3. [PyTorch 공식 문서, Dropout](https://docs.pytorch.org/docs/main/generated/torch.nn.Dropout.html) — 훈련 중 독립적인 베르누이 마스크와 $1/(1-p)$ 배율, 평가 시 항등 동작을 확인한 공식 문서.
