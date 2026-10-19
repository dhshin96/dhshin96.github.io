---
title: "머신러닝 입문 ④: 로지스틱 회귀 시그모이드와 로그 손실 손계산"
date: 2026-10-12 20:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [머신러닝, 로지스틱 회귀, 시그모이드, 이진 분류, 로그 손실, 경사하강법, Python]
description: "로지스틱 회귀가 선형 점수를 시그모이드로 확률로 바꾸는 원리를 배웁니다. 여섯 개의 합성 자료로 로그 손실을 0.693에서 0.051로 낮추고 이진 분류를 재현합니다."
reading_time: 12
series: "처음 시작하는 머신러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 선형 점수를 0과 1 사이의 확률로 바꾸는 시그모이드 함수와, 정답 확률을 높이도록 모델을 학습시키는 로그 손실을 연결한다.

입력 `0, 1, 2, 3, 4, 5`와 이진 레이블 `0, 0, 0, 1, 1, 1`을 사용해 로지스틱 회귀를 학습했다. 순수 Python 경사하강법을 2,000번 실행하자 평균 로그 손실은 **0.693에서 0.051**로 줄었다. 학습된 확률은 약 `0.000, 0.005, 0.162, 0.888, 0.997, 1.000`이고, 기준값 0.5로 나눈 예측은 여섯 레이블과 모두 같았다.

앞 글에서 만든 [선형회귀의 기울기와 절편]({{ '/posts/machine-learning-linear-regression-least-squares/' | relative_url }})은 숫자를 그대로 예측했다. 로지스틱 회귀도 먼저 $z=wx+b$라는 선형 점수를 만들지만, 그 점수를 확률로 바꾸는 단계가 하나 더 있다.

## 1. 직관: 직선의 점수를 확률로 접어 넣기

선형 점수 $z$는 음의 무한대부터 양의 무한대까지 갈 수 있다. 그대로는 확률로 읽기 어렵다. 시그모이드 함수는 큰 음수는 0에 가깝게, 큰 양수는 1에 가깝게 보내고, $z=0$은 정확히 0.5로 보낸다. 그래서 점수의 순서는 유지하면서 범위만 확률에 맞게 바꿀 수 있다.

예를 들어 $z=2x-5$라 두자. $x=2$이면 $z=-1$과 확률 0.269, $x=2.5$이면 $z=0$과 확률 0.5, $x=3$이면 $z=1$과 확률 0.731을 얻는다. 확률이 0.5 이상이면 클래스 1로 정한다는 규칙에서는 앞의 예만 클래스 0이다.

<figure class="study-figure">
  <img src="{{ '/assets/images/machine-learning/logistic-regression-sigmoid-log-loss.svg' | relative_url }}" alt="입력 x가 선형 점수 z=2x-5, 시그모이드 확률, 0.5 기준 이진 분류로 이어지는 과정. x가 2, 2.5, 3일 때 확률은 각각 0.269, 0.500, 0.731이고 예측 클래스는 0, 1, 1이다">
  <figcaption>그림 1. 로지스틱 회귀는 선형 점수를 시그모이드로 확률화한 뒤, 별도의 기준값으로 클래스를 정한다.</figcaption>
</figure>

## 2. 기호와 수식: 시그모이드와 로그 손실

특성 하나를 쓰는 로지스틱 회귀의 선형 점수와 클래스 1의 예측 확률은 다음과 같다.

$$
z_i=wx_i+b, \qquad p_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}}
$$

$w$는 입력이 점수에 미치는 방향과 크기, $b$는 기준 위치를 정한다. $z=0$이면 $e^0=1$이므로 $p=1/(1+1)=0.5$다. 이 모델의 이름에 회귀가 들어가지만, 결과는 연속적인 확률이고 최종 목적은 클래스 0 또는 1을 나누는 분류다.

확률을 다른 각도에서 보면 선형 점수의 뜻이 더 분명해진다. 클래스 1의 확률이 $p$일 때 **오즈**는 $p/(1-p)$이고, 그 로그를 취한 **로그 오즈**가 바로 $z$다.

$$
\log\left(\frac{p_i}{1-p_i}\right)=wx_i+b
$$

$z=0$이면 오즈가 1이어서 두 클래스의 확률이 같다. $z$가 1만큼 커질 때 오즈는 $e$배가 되지만, 확률의 증가 폭은 현재 위치에 따라 달라진다. 그래서 $z$가 0 근처일 때는 확률이 빠르게 변하고, 이미 0이나 1에 가까울 때는 같은 점수 변화에도 확률이 조금만 변한다. 계수 $w$는 확률 자체가 일정하게 늘어난다는 뜻이 아니라 로그 오즈가 일정하게 변한다는 뜻이다.

정답 $y_i$가 0 또는 1일 때 한 자료의 이진 로그 손실은 다음 식으로 두 경우를 한 번에 표현한다.

$$
\ell_i=-\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
$$

정답이 1이면 손실은 $-\log(p_i)$, 정답이 0이면 $-\log(1-p_i)$만 남는다. 정답 클래스에 높은 확률을 주면 손실은 0에 가까워지고, 틀린 답을 매우 확신하면 손실이 크게 늘어난다. 제곱오차와 교차엔트로피의 차이는 [손실함수 입문 글]({{ '/posts/deep-learning-loss-functions-mse-cross-entropy/' | relative_url }})에서 더 자세히 비교할 수 있다.

## 3. 작은 손계산: 같은 오차와 다른 레이블

$z=2x-5$에서 $x=2$의 확률은 $sigma(-1)\approx0.269$다. 정답이 0이라면 손실은 $-\log(1-0.269)\approx0.313$이다. $x=3$의 확률은 $sigma(1)\approx0.731$이고 정답이 1이라면 손실은 $-\log(0.731)\approx0.313$이다.

두 예는 결정 경계 $z=0$에서 같은 거리만큼 떨어져 있어 손실도 같다. 반면 정답이 1인데 확률을 0.1만 주면 손실은 $-\log(0.1)\approx2.303$으로 커진다. 로그 손실은 단순히 맞았는지만 보지 않고, 정답에 얼마만큼의 확률을 주었는지까지 평가한다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 패키지 없이 평균 로그 손실의 기울기를 계산해 $w$와 $b$를 조금씩 갱신한다. 자료와 초기값, 반복 횟수가 고정되어 실행 결과도 같다.

초기 $w=b=0$에서는 모든 입력의 확률이 0.5다. 정답이 1인 자료는 확률을 올리고, 정답이 0인 자료는 확률을 내리는 방향으로 매개변수를 움직여야 한다. 코드의 `prob - label`은 그 방향을 담은 오차다. 이를 입력과 곱해 평균 낸 `dw`는 기울기 $w$를, 그대로 평균 낸 `db`는 절편 $b$를 어느 쪽으로 바꿀지 알려 준다. 한 번에 정답을 구하는 대신 작은 갱신을 반복하면서 로그 손실이 낮아지는 지점을 찾는 방식이 경사하강법이다.

```python
import math

x = [0, 1, 2, 3, 4, 5]
y = [0, 0, 0, 1, 1, 1]
w, b = 0.0, 0.0
learning_rate = 0.2

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def log_loss(labels, probabilities):
    losses = [
        -(label * math.log(prob) + (1 - label) * math.log(1 - prob))
        for label, prob in zip(labels, probabilities)
    ]
    return sum(losses) / len(losses)

initial_probabilities = [sigmoid(w * xi + b) for xi in x]
initial_loss = log_loss(y, initial_probabilities)

for _ in range(2000):
    probabilities = [sigmoid(w * xi + b) for xi in x]
    dw = sum((prob - label) * xi for xi, label, prob in zip(x, y, probabilities)) / len(x)
    db = sum(prob - label for label, prob in zip(y, probabilities)) / len(x)
    w -= learning_rate * dw
    b -= learning_rate * db

probabilities = [sigmoid(w * xi + b) for xi in x]
predictions = [int(prob >= 0.5) for prob in probabilities]

print(f"w={w:.3f}, b={b:.3f}")
print("확률:", [round(prob, 3) for prob in probabilities])
print(f"로그 손실: {initial_loss:.3f} -> {log_loss(y, probabilities):.3f}")
print("예측:", predictions)
print(f"x=2.5 확률: {sigmoid(w * 2.5 + b):.3f}")
```

실행 출력은 다음과 같다.

```text
w=3.710, b=-9.063
확률: [0.0, 0.005, 0.162, 0.888, 0.997, 1.0]
로그 손실: 0.693 -> 0.051
예측: [0, 0, 0, 1, 1, 1]
x=2.5 확률: 0.553
```

## 5. 결과 해석: 확률과 분류 기준은 다르다

학습된 $w$가 양수이므로 입력이 커질수록 클래스 1의 확률도 커진다. $x=2$의 확률 0.162와 $x=3$의 확률 0.888 사이에서 결정이 바뀌며, $x=2.5$에서는 0.553이다. 다만 여섯 자료에 잘 맞았다는 사실만으로 새 자료에서도 같은 성능을 보장하지 않는다. [학습·검증·테스트 분리]({{ '/posts/machine-learning-train-validation-test-split/' | relative_url }})처럼 보지 않은 자료로 확인해야 한다.

또한 0.5는 학습된 확률을 클래스로 바꾸는 한 가지 기준일 뿐이다. 기준을 0.8로 올리면 더 높은 확률에서만 클래스 1로 판단하고, 예측 결과가 달라질 수 있다. 어떤 기준이 적절한지는 거짓 양성과 거짓 음성의 비용, 검증 자료의 평가 지표를 함께 보고 정한다. 이 작은 예의 확률도 실제 세계의 정확한 빈도라고 단정할 수 없으며, 확률 보정은 별도의 검증 주제다.

## 6. 직접 바꿔 보는 연습

1. 반복 횟수를 20, 200, 2,000으로 바꾸고 로그 손실과 확률이 어떻게 달라지는지 기록한다.
2. 분류 기준을 0.5에서 0.8로 바꾸고 어느 입력의 예측이 달라지는지 확인한다.
3. 레이블 하나를 반대로 바꾼 뒤 학습된 $w$, $b$, 로그 손실을 비교한다.

## 핵심 요약

- 로지스틱 회귀는 선형 점수 $z=wx+b$를 시그모이드에 넣어 클래스 1의 확률을 만든다.
- 로그 손실은 정답 클래스에 낮은 확률을 줄수록 큰 벌점을 주며, 경사하강법은 평균 손실을 줄이는 방향으로 매개변수를 갱신한다.
- 확률을 클래스로 바꾸는 기준값은 모델의 확률 계산과 구별하고, 보지 않은 검증·테스트 자료에서 선택하고 평가한다.

## 다음 글 예고

다음 글에서는 [결정트리의 분할 기준과 지니 불순도]({{ '/posts/machine-learning-decision-tree-gini-split/' | relative_url }})를 다룬다. 한 번에 하나의 조건으로 자료를 나누는 과정을 따라가며, 불순도가 가장 많이 줄어드는 분할을 작은 자료에서 손으로 계산한다.

## 참고 자료

- [Google Machine Learning Crash Course, Logistic Regression](https://developers.google.com/machine-learning/crash-course/logistic-regression) — 로지스틱 회귀의 확률 예측, 시그모이드, 선형회귀와의 차이.
- [Google Machine Learning Crash Course, Loss and Regularization](https://developers.google.com/machine-learning/crash-course/logistic-regression/loss-regularization) — 이진 로그 손실의 식과 레이블·예측 확률의 의미.
- [scikit-learn API, LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) — 로지스틱 회귀 분류기와 정규화·확률 예측 API의 공식 설명.
