---
title: "머신러닝 입문 ③: 선형회귀 기울기·절편과 최소제곱법 손계산"
date: 2026-10-05 20:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [머신러닝, 선형회귀, 최소제곱법, 기울기, 절편, 평균제곱오차, Python]
description: "선형회귀가 기울기와 절편으로 숫자를 예측하는 원리를 배웁니다. 다섯 점에서 최소제곱 직선을 손으로 구하고 순수 Python으로 기울기 5.1, 절편 37.7, MSE 0.38을 재현합니다."
reading_time: 11
series: "처음 시작하는 머신러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 하나의 특성으로 숫자를 예측하는 직선 $\hat y=b_0+b_1x$를 이해하고, 다섯 점에 가장 잘 맞는 기울기와 절편을 최소제곱법으로 직접 구한다.

공부 시간과 점수의 다섯 쌍 `(1,43), (2,47), (3,54), (4,58), (5,63)`에 선형회귀를 적용하면 **기울기 5.1**, **절편 37.7**인 직선이 나온다. 이 직선의 학습 평균제곱오차(MSE)는 **0.38**이다. 정수로 맞춘 직선 $\hat y=38+5x$의 MSE 0.40보다 작다. 차이는 작지만, 최소제곱법이 눈대중 대신 무엇을 기준으로 한 직선을 고르는지 보여 준다.

이번 계산에서 학습 자료와 평가 자료의 역할을 구별해야 하는 이유는 [앞 글의 학습·검증·테스트 분리]({{ '/posts/machine-learning-train-validation-test-split/' | relative_url }})에서 먼저 다뤘다. 여기서는 학습 세트 안에서 직선을 구하는 과정에 집중한다.

## 1. 직관: 흩어진 점 사이에 예측 직선 놓기

입력 $x$가 한 시간 늘 때 예측 점수가 일정하게 변한다고 가정해 보자. 선형회귀는 그 변화량을 **기울기** $b_1$로, $x=0$일 때의 예측값을 **절편** $b_0$로 표현한다. 기울기 5.1은 공부 시간이 한 시간 늘 때 예측 점수가 5.1점 높아진다는 뜻이다. 절편 37.7은 직선이 세로축과 만나는 위치다.

각 점에서 실제값과 직선의 예측값 사이에는 세로 간격이 남는다. 이 차이 $y_i-\hat y_i$가 **잔차**다. 어떤 잔차는 양수이고 어떤 잔차는 음수이므로 그대로 더하면 서로 지워질 수 있다. 최소제곱법은 잔차를 제곱해 모두 양수로 만든 뒤, 그 합이 가장 작은 기울기와 절편을 고른다.

<figure class="study-figure">
  <img src="{{ '/assets/images/machine-learning/linear-regression-least-squares.svg' | relative_url }}" alt="공부 시간 1시간부터 5시간까지의 점수 다섯 점과 기울기 5.1, 절편 37.7인 최소제곱 회귀선. 각 실제 점과 회귀선 사이의 잔차를 세로 점선으로 표시하고 학습 MSE 0.38을 함께 제시">
  <figcaption>그림 1. 최소제곱 직선은 모든 점을 지나지 않아도 잔차제곱합이 가장 작다. 점선은 실제값과 예측값의 차이인 잔차다.</figcaption>
</figure>

## 2. 기호와 수식: 잔차제곱합을 가장 작게

입력 $x_i$ 하나로 숫자 $y_i$를 예측하는 단순 선형회귀식은 다음과 같다.

$$
\hat y_i=b_0+b_1x_i
$$

$b_1$은 기울기, $b_0$은 절편이다. 최소제곱법은 아래 잔차제곱합(SSE)을 가장 작게 하는 두 값을 찾는다.

$$
\operatorname{SSE}(b_0,b_1)=\sum_{i=1}^{n}\left(y_i-(b_0+b_1x_i)\right)^2
$$

특성이 하나이고 절편을 포함할 때 해는 평균 $\bar x,\bar y$를 이용해 계산할 수 있다.

$$
b_1=\frac{\sum_{i=1}^{n}(x_i-\bar x)(y_i-\bar y)}{\sum_{i=1}^{n}(x_i-\bar x)^2},
\qquad
b_0=\bar y-b_1\bar x
$$

분자는 $x$와 $y$가 평균에서 같은 방향으로 움직이는 정도를 모으고, 분모는 $x$ 자체가 퍼진 정도를 모은다. 모든 $x_i$가 같으면 분모가 0이 되어 기울기를 정할 수 없다.

## 3. 작은 손계산: 기울기 5.1과 절편 37.7

다섯 입력의 평균은 $\bar x=3$, 점수의 평균은 $\bar y=53$이다. 평균에서 벗어난 값을 차례대로 적으면 $x_i-\bar x=(-2,-1,0,1,2)$, $y_i-\bar y=(-10,-6,1,5,10)$이다.

따라서 분자는 $20+6+0+5+20=51$, 분모는 $4+1+0+1+4=10$이다. 기울기는 $b_1=51/10=5.1$, 절편은 $b_0=53-5.1\times3=37.7$이 된다. 완성된 직선은 $\hat y=37.7+5.1x$다.

예측값은 차례대로 42.8, 47.9, 53.0, 58.1, 63.2다. 실제값과의 잔차는 0.2, -0.9, 1.0, -0.1, -0.2이고, 잔차제곱합은 $0.04+0.81+1+0.01+0.04=1.90$이다. 이를 다섯 개로 나누면 학습 MSE는 $1.90/5=0.38$이다. MSE 계산 자체가 낯설다면 [손실함수의 평균제곱오차 손계산]({{ '/posts/deep-learning-loss-functions-mse-cross-entropy/' | relative_url }})에서 제곱과 평균의 역할을 먼저 확인할 수 있다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 패키지 없이 같은 계산을 재현한다. `fit_simple_linear_regression`은 평균, 기울기, 절편을 학습 자료에서만 구한다.

```python
x = [1, 2, 3, 4, 5]
y = [43, 47, 54, 58, 63]

def fit_simple_linear_regression(x_values, y_values):
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum(
        (xi - x_mean) * (yi - y_mean)
        for xi, yi in zip(x_values, y_values)
    )
    denominator = sum((xi - x_mean) ** 2 for xi in x_values)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept

def predict(x_value, slope, intercept):
    return intercept + slope * x_value

slope, intercept = fit_simple_linear_regression(x, y)
predictions = [predict(xi, slope, intercept) for xi in x]
mse = sum((yi - y_hat) ** 2 for yi, y_hat in zip(y, predictions)) / len(y)

print(f"기울기: {slope:.1f}")
print(f"절편: {intercept:.1f}")
print("예측값:", [round(value, 1) for value in predictions])
print(f"학습 MSE: {mse:.2f}")
print(f"6시간 예측: {predict(6, slope, intercept):.1f}점")
```

실행 출력은 다음과 같다.

```text
기울기: 5.1
절편: 37.7
예측값: [42.8, 47.9, 53.0, 58.1, 63.2]
학습 MSE: 0.38
6시간 예측: 68.3점
```

## 5. 결과 해석: 직선의 의미와 한계

기울기 5.1은 이 합성 자료 범위에서 $x$가 1 늘 때 예측값이 5.1 늘어나는 관계를 요약한다. 이것만으로 공부 시간을 늘리면 점수가 반드시 5.1점 오른다는 인과관계를 뜻하지는 않는다. 관찰하지 않은 수면, 이전 성취도 같은 변수가 함께 움직였을 수 있다.

절편 37.7도 $x=0$이 관측 범위 밖이라면 계산상 직선의 위치를 정하는 값일 뿐, 실제 0시간 점수를 정확히 설명한다고 볼 수 없다. 6시간 예측 68.3점은 관측 범위에 가까운 한 칸 바깥의 계산값이다. 더 먼 구간으로 갈수록 같은 직선 관계가 유지된다는 보장이 없으므로 외삽에는 특히 주의한다.

학습 MSE 0.38은 이 다섯 점에 얼마나 맞았는지를 말한다. 새 자료에서도 잘 맞는지는 별도의 검증·테스트 자료로 확인해야 한다. 또한 제곱오차는 큰 잔차를 더 크게 반영하므로 이상치 하나가 직선을 많이 움직일 수 있다. 먼저 산점도와 잔차를 함께 보는 습관이 필요하다.

## 6. 직접 바꿔 보는 연습

1. 마지막 점 `(5,63)`을 `(5,70)`으로 바꾸고 기울기, 절편, 학습 MSE가 얼마나 달라지는지 실행한다.
2. 모든 점수에 10을 더한다. 실행 전에 기울기와 절편 중 무엇이 바뀔지 예상한다.
3. `predict(10, slope, intercept)`를 계산하고, 10시간 예측을 그대로 믿기 어려운 이유를 관측 범위와 연결해 적는다.

## 핵심 요약

- 단순 선형회귀는 하나의 특성으로 $\hat y=b_0+b_1x$라는 숫자 예측식을 만든다.
- 최소제곱법은 잔차제곱합이 가장 작은 기울기와 절편을 고른다.
- 기울기와 절편은 관측 범위와 자료 생성 맥락 안에서 해석하고, 학습 오차만으로 새 자료의 성능을 판단하지 않는다.

지도학습에서 왜 이 식이 회귀 문제에 해당하는지는 [첫 글의 특성·레이블과 회귀·분류 구분]({{ '/posts/machine-learning-supervised-features-labels/' | relative_url }})으로 돌아가 연결해 볼 수 있다.

## 다음 글 예고

다음 글에서는 **로지스틱 회귀**를 다룬다. 직선의 출력값을 0과 1 사이의 확률로 바꾸는 시그모이드 함수를 손으로 계산하고, 기준값에 따라 이진 분류가 어떻게 결정되는지 Python으로 확인한다.

## 참고 자료

- [NIST/SEMATECH e-Handbook, Linear Least Squares Regression](https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd141.htm) — 선형 최소제곱 모형과 잔차제곱합 최소화의 정의, 외삽과 이상치의 한계.
- [Penn State STAT 501, Simple Linear Regression](https://online.stat.psu.edu/stat501/Lesson01) — 단순 선형회귀의 기울기·절편 공식과 최소제곱 기준.
- [scikit-learn User Guide, Ordinary Least Squares](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) — 선형 예측식과 `LinearRegression`이 잔차제곱합을 최소화한다는 공식 설명.
