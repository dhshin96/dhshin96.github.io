---
title: "머신러닝 입문 ⑩: 과대적합과 L2 규제 학습·검증 오차 비교"
date: 2026-11-23 20:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [머신러닝, 과대적합, 규제, L2, 릿지회귀, 일반화, Python]
description: "과대적합을 학습 오차와 검증 오차의 차이로 확인합니다. 8차 다항식에 L2 규제를 적용해 검증 MSE가 0.188에서 0.008로 줄어드는 과정을 순수 Python으로 재현합니다."
reading_time: 12
series: "처음 시작하는 머신러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 학습 오차가 0이어도 새 자료의 오차가 클 수 있는 이유를 이해하고, L2 규제가 큰 계수를 누르는 과정을 계산한다.

9개 점에 8차 다항식을 맞추면 학습 MSE는 **0.000**이지만 검증 MSE는 **0.188**이다. L2 규제 강도를 $\alpha=0.01$로 두면 학습 MSE는 **0.025**로 늘지만 검증 MSE는 **0.008**로 줄어든다. 학습 자료를 완벽히 외우는 대신 새 점에서 더 안정적인 예측을 얻은 결과다.

앞 글의 [K-겹 교차검증 평균과 표준편차]({{ '/posts/machine-learning-k-fold-cross-validation/' | relative_url }})는 검증 점수가 분할에 따라 흔들리는 모습을 보여 줬다. 이번에는 모델이 지나치게 유연할 때 학습 점수와 검증 점수가 왜 벌어지는지 살펴본다.

## 1. 직관: 모든 점을 지나가는 곡선이 답은 아니다

학습 자료는 직선 $y=1+2x$ 주변의 아홉 점이다. 가운데 점들은 위아래로 조금 흔들려 있다. 8차 다항식에는 계수가 아홉 개 있으므로 아홉 학습점을 모두 지나는 곡선을 만들 수 있다. 문제는 이 곡선이 직선 관계뿐 아니라 작은 흔들림까지 규칙처럼 따라간다는 점이다.

**과대적합(overfitting)**은 모델이 학습 자료에서는 매우 작은 오차를 내지만, 학습에 쓰지 않은 자료에서는 오차가 커지는 상태다. 학습 오차만 보면 발견하기 어렵다. [학습·검증·테스트 자료의 역할]({{ '/posts/machine-learning-train-validation-test-split/' | relative_url }})처럼 모델 선택에 쓰지 않은 검증 자료를 함께 봐야 한다.

학습 MSE와 검증 MSE의 차이가 몇 이상이면 과대적합이라는 고정 기준은 없다. 오차의 단위와 자료의 잡음 수준이 문제마다 다르기 때문이다. 대신 같은 평가 방법으로 모델 복잡도나 규제 강도를 바꾸면서 두 오차의 흐름을 비교한다. 모델이 복잡해질수록 학습 오차는 계속 줄지만 검증 오차가 다시 커진다면, 그 지점부터는 학습 자료에만 있는 흔들림을 따라갔다는 신호로 볼 수 있다.

<figure class="study-figure">
  <img src="{{ '/assets/images/machine-learning/overfitting-l2-regularization.svg' | relative_url }}" alt="8차 다항식에서 L2 규제 강도 알파가 0, 0.01, 10일 때의 곡선과 학습점. 알파 0은 학습 MSE 0.000과 검증 MSE 0.188, 알파 0.01은 0.025와 0.008, 알파 10은 0.521과 0.563이다">
  <figcaption>그림 1. 규제가 없으면 잡음까지 굽어 따라가고, 규제가 너무 강하면 필요한 기울기까지 줄어든다.</figcaption>
</figure>

## 2. 기호와 수식: 오차에 계수 크기를 더한다

8차 다항식은 입력 $x$로 다음 예측을 만든다.

$$
\hat y=w_0+w_1x+w_2x^2+\cdots+w_8x^8
$$

보통 최소제곱법은 잔차제곱합만 작게 한다. L2 규제를 적용한 릿지 회귀는 여기에 절편을 제외한 계수 제곱합을 더한다.

$$
J(\mathbf{w})=\sum_{i=1}^{n}(y_i-\hat y_i)^2+\alpha\sum_{j=1}^{8}w_j^2
$$

$\alpha$는 규제 강도다. $\alpha=0$이면 일반 최소제곱과 같고, 값이 커질수록 큰 계수를 더 비싸게 취급한다. 계수가 작아지면 곡선이 작은 흔들림마다 급하게 꺾이기 어려워진다. 다만 규제가 지나치면 실제 관계에 필요한 계수도 줄어 **과소적합**이 생긴다.

L2 규제는 차수를 직접 낮추지 않는다. 8차 항까지 그대로 두되 모든 계수가 목적함수 안에서 비용을 치르게 한다. 따라서 모델은 높은 차수의 항을 꼭 써야 얻는 오차 감소가 규제 비용보다 클 때만 그 계수를 크게 유지한다. 이는 복잡한 표현을 완전히 금지하는 방식과 다르다. 필요한 만큼은 남기고, 학습점 하나를 맞추기 위해 과도하게 커지는 계수는 줄이는 절충이다.

## 3. 작은 손계산: 완벽한 암기보다 작은 목적함수

규제가 없는 8차식의 계수 제곱합은 2218.456이다. 이 해를 $\alpha=0.01$인 목적함수에 넣으면 학습 잔차제곱합은 0이지만 규제 항만 $0.01\times2218.456=22.185$가 된다.

반면 $\alpha=0.01$로 다시 학습한 모델은 학습 MSE가 0.025이므로 아홉 점의 잔차제곱합은 약 $9\times0.025=0.225$다. 계수 제곱합 4.820의 규제 항은 0.048이다. 반올림 전 값으로 계산한 전체 목적함수는 약 0.272로, 22.185보다 훨씬 작다. 약간의 학습 오차를 허용하고 큰 계수를 크게 줄인 선택이다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 패키지 없이 8차 다항식의 정규방정식을 가우스 소거법으로 푼다. 절편 $w_0$에는 규제를 적용하지 않고, 같은 학습·검증 자료에서 세 규제 강도를 비교한다.

```python
train_x = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
noise = [0.0, 0.25, -0.2, 0.2, 0.0, -0.2, 0.2, -0.25, 0.0]
train_y = [1 + 2 * x + e for x, e in zip(train_x, noise)]

valid_x = [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875]
valid_y = [1 + 2 * x for x in valid_x]

def solve(matrix, vector):
    n = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        for j in range(column, n + 1):
            augmented[column][j] /= scale
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            for j in range(column, n + 1):
                augmented[row][j] -= factor * augmented[column][j]
    return [augmented[i][-1] for i in range(n)]

def fit_polynomial(alpha, degree=8):
    design = [[x ** power for power in range(degree + 1)] for x in train_x]
    normal = [
        [sum(row[i] * row[j] for row in design) for j in range(degree + 1)]
        for i in range(degree + 1)
    ]
    for i in range(1, degree + 1):
        normal[i][i] += alpha
    target = [
        sum(row[i] * y for row, y in zip(design, train_y))
        for i in range(degree + 1)
    ]
    return solve(normal, target)

def predict(x, weights):
    return sum(weight * x ** power for power, weight in enumerate(weights))

def mse(x_values, y_values, weights):
    return sum(
        (y - predict(x, weights)) ** 2
        for x, y in zip(x_values, y_values)
    ) / len(x_values)

for alpha in [0.0, 0.01, 10.0]:
    weights = fit_polynomial(alpha)
    train_mse = mse(train_x, train_y, weights)
    valid_mse = mse(valid_x, valid_y, weights)
    weight_norm = sum(weight ** 2 for weight in weights[1:])
    print(
        f"alpha={alpha:.2f} | 학습 MSE={train_mse:.3f} | "
        f"검증 MSE={valid_mse:.3f} | 계수 제곱합={weight_norm:.3f}"
    )
```

실행 출력은 다음과 같다.

```text
alpha=0.00 | 학습 MSE=0.000 | 검증 MSE=0.188 | 계수 제곱합=2218.456
alpha=0.01 | 학습 MSE=0.025 | 검증 MSE=0.008 | 계수 제곱합=4.820
alpha=10.00 | 학습 MSE=0.521 | 검증 MSE=0.563 | 계수 제곱합=0.323
```

## 5. 결과 해석: 규제 강도도 검증으로 고른다

$\alpha=0$ 모델은 아홉 학습점을 모두 통과하지만 중간의 여덟 검증점에서는 크게 흔들린다. 반대로 $\alpha=0.01$ 모델은 학습 오차를 조금 허용하면서 계수 제곱합을 약 460분의 1로 줄였고, 검증 MSE도 0.008로 낮췄다. 규제가 새 자료의 정답을 미리 본 것이 아니라, 작은 잡음에 반응하려면 필요했던 큰 계수를 억제한 것이다.

검증점은 학습점 사이의 중간 위치에 두었고 정답은 자료를 만든 직선에서 계산했다. 그래서 이 실습은 관측 구간을 멀리 벗어나는 외삽보다, 같은 구간 안에서 잡음에 덜 흔들리는지를 비교한다. 실제 문제에서는 숨은 직선을 알 수 없으므로 별도로 남긴 검증 자료의 실제 정답으로 오차를 계산한다. 한 번의 분할이 우연히 쉽거나 어려울 수 있다면 앞 글의 K-겹 교차검증으로 여러 폴드의 평균 검증 오차를 비교한다.

$\alpha=10$에서는 학습과 검증 MSE가 모두 0.5를 넘는다. 규제가 강할수록 항상 좋은 것이 아니다. 규제 강도는 학습 자료에서 직접 맞추는 계수가 아니라, 검증이나 교차검증으로 비교하는 **하이퍼파라미터**다. 여러 $\alpha$를 비교한 뒤 최종 테스트 세트는 선택이 끝났을 때 한 번만 사용한다.

선택한 $\alpha$가 모든 새 자료에서 반드시 더 좋은 성능을 보장하지도 않는다. 검증 오차는 주어진 표본에서 일반화 성능을 추정한 값이다. 후보를 지나치게 많이 시험하면 검증 세트에도 간접적으로 맞출 수 있으므로, 탐색 범위를 미리 정하고 마지막 테스트 평가를 분리해야 한다. 이 경계가 학습 오차 0보다 검증 절차를 더 중요하게 만드는 이유다.

이 예제는 입력을 -1에서 1 사이로 맞췄다. L2 규제는 계수 크기를 비교하므로 특성의 단위가 크게 다르면 같은 $\alpha$라도 특성마다 실제 압박이 달라진다. 여러 특성을 쓰는 모델에서는 표준화를 학습 폴드 안에서 맞추고 검증 폴드에는 그 변환만 적용해야 한다. 기본 직선과 최소제곱법이 낯설다면 [선형회귀의 기울기·절편 계산]({{ '/posts/machine-learning-linear-regression-least-squares/' | relative_url }})을 먼저 확인하면 목적함수의 차이가 선명해진다.

## 6. 직접 바꿔 보는 연습

1. `alpha` 목록에 `0.001`, `0.1`, `1.0`을 넣고 검증 MSE가 가장 작은 값을 찾는다.
2. `noise`의 모든 값을 0으로 바꾼 뒤 규제가 없는 모델과 규제 모델을 비교한다.
3. 8차 대신 3차 다항식을 사용하고 학습·검증 오차와 계수 제곱합이 어떻게 달라지는지 확인한다.

## 핵심 요약

- 과대적합은 학습 오차는 작지만 학습하지 않은 자료의 오차가 큰 상태다.
- L2 규제는 손실에 계수 제곱합을 더해 큰 계수와 급한 곡선을 억제한다.
- 규제가 너무 강하면 과소적합하므로 $\alpha$는 검증이나 교차검증으로 고른다.

## 다음 글 예고

다음 글에서는 **학습곡선과 데이터 크기**를 다룬다. 학습 자료 수를 늘리며 학습·검증 오차가 어떻게 가까워지는지 보고, 과대적합과 과소적합을 구분한다.

## 참고 자료

- scikit-learn, [Underfitting vs. Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html) — 다항식 차수와 교차검증 MSE로 과소적합·과대적합을 비교한 공식 예제.
- scikit-learn, [Ridge API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) — L2 규제 목적함수와 `alpha`의 의미를 정의한 공식 문서.
- scikit-learn, [Effect of model regularization on training and test error](https://scikit-learn.org/stable/auto_examples/model_selection/plot_train_error_vs_test_error.html) — 규제 강도에 따른 학습·검증 성능의 균형을 설명한 공식 예제.
