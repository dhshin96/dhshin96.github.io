---
title: "생존분석 입문 ③: Cox 비례위험모형과 위험비를 제대로 해석하기"
date: 2026-08-27 09:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [생존분석, Cox-비례위험모형, 위험함수, 위험비, 비례위험-가정, Python]
description: "위험함수와 생존확률의 차이를 구분하고 Cox 모형의 계수를 위험비로 바꾸는 과정을 손계산과 Python으로 재현합니다. 비례위험 가정이 요구하는 것도 함께 확인합니다."
reading_time: 12
series: "처음 시작하는 생존분석"
level: "입문"
math: true
---

> **이번 글의 목표:** 위험함수의 조건부 의미를 이해하고, Cox 비례위험모형의 계수에서 위험비를 계산해 정확히 해석한다.

“위험비가 2”라는 결과를 보면 사건이 일어날 확률도 2배라고 읽기 쉽다. 하지만 위험비는 특정 시점까지 사건 없이 남아 있는 사람들 사이의 **순간적인 사건 발생률**을 비교한다. 누적된 생존확률의 비나 전체 사건확률의 비와는 다르다. 이번 글에서는 이 구분을 먼저 세운 뒤 Cox 모형이 여러 설명변수를 어떻게 연결하는지 살펴본다.

## 1. 위험함수는 지금까지 남은 사람에게 묻는다

사건시간을 $T$라고 할 때 생존함수 $S(t)=P(T>t)$는 시점 $t$를 지나도록 사건이 없을 확률이다. 반면 **위험함수**는 $t$까지 사건 없이 남았다는 조건에서 바로 다음 짧은 구간에 사건이 발생하는 속도다.

$$
h(t)=\lim_{\Delta t\to0}
\frac{P(t\le T<t+\Delta t\mid T\ge t)}{\Delta t}
$$

위험은 확률 그 자체가 아니라 시간당 발생률이다. 따라서 1보다 클 수도 있다. 다만 아주 짧은 구간에서는 $h(t)\Delta t$를 그 구간의 조건부 사건확률로 근사할 수 있다.

예를 들어 한 달 동안 위험집합 100명이 같은 정도로 관찰되고 사건이 2건 발생했다면 위험률을 대략 $2/(100\times1)=0.02$/인월로 생각할 수 있다. 같은 기간 다른 집단에서 100인월당 4건이라면 단순 위험비는 $0.04/0.02=2$다. 이는 그 한 달의 순간적 발생 속도가 약 2배라는 뜻이지, 한 달 안에 사건을 겪을 확률이나 중앙생존시간이 정확히 2배라는 뜻은 아니다.

위험집합과 검열이 아직 낯설다면 [검열 자료와 Kaplan–Meier 계산을 다룬 생존분석 입문 ①]({{ '/posts/survival-analysis-roadmap/' | relative_url }})을 먼저 보면 분모에 누가 남는지 이해하기 쉽다.

## 2. Cox 모형은 기준 위험에 배수를 곱한다

Cox 비례위험모형은 공변량이 $x$인 대상의 위험을 다음처럼 표현한다.

$$
h(t\mid x)=h_0(t)\exp(\beta x)
$$

$h_0(t)$는 $x=0$인 기준 대상의 **기준 위험함수**다. 시간에 따라 어떤 모양으로 변할지 미리 정하지 않는다. 공변량의 효과는 $\exp(\beta x)$라는 양의 배수로 붙는다. 이처럼 기준 위험의 모양은 비모수적으로 두고 계수는 모수로 추정하기 때문에 Cox 모형을 준모수 모형이라고 부른다.

이진 변수 $x$가 0 또는 1이라면 두 집단의 위험비는 간단해진다.

$$
\frac{h(t\mid x=1)}{h(t\mid x=0)}
=\frac{h_0(t)e^\beta}{h_0(t)}=e^\beta
$$

$\beta=\log 2\approx0.693$이면 위험비는 $e^{0.693}\approx2$다. $\beta$가 0이면 위험비는 1, 음수이면 1보다 작다. 연속변수에서는 $x$가 한 단위 증가할 때 위험이 $e^\beta$배가 된다고 읽는다. 다른 공변량이 함께 들어 있다면 **나머지 공변량이 같은 조건에서**라는 문구도 빠뜨리지 않는다.

<figure class="study-figure">
  <img src="{{ '/assets/images/survival-analysis/cox-proportional-hazards.svg' | relative_url }}" alt="시간에 따라 함께 변하는 두 위험함수와 모든 시점에서 일정한 위험비 2를 보여주는 그림">
  <figcaption>그림 1. 기준 위험은 시간에 따라 변해도 두 곡선 사이의 곱셈 배수가 일정하면 비례위험 가정을 만족한다.</figcaption>
</figure>

## 3. 부분우도는 사건 시점의 순서를 이용한다

Cox 모형은 $h_0(t)$의 모양을 정하지 않고도 $\beta$를 추정한다. 사건이 발생한 시점마다 “그때 위험집합에 남아 있던 사람 중 왜 이 대상에게 사건이 관측됐는가?”를 비교하는 **부분우도**를 사용하기 때문이다.

사건 대상의 공변량이 $x_i$이고 그 시점의 위험집합이 $R_i$라면 해당 대상이 선택될 상대적 몫은 다음과 같다.

$$
\frac{\exp(\beta x_i)}
{\sum_{j\in R_i}\exp(\beta x_j)}
$$

작은 손계산을 해 보자. 위험집합에 $x=1$인 사람이 2명, $x=0$인 사람이 2명 있고 현재 추정 위험비가 2라면 가중치는 각각 2와 1이다. 다음 사건이 $x=1$ 집단에서 나올 상대적 몫은 $(2+2)/(2+2+1+1)=2/3$이다. 모든 사건 시점의 이런 몫을 곱해 가장 크게 만드는 $\beta$를 찾는다. 검열된 사람도 검열되기 전까지는 위험집합에 기여한다.

직전 글의 [Kaplan–Meier 곡선과 log-rank 검정]({{ '/posts/km-curve-logrank/' | relative_url }})은 공변량 없는 두 집단 비교에 집중했다. Cox 모형은 그 비교를 회귀 형태로 넓혀 여러 공변량의 계수와 위험비를 추정한다.

## 4. Python으로 계수와 위험비 재현하기

아래 코드는 하나의 이진 공변량과 서로 다른 사건 시점을 가진 작은 자료에서 Newton–Raphson 방법으로 Cox 부분우도를 최대화한다. `numpy`만 필요하다.

```python
import numpy as np

time = np.arange(2, 18, dtype=float)
event = np.array([1, 1, 0, 1, 1, 0, 1, 0,
                  1, 1, 0, 1, 0, 1, 0, 1])
x = np.array([1, 1, 0, 1, 0, 1, 1, 0,
              0, 1, 1, 0, 0, 0, 0, 1], dtype=float)

beta = 0.0
for _ in range(20):
    score = 0.0
    information = 0.0

    for t in time[event == 1]:
        risk = time >= t
        weight = np.exp(beta * x[risk])
        mean_x = np.sum(weight * x[risk]) / np.sum(weight)
        variance_x = (
            np.sum(weight * (x[risk] - mean_x) ** 2) / np.sum(weight)
        )
        event_x = x[(time == t) & (event == 1)][0]
        score += event_x - mean_x
        information += variance_x

    step = score / information
    beta += step
    if abs(step) < 1e-10:
        break

standard_error = 1 / np.sqrt(information)
lower = np.exp(beta - 1.96 * standard_error)
upper = np.exp(beta + 1.96 * standard_error)

print(f"beta={beta:.3f}")
print(f"hazard ratio={np.exp(beta):.3f}")
print(f"95% CI=({lower:.3f}, {upper:.3f})")
```

실행 결과는 다음과 같다.

```text
beta=0.659
hazard ratio=1.933
95% CI=(0.507, 7.375)
```

$x=1$ 집단의 추정 위험은 $x=0$ 집단의 약 1.93배다. 그러나 표본이 16명뿐이라 95% 신뢰구간이 0.507~7.375로 넓고 1을 포함한다. 따라서 점추정치만 보고 두 집단의 위험이 다르다고 단정할 수 없다. 이 결과 역시 관찰 자료라면 자동으로 인과효과가 되지 않는다.

## 5. 비례위험 가정은 무엇을 고정할까?

비례위험 가정은 각 집단의 위험이 시간에 따라 일정하다는 뜻이 아니다. $h_0(t)$와 $h(t\mid x=1)$은 함께 오르내릴 수 있다. 고정되는 것은 두 위험함수의 **비율** $e^\beta$다.

초기에는 위험비가 3인데 후기에는 0.7처럼 효과의 방향이나 크기가 크게 바뀐다면 하나의 상수 위험비가 전체 기간을 잘 설명하지 못한다. 생존곡선의 교차는 이런 가능성을 살펴볼 신호지만 그것만으로 가정을 판정하지는 않는다. 실제 분석에서는 시간에 따른 잔차 패턴과 비례위험 검정을 확인하고, 위반이 뚜렷하면 시간 상호작용이나 층화 모형처럼 질문에 맞는 대안을 검토한다.

## 직접 바꿔 보는 연습

1. `x`의 첫 번째 값을 1에서 0으로 바꾸고 위험비와 신뢰구간이 어떻게 변하는지 확인한다.
2. `event`에서 검열 하나를 사건으로 바꿔 위험집합과 부분우도에 미치는 영향을 설명한다.
3. 추적 초반과 후반에 각각 위험비를 계산한다고 가정하고, 두 값이 크게 다를 때 상수 위험비가 무엇을 가릴지 적어 본다.

## 핵심 요약

1. 위험함수는 지금까지 사건 없이 남은 대상의 순간적인 사건 발생률이며 생존확률과 다르다.
2. Cox 모형은 $h_0(t)$에 $\exp(\beta x)$를 곱하고, $e^\beta$를 위험비로 해석한다.
3. 부분우도는 각 사건 시점의 위험집합을 비교하므로 기준 위험의 모양을 정하지 않고 계수를 추정할 수 있다.
4. 비례위험 가정은 위험 자체가 아니라 집단 간 위험비가 시간에 따라 일정하다고 가정한다.

## 다음 글

다음 글인 [생존분석 입문 ④: Schoenfeld 잔차로 비례위험 가정 진단]({{ '/posts/schoenfeld-residuals-ph-diagnostic/' | relative_url }})에서는 사건 대상과 위험집합의 공변량 차이를 계산하고, 시간에 따른 잔차 추세를 Python으로 확인한다.

## 참고 자료

- D. R. Cox (1972), [*Regression Models and Life-Tables*](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x), 설명변수와 미지의 기준 위험함수를 결합하고 부분우도로 계수를 추론하는 Cox 모형의 원 논문이다.
- [lifelines 생존 회귀 공식 문서](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html), Cox 모형식과 계수·위험비 해석, 비례위험 가정의 의미를 설명한다.
- [lifelines `CoxPHFitter` 공식 문서](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html), Cox 모형 적합과 계수·신뢰구간 출력을 확인할 수 있다.
