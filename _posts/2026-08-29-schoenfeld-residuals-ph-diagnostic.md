---
title: "생존분석 입문 ④: Schoenfeld 잔차로 비례위험 가정 진단하기"
date: 2026-08-29 09:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [생존분석, Schoenfeld-잔차, 비례위험-가정, Cox-모형, 잔차진단, Python]
description: "사건 대상의 공변량과 위험집합의 가중평균 차이로 잔차를 계산합니다. 시간에 따른 잔차 추세를 Python으로 재현하고 비례위험 위반 신호를 해석합니다."
reading_time: 12
series: "처음 시작하는 생존분석"
level: "입문"
math: true
---

> **이번 글의 목표:** Schoenfeld 잔차가 무엇을 비교하는지 손으로 계산하고, 잔차의 시간 추세로 Cox 모형의 비례위험 가정을 진단한다.

작은 자료에 Cox 모형을 맞추니 위험비가 1.552로 나왔다. 이 한 숫자는 전체 관찰 기간에 같은 효과가 이어졌다고 요약한다. 그런데 사건 순서에 따라 계산한 잔차는 초반 평균 0.450에서 후반 평균 -0.375로 내려갔다. 위험비 하나가 시간에 따른 변화를 가리고 있다는 신호다.

이 글에서는 p값 하나로 가정을 통과·탈락시키기보다, **사건이 일어난 대상과 그때 사건이 예상되던 대상을 비교한 차이**부터 읽는다. 위험함수와 위험비가 아직 낯설다면 [Cox 모형의 계수와 위험비를 설명한 생존분석 입문 ③]({{ '/posts/cox-hazard-ratio/' | relative_url }})을 먼저 복습하면 잔차식의 가중치가 선명해진다.

## 1. 잔차는 관측값에서 기대값을 뺀다

Cox 모형에서 사건 시점 $t_i$의 위험집합을 $R_i$라고 하자. 공변량이 하나일 때 사건을 경험한 대상의 값은 $x_i$다. 모형이 그 시점에 기대하는 공변량은 위험점수 $\exp(\hat\beta x_j)$로 가중한 평균이다.

$$
\bar{x}(t_i;\hat\beta)=
\frac{\sum_{j\in R_i}x_j\exp(\hat\beta x_j)}
{\sum_{j\in R_i}\exp(\hat\beta x_j)}
$$

**Schoenfeld 잔차**는 두 값의 차이다.

$$
r_i=x_i-\bar{x}(t_i;\hat\beta)
$$

사건이 관측된 시점마다 잔차가 하나 생긴다. 검열 시점에는 사건 대상 $x_i$가 없으므로 잔차도 만들지 않는다. 다만 검열된 대상은 검열되기 전 사건 시점의 위험집합에는 포함된다.

잔차가 양수면 사건 대상의 공변량이 모형의 가중평균보다 컸다는 뜻이다. 음수면 더 작았다는 뜻이다. 개별 잔차 하나를 이상치처럼 판정하는 것이 목적은 아니다. 사건 시간이 흐르면서 잔차의 중심이 계속 오르거나 내리는지를 본다.

## 2. 작은 손계산: 첫 사건의 잔차

이진 공변량 $x$가 1인 대상 10명과 0인 대상 10명이 모두 첫 사건 직전까지 남아 있다고 하자. 전체 자료로 추정한 계수가 $\hat\beta=0.440$이면 $x=1$의 위험점수는 $e^{0.440}=1.552$, $x=0$은 1이다.

첫 사건이 $x=1$인 대상에게 발생했다면 위험집합의 가중평균은 다음과 같다.

$$
\bar{x}(t_1)=
\frac{10\times1\times1.552+10\times0\times1}
{10\times1.552+10\times1}=0.608
$$

따라서 첫 Schoenfeld 잔차는 $1-0.608=0.392$다. 초반에 $x=1$의 사건이 예상보다 반복해서 많으면 양의 잔차가 이어진다. 후반에 $x=0$의 사건이 몰리면 음의 잔차가 이어진다. 모든 기간을 합친 위험비는 1.552여도 시간대별 상대 위험은 같지 않을 수 있다.

<figure class="study-figure">
  <img src="{{ '/assets/images/survival-analysis/schoenfeld-residual-trend.svg' | relative_url }}" alt="사건 순서 초반에는 양수이고 후반에는 음수로 내려가는 Schoenfeld 잔차 산점도와 하향 추세선">
  <figcaption>그림 1. 잔차가 0 주변에 무작위로 흩어지지 않고 사건 순서에 따라 내려가면 공변량 효과가 시간에 따라 달라질 가능성을 의심한다.</figcaption>
</figure>

## 3. 비례위험 가정과 잔차 추세

비례위험 가정 아래에서는 공변량의 계수 $\beta$가 시간에 따라 변하지 않는다. 적절히 척도화한 Schoenfeld 잔차 $s_i$는 시간가변 계수와 다음 관계를 갖는다.

$$
E\{s_i\}+\hat\beta\approx\beta(t_i)
$$

$\beta(t_i)=\hat\beta$라면 잔차의 평균은 시간과 무관하게 0 부근이어야 한다. 그래서 잔차를 사건 시간이나 사건 순위에 대해 그리고 평활선을 확인한다. 선이 대체로 평평하면 가정과 잘 맞고, 꾸준히 기울거나 휘면 시간가변 효과를 의심한다.

원래 잔차와 척도화 잔차는 숫자의 크기가 다를 수 있지만, 둘 다 사건 시점에 남은 체계적 패턴을 찾는다는 목적은 같다. 공변량이 여러 개라면 변수마다 잔차 열이 하나씩 생긴다. 어떤 변수의 잔차만 기울어진다면 모형 전체를 한꺼번에 실패로 부르기보다 그 변수의 함수 형태와 시간 의존성을 먼저 점검한다. 사건이 드문 구간의 평활선은 소수 점에 크게 흔들릴 수 있으므로, 그림 아래의 위험집합과 사건 수도 함께 확인해야 한다.

이 진단은 [Kaplan–Meier 곡선과 log-rank 검정을 다룬 생존분석 입문 ②]({{ '/posts/km-curve-logrank/' | relative_url }})의 곡선 비교와 역할이 다르다. 생존곡선의 교차는 비례위험 위반 가능성을 눈으로 알리는 신호다. Schoenfeld 잔차는 Cox 모형에 넣은 **각 공변량**의 효과가 시간에 따라 일정한지를 직접 살핀다.

## 4. Python으로 잔차와 시간 추세 재현하기

아래 코드는 `numpy`만 사용해 이진 공변량의 Cox 계수를 추정하고, 각 사건 순서의 Schoenfeld 잔차를 계산한다. 첫 8건의 사건은 주로 $x=1$, 중간 사건은 주로 $x=0$에서 발생하도록 만든 학습용 자료다.

```python
import numpy as np

time = np.arange(1, 21, dtype=float)
event = np.ones(20, dtype=int)
x = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
], dtype=float)

beta = 0.0
for _ in range(50):
    score = 0.0
    information = 0.0
    for t in time[event == 1]:
        risk = time >= t
        weight = np.exp(beta * x[risk])
        mean_x = np.sum(weight * x[risk]) / np.sum(weight)
        variance_x = np.sum(
            weight * (x[risk] - mean_x) ** 2
        ) / np.sum(weight)
        event_x = x[(time == t) & (event == 1)][0]
        score += event_x - mean_x
        information += variance_x

    step = score / information
    beta += step
    if abs(step) < 1e-12:
        break

residuals = []
event_times = time[event == 1]
for t in event_times:
    risk = time >= t
    weight = np.exp(beta * x[risk])
    expected_x = np.sum(weight * x[risk]) / np.sum(weight)
    event_x = x[(time == t) & (event == 1)][0]
    residuals.append(event_x - expected_x)

residuals = np.array(residuals)
correlation = np.corrcoef(event_times, residuals)[0, 1]
slope = np.polyfit(event_times, residuals, 1)[0]

print(f"beta={beta:.3f}, hazard ratio={np.exp(beta):.3f}")
print(f"early mean={residuals[:5].mean():.3f}")
print(f"late mean={residuals[-5:].mean():.3f}")
print(f"time correlation={correlation:.3f}, slope={slope:.3f}")
```

실행 결과는 다음과 같다.

```text
beta=0.440, hazard ratio=1.552
early mean=0.450
late mean=-0.375
time correlation=-0.756, slope=-0.060
```

상수 Cox 모형은 $x=1$의 위험을 평균적으로 1.552배라고 요약한다. 그러나 잔차는 초반 양수에서 후반 음수로 이동하고 시간 상관도 -0.756이다. 이 자료에서는 공변량 효과가 일정하다는 설명보다, 초반에는 더 크고 후반에는 약해지는 설명이 관측된 사건 순서와 더 잘 맞는다.

여기서 계산한 상관과 기울기는 원리를 확인하는 간단한 진단값이지 정식 검정의 p값이 아니다. 실제 분석에서는 척도화 잔차, 시간 변환, 평활선과 비례위험 검정을 함께 보고 표본이 크면 작은 이탈도 유의해질 수 있음을 고려한다.

## 5. 위반 신호를 본 뒤 할 일

잔차 추세가 보였다고 데이터를 지우거나 결과를 숨기면 안 된다. 먼저 공변량의 함수 형태가 잘못된 것은 아닌지, 중요한 상호작용이 빠지지 않았는지, 소수 관측이 추세를 만드는지 확인한다. 그다음 연구 질문에 맞춰 시간과 공변량의 상호작용을 넣어 $\beta(t)$를 모델링하거나, 계수를 추정할 필요가 없는 범주형 변수라면 층화 Cox 모형을 검토할 수 있다.

검정 p값이 0.05보다 크다는 사실도 가정이 참이라는 증명은 아니다. 사건 수가 적으면 실제 변화가 있어도 검정력이 부족하다. 반대로 자료가 매우 크면 해석상 작고 중요하지 않은 변화도 검출될 수 있다. 그림, 효과 크기, 연구 질문을 함께 판단해야 한다.

## 직접 바꿔 보는 연습

1. `x`의 9번째 값을 0에서 1로 바꾸고 위험비와 잔차 기울기가 함께 어떻게 변하는지 확인한다.
2. 마지막 두 사건의 `event`를 0으로 바꿔 검열 처리한 뒤 잔차 개수와 후반 평균을 다시 계산한다.
3. `x`의 1과 0이 사건 순서 전체에 번갈아 나타나도록 바꾸고 잔차 추세가 평평해지는지 비교한다.

## 핵심 요약

1. Schoenfeld 잔차는 사건 대상의 공변량에서 그 시점 위험집합의 위험점수 가중평균을 뺀 값이다.
2. 잔차는 사건 시점에만 정의되며, 검열 대상은 검열 전까지 위험집합에 기여한다.
3. 비례위험 가정이 맞으면 척도화 잔차의 중심은 시간에 따라 체계적으로 움직이지 않아야 한다.
4. 잔차 그림과 검정을 함께 보고, 위반 신호가 있으면 함수 형태·시간가변 효과·층화를 질문에 맞게 검토한다.

## 다음 글

다음 글인 [생존분석 입문 ⑤: 시간가변 효과와 구간별 위험비]({{ '/posts/time-varying-effect-cox/' | relative_url }})에서는 시간과 공변량의 상호작용으로 초반·후반 효과를 표현하고, 하나의 상수 위험비가 반대 방향의 효과를 어떻게 가리는지 Python으로 계산한다.

## 참고 자료

- David Schoenfeld (1982), [*Partial Residuals for the Proportional Hazards Regression Model*](https://doi.org/10.1093/biomet/69.1.239), 사건 시점별 공변량 잔차를 제안한 원 논문이다.
- [lifelines 비례위험 가정 진단 공식 안내](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html), 척도화 Schoenfeld 잔차의 시간 추세와 시각·통계 진단을 설명한다.
- [lifelines `proportional_hazard_test` 공식 문서](https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html#lifelines.statistics.proportional_hazard_test), 시간 변환과 정식 비례위험 검정의 입력을 확인할 수 있다.
