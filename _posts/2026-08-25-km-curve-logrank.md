---
title: "생존분석 입문 ②: Kaplan–Meier 곡선 비교와 log-rank 검정"
date: 2026-08-25 09:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [생존분석, Kaplan-Meier, log-rank, 중앙생존시간, 신뢰구간, Python]
description: "두 집단의 생존곡선에서 중앙생존시간과 불확실성을 읽고, 관측·기대 사건 수로 log-rank 검정 통계량과 p값을 계산하는 Python 실습을 진행합니다."
reading_time: 12
series: "처음 시작하는 생존분석"
level: "입문"
math: true
---

> **이번 글의 목표:** 두 Kaplan–Meier 곡선에서 중앙생존시간과 불확실성을 읽고, log-rank 검정이 두 집단의 사건 수를 어떻게 비교하는지 손과 Python으로 계산한다.

두 집단의 생존곡선을 그렸더니 한 곡선이 더 위에 있다고 하자. 눈으로 보이는 간격만으로 두 집단이 다르다고 결론 내려도 될까? 표본이 작거나 검열이 많으면 곡선은 크게 흔들릴 수 있다. 이번 글에서는 **곡선이 어디에 있는가**와 **차이가 우연한 변동보다 큰가**를 구분한다.

위험집합과 Kaplan–Meier 곱셈이 낯설다면 먼저 [생존분석 입문 ①의 검열과 Kaplan–Meier 손계산]({{ '/posts/survival-analysis-roadmap/' | relative_url }})을 복습하자. 이번 계산의 $n_j$와 $d_j$가 바로 그 글에서 사용한 값이다.

## 1. 곡선에서 먼저 읽을 세 가지

두 집단 A와 B의 Kaplan–Meier 추정량을 각각 $\widehat S_A(t)$, $\widehat S_B(t)$라고 하자. 그림에서는 다음 순서로 읽는다.

1. 사건 시점에 각 곡선이 얼마나 내려가는가
2. 시간에 따라 두 곡선의 간격과 순서가 유지되는가
3. 뒤쪽까지 남아 있는 위험집합이 충분한가

<figure class="study-figure">
  <img src="{{ '/assets/images/survival-analysis/km-two-groups-logrank.svg' | relative_url }}" alt="A와 B 두 집단의 Kaplan-Meier 생존곡선, 중앙생존시간 7과 9, 후반부의 작은 위험집합을 함께 보여주는 그림">
  <figcaption>그림 1. B 곡선이 대체로 위에 있지만, 표본이 8명씩뿐이므로 곡선 간격과 함께 불확실성을 확인해야 한다.</figcaption>
</figure>

곡선의 오른쪽 끝은 특히 조심해서 읽어야 한다. 시간이 지날수록 사건이나 검열로 위험집합이 작아져 한 번의 사건이 만드는 계단 폭이 커진다. 끝부분의 큰 차이가 반드시 강한 증거는 아니다.

### 중앙생존시간

**중앙생존시간**은 추정 생존확률이 처음으로 0.5 이하가 되는 시점이다.

$$
\widehat m=\inf\{t:\widehat S(t)\le 0.5\}
$$

그림에서는 A가 7, B가 9다. 따라서 B에서 사건까지의 시간이 표본상 더 길었다고 요약할 수 있다. 다만 곡선이 0.5까지 내려오지 않으면 중앙생존시간은 아직 추정할 수 없다. 마지막 관찰 시간을 중앙값이라고 대신 쓰면 안 된다.

## 2. 한 줄의 곡선에도 불확실성이 있다

Kaplan–Meier 곡선은 표본으로 계산한 추정값이다. 사건 시점 $t_j$의 위험집합을 $n_j$, 사건 수를 $d_j$라고 할 때 Greenwood 공식은 분산을 다음처럼 근사한다.

$$
\widehat{\mathrm{Var}}\{\widehat S(t)\}
=\widehat S(t)^2
\sum_{t_j\le t}\frac{d_j}{n_j(n_j-d_j)}
$$

$n_j$가 작거나 사건이 많이 발생하면 합이 커져 신뢰구간도 넓어진다. 실제 라이브러리는 구간이 0과 1 사이에 머물도록 흔히 log-log 변환을 적용한다. 신뢰구간은 “모집단 생존확률의 그럴듯한 범위”를 보여주지만, 같은 자료를 반복했을 때 개인의 사건 시간이 들어갈 범위는 아니다.

작은 예제의 시간 7에서 A의 생존확률은 0.400이고 95% 신뢰구간은 약 0.066~0.734다. B는 0.686이고 약 0.213~0.912다. 점추정치는 다르지만 범위가 넓다는 사실이 먼저 보인다. 신뢰구간의 겹침만으로 검정 결론을 정하지 말고, 전체 사건 시점을 사용하는 검정을 따로 계산해야 한다.

## 3. log-rank 검정의 직관

log-rank 검정의 귀무가설은 두 집단의 사건 발생 양상이 시간 전체에서 같다는 것이다. 핵심 질문은 간단하다.

> 두 집단이 같다면 각 사건 시점에 A에서 몇 건의 사건이 발생할 것으로 기대되는가?

시점 $t_j$ 직전 A와 B의 위험집합을 $n_{Aj}$, $n_{Bj}$, 전체 사건 수를 $d_j$라고 하자. A의 기대 사건 수는 위험집합 비율만큼 배분한다.

$$
E_{Aj}=d_j\frac{n_{Aj}}{n_{Aj}+n_{Bj}}
$$

예제의 첫 사건 시점 2에는 두 집단이 각각 8명이고 전체 사건은 1건이다. 두 집단이 같다면 A의 기대 사건 수는 $1\times8/16=0.5$건이다. 실제로는 A에서 1건이 관측됐다. 이 **관측값과 기대값의 차이**를 모든 사건 시점에 걸쳐 더한다.

예제 전체에서는 A의 관측 사건 수 $O_A=5$, 기대 사건 수 $E_A=3.05$다. 사건 시점별 차이의 분산을 합하면 $V_A=1.593$이고 검정통계량은 다음과 같다.

$$
\chi^2=\frac{(O_A-E_A)^2}{V_A}
=\frac{(5-3.05)^2}{1.593}=2.388
$$

자유도 1인 카이제곱 분포에서 p값은 0.122다. 0.05보다 크므로 이 작은 자료만으로 두 집단의 생존분포가 다르다는 귀무가설을 기각하지 못한다. 이는 “두 집단이 같다”는 증명이 아니다. 중앙생존시간은 7과 9로 달랐지만, 표본 16명의 불확실성이 커서 차이를 분명히 구별하지 못했다는 뜻이다.

## 4. Python으로 곡선 요약과 검정 재현하기

아래 코드는 `numpy`와 Python 표준 라이브러리만 사용한다. Greenwood log-log 신뢰구간과 log-rank 통계량을 한 번에 계산한다.

```python
import math
from statistics import NormalDist

import numpy as np

a_time = np.array([2, 3, 4, 5, 6, 7, 8, 9])
a_event = np.array([1, 1, 0, 1, 0, 1, 0, 1])
b_time = np.array([4, 5, 6, 7, 8, 9, 10, 11])
b_event = np.array([0, 1, 0, 1, 0, 1, 0, 1])


def km_summary(time, event, target=7):
    survival = 1.0
    greenwood_sum = 0.0
    median = math.inf

    for t in np.unique(time[event == 1]):
        n = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        survival *= 1 - d / n
        greenwood_sum += d / (n * (n - d)) if n > d else 0
        if survival <= 0.5 and math.isinf(median):
            median = float(t)
        if t == target:
            survival_at_target = survival
            greenwood_at_target = greenwood_sum

    z = NormalDist().inv_cdf(0.975)
    loglog = math.log(-math.log(survival_at_target))
    se_loglog = (
        math.sqrt(greenwood_at_target)
        / abs(math.log(survival_at_target))
    )
    lower = math.exp(-math.exp(loglog + z * se_loglog))
    upper = math.exp(-math.exp(loglog - z * se_loglog))
    return survival_at_target, lower, upper, median


def logrank(time_a, event_a, time_b, event_b):
    observed_a = expected_a = variance_a = 0.0
    event_times = np.unique(np.r_[
        time_a[event_a == 1], time_b[event_b == 1]
    ])

    for t in event_times:
        n_a = np.sum(time_a >= t)
        n_b = np.sum(time_b >= t)
        d_a = np.sum((time_a == t) & (event_a == 1))
        d_b = np.sum((time_b == t) & (event_b == 1))
        n, d = n_a + n_b, d_a + d_b

        observed_a += d_a
        expected_a += d * n_a / n
        if n > 1:
            variance_a += n_a * n_b * d * (n - d) / (n**2 * (n - 1))

    statistic = (observed_a - expected_a) ** 2 / variance_a
    p_value = math.erfc(math.sqrt(statistic / 2))
    return observed_a, expected_a, variance_a, statistic, p_value


for name, time, event in [
    ("A", a_time, a_event), ("B", b_time, b_event)
]:
    s, lower, upper, median = km_summary(time, event)
    print(
        f"{name}: S(7)={s:.3f}, 95% CI=({lower:.3f}, {upper:.3f}), "
        f"median={median:.0f}"
    )

o, e, v, chi2, p = logrank(a_time, a_event, b_time, b_event)
print(f"log-rank: O_A={o:.0f}, E_A={e:.3f}, V_A={v:.3f}")
print(f"chi2={chi2:.3f}, p={p:.3f}")
```

실행 결과는 다음과 같다.

```text
A: S(7)=0.400, 95% CI=(0.066, 0.734), median=7
B: S(7)=0.686, 95% CI=(0.213, 0.912), median=9
log-rank: O_A=5, E_A=3.050, V_A=1.593
chi2=2.388, p=0.122
```

## 5. 결과를 어디까지 해석할 수 있을까?

log-rank 검정은 **차이가 있는지**를 묻지만 차이의 크기를 직접 주지 않는다. 따라서 p값만 쓰지 말고 두 곡선, 시간별 위험집합, 중앙생존시간과 신뢰구간을 함께 제시해야 한다. 곡선이 교차하면 이른 차이와 늦은 차이가 상쇄될 수도 있으므로, 전체 기간을 한 숫자로 요약한 결과를 특히 조심해서 읽는다.

또한 관찰 자료의 두 집단을 비교한 p값은 원인과 효과를 말해 주지 않는다. 집단 선택에 공통 원인이 섞일 수 있는 이유는 [DAG로 교란과 뒷문 경로를 설명한 인과추론 입문]({{ '/posts/causal-dag-confounding/' | relative_url }})에서 확인할 수 있다. log-rank 검정은 검열이 각 집단에서 사건 위험과 독립적이라는 가정도 필요로 한다.

## 직접 바꿔 보는 연습

1. A의 시간 4 검열을 사건으로 바꾸고 곡선, 중앙생존시간, p값이 어떻게 달라지는지 확인한다.
2. 두 집단의 마지막 관측 2개씩을 제거해 위험집합이 줄면 신뢰구간이 어떻게 변하는지 살펴본다.
3. B의 사건 시간을 앞당겨 두 곡선이 교차하게 만든 뒤, 그림과 하나의 p값이 각각 무엇을 놓치는지 적어 본다.

## 핵심 요약

1. 중앙생존시간은 $\widehat S(t)$가 처음 0.5 이하가 되는 시점이며, 0.5에 도달하지 않으면 추정할 수 없다.
2. Greenwood 공식은 위험집합이 작아질수록 Kaplan–Meier 추정의 불확실성이 커지는 이유를 보여준다.
3. log-rank 검정은 각 사건 시점의 관측 사건 수와 귀무가설 아래 기대 사건 수를 누적해 비교한다.
4. 큰 p값은 두 집단이 같다는 증명이 아니며, 작은 p값도 효과 크기나 인과효과를 뜻하지 않는다.

## 다음 글

다음 글인 [생존분석 입문 ③: Cox 비례위험모형과 위험비 해석]({{ '/posts/cox-hazard-ratio/' | relative_url }})에서는 순간적인 사건 발생 정도를 나타내는 위험함수와 두 집단의 위험비를 배운다. Cox 모형이 공변량을 포함해 위험비를 추정하는 원리와 비례위험 가정도 함께 확인한다.

## 참고 자료

- Nathan Mantel (1966), [*Evaluation of Survival Data and Two New Rank Order Statistics Arising in Its Consideration*](https://pubmed.ncbi.nlm.nih.gov/5910392/), 검열된 생존 자료의 집단 비교를 위한 고전적 log-rank 방법을 제시한 논문이다.
- [lifelines `KaplanMeierFitter` 공식 문서](https://lifelines.readthedocs.io/en/stable/fitters/univariate/KaplanMeierFitter.html), 생존함수, 중앙생존시간과 Greenwood log-log 신뢰구간 구현을 설명한다.
- [lifelines `logrank_test` 공식 문서](https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html#lifelines.statistics.logrank_test), 두 집단 log-rank 검정의 입력과 통계량·p값 출력을 설명한다.
