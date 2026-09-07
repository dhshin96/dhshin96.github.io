---
title: "생존분석 입문 ⑧: 랜드마크 동적 예측과 예측 구간 Python 실습"
date: 2026-09-07 09:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [생존분석, 동적-예측, 랜드마크-분석, 조건부-생존, Kaplan-Meier, Python]
description: "추적 중 살아 있는 대상의 정보를 기준으로 미래 생존확률을 다시 계산하는 원리를 배웁니다. 세 예측 시점의 조건부 Kaplan–Meier 값을 손계산과 Python으로 재현합니다."
reading_time: 12
series: "처음 시작하는 생존분석"
level: "입문"
math: true
---

> **이번 글의 목표:** 랜드마크 시점 $s$와 예측창 $w$를 구분하고, 사건 없이 $s$에 도달한 사람의 $w$만큼 이후 생존확률을 반복해서 갱신한다.

8명의 합성 자료에서 4시간 단위 생존확률을 세 번 계산했다. 시간 2에서 시간 6까지의 예측은 0.750, 시간 4에서 시간 8까지는 0.514, 시간 6에서 시간 10까지는 0.300이었다. 같은 사람을 처음부터 끝까지 한 번 예측한 결과가 아니다. 매 시점에 아직 사건 없이 남은 사람만 새 출발선에 세우고, **지금부터 같은 길이의 미래**를 다시 물은 결과다.

## 1. 동적 예측은 질문의 시계를 앞으로 옮긴다

기준시점의 나이와 검사값만으로 만든 예측은 추적 중 새로 생긴 정보를 반영하지 못한다. 반면 동적 예측은 대상이 시간 $s$까지 사건 없이 도달했다는 사실과 그때까지 관측된 정보를 사용해 미래 위험을 갱신한다. 오늘은 공변량 없이 위험집합만 갱신하는 가장 단순한 형태부터 계산한다.

직전 글에서 [한 랜드마크 시점에 표본과 집단을 다시 정의하는 이유]({{ '/posts/survival-landmark-analysis/' | relative_url }})를 배웠다. 그 글은 시간 4의 한 장면을 비교했다. 이번에는 $s=2,4,6$으로 장면을 옮기며 같은 계산을 반복한다. 각 장면에서 사건이 이미 발생한 사람은 제외하고, 아직 위험집합에 남은 사람만 미래 예측의 대상이 된다.

여기서 두 시간을 섞지 않아야 한다.

- **예측 시점 $s$:** 지금까지의 정보를 끊어 보는 랜드마크
- **예측창 $w$:** 지금부터 얼마나 먼 미래까지 볼지 정한 길이
- **예측 끝점 $s+w$:** 실제로 생존 여부를 묻는 달력상의 시점

$s=4$, $w=4$라면 “시간 4까지 사건 없이 남은 사람이 시간 8까지도 사건 없이 지낼 확률”을 묻는다. $s$만 6으로 옮기고 $w=4$를 유지하면 끝점은 10이 된다. 서로 다른 달력 구간을 비교한다는 점이 핵심이다.

## 2. 조건부 Kaplan–Meier로 첫 동적 예측 만들기

대상 $i$의 사건 또는 검열 시간을 $T_i$, 사건 표시를 $\Delta_i$라 하자. 랜드마크 $s$에서 $T_i>s$인 대상만 남긴다. 이후 사건 시점 $t_j$의 직전 위험집합 크기를 $n_j(s)$, 사건 수를 $d_j$라 하면 다음 확률을 계산할 수 있다.

$$
\widehat S(s+w\mid T>s)
=\prod_{s<t_j\le s+w}
\left(1-\frac{d_j}{n_j(s)}\right)
$$

이 식은 시간 $s$까지 생존했다는 조건 아래 $s+w$까지 생존할 확률이다. 위험은 $1-\widehat S$로 바꿀 수 있다. 과거 사건은 곱에 들어가지 않고, $s$ 이후의 검열 대상은 검열 시점 전까지만 위험집합에 기여한다. 따라서 매 랜드마크에서 자료를 자르는 규칙과 [Kaplan–Meier의 위험집합 곱셈]({{ '/posts/km-curve-logrank/' | relative_url }})이 정확히 이어진다.

<figure class="study-figure">
  <img src="{{ '/assets/images/survival-analysis/dynamic-landmark-prediction-windows.svg' | relative_url }}" alt="시간 2, 4, 6을 각각 랜드마크로 삼고 길이 4의 예측창을 시간 6, 8, 10까지 이동시키며 조건부 생존확률을 갱신하는 세 개의 시간선">
  <figcaption>그림 1. 예측창 길이 $w=4$는 같지만 랜드마크 $s$가 이동하면 위험집합과 끝점 $s+w$가 함께 달라진다.</figcaption>
</figure>

## 3. 시간 4에서 작은 손계산 해보기

8명의 관측시간은 A부터 H까지 각각 3, 5, 6, 7, 8, 9, 10, 10이다. A, B, D, E, G는 사건이고 C, F, H는 검열이다. 시간 4에는 시간 3에 사건을 겪은 A를 제외한 7명이 남는다.

시간 5에 B의 사건이 발생할 때 위험집합은 7명이다. 생존확률은 $1-1/7=6/7$이 된다. 시간 6의 C는 검열이므로 계단이 내려가지는 않지만 그 뒤 위험집합에서는 빠진다. 시간 7에는 5명 중 D에게 사건이 발생하고, 시간 8에는 4명 중 E에게 사건이 발생한다.

$$
\widehat S(8\mid T>4)
=\left(1-\frac17\right)
 \left(1-\frac15\right)
 \left(1-\frac14\right)
=\frac{18}{35}=0.514
$$

따라서 시간 4에 살아 있는 대상의 다음 4시간 사건 위험은 $1-0.514=0.486$이다. 시간 6의 검열을 사건처럼 세면 안 되고, 위험집합에서도 끝까지 남겨 두면 안 된다. 이 두 규칙이 손계산과 코드에서 같아야 한다.

## 4. Python으로 세 랜드마크를 갱신하기

아래 코드는 Python 표준 라이브러리만 사용한다. 사건과 검열이 같은 시점에 있으면 그 시점 직전에는 둘 다 위험집합에 있으므로 사건 확률을 먼저 계산한 뒤 다음 시점으로 넘어간다.

```python
records = [
    ("A", 3, 1), ("B", 5, 1), ("C", 6, 0), ("D", 7, 1),
    ("E", 8, 1), ("F", 9, 0), ("G", 10, 1), ("H", 10, 0),
]


def landmark_km(data, landmark, window):
    horizon = landmark + window
    eligible = [(i, t, e) for i, t, e in data if t > landmark]
    survival = 1.0
    steps = []

    event_times = sorted({
        t for _, t, e in eligible if e == 1 and t <= horizon
    })
    for t in event_times:
        at_risk = sum(obs_time >= t for _, obs_time, _ in eligible)
        events = sum(
            obs_time == t and event == 1
            for _, obs_time, event in eligible
        )
        survival *= 1 - events / at_risk
        steps.append((t, at_risk, events, survival))

    return len(eligible), horizon, survival, steps


for s in (2, 4, 6):
    n, end, survival, steps = landmark_km(records, s, window=4)
    print(
        f"s={s}, end={end}, n={n}, "
        f"survival={survival:.3f}, risk={1-survival:.3f}"
    )
    print("steps:", steps)
```

핵심 출력은 다음과 같다.

```text
s=2, end=6, n=8, survival=0.750, risk=0.250
s=4, end=8, n=7, survival=0.514, risk=0.486
s=6, end=10, n=5, survival=0.300, risk=0.700
```

값이 0.750에서 0.300으로 낮아졌다고 해서 한 개인의 상태가 반드시 악화됐다고 단정할 수는 없다. 예측 시점마다 포함되는 위험집합이 달라졌고, 각 창이 덮는 사건 시점도 다르다. 이 예제는 집단의 조건부 Kaplan–Meier 추정값이지 개인별 예후 점수가 아니다.

## 5. 반복 측정값을 넣으면 개인화가 시작된다

실제 동적 예측은 시간 $s$까지 얻은 검사값이나 상태 $Z(s)$를 함께 사용한다. 각 랜드마크에서 살아 있는 대상만 남기고, 그 시점까지 관측된 최신 값으로 Cox 모형 등을 다시 적합하면 $\widehat S(s+w\mid T>s,Z(s))$를 계산할 수 있다. Cox 위험함수와 생존확률의 관계가 낯설다면 [Cox 모형의 위험비와 비례위험 가정]({{ '/posts/cox-hazard-ratio/' | relative_url }})을 먼저 복습하면 좋다.

여러 $s$마다 별도 모형을 만들면 표본이 줄고 추정값이 흔들릴 수 있다. 그래서 실제 연구에서는 여러 랜드마크 자료를 쌓고 $s$의 효과를 매끄럽게 연결하는 landmark supermodel도 사용한다. 어느 방식을 쓰든 예측 대상은 $s$에 사건 없이 남은 사람이며, 입력값은 $s$까지 알 수 있었던 정보로 제한해야 한다.

동적이라는 말이 미래 정보를 허용한다는 뜻은 아니다. 시간 6 예측에 시간 7 검사값을 넣거나, 전체 자료로 전처리한 값을 시간 2에 사용하면 누출이 생긴다. 훈련·검증 자료도 랜드마크 규칙에 맞춰 자르고, 예측 성능은 각 $(s,w)$ 조합에서 따로 평가해야 한다.

<div class="related-reading">
  <strong>이어 읽기</strong>
  <p>랜드마크 분석 글로 조건부 대상 정의를 확인하고, Kaplan–Meier 글과 Cox 위험비 글로 비모수 예측과 공변량 모형의 차이를 연결해 보세요.</p>
</div>

## 직접 바꿔 보는 연습

1. `window=3`으로 바꾸고 각 랜드마크의 끝점과 생존확률을 다시 계산한다.
2. C의 시간 6 검열을 사건으로 바꿔 $s=4$ 예측의 계단이 어떻게 달라지는지 확인한다.
3. 랜드마크를 5로 하나 추가하고 대상 수, 사건 시점, 4시간 위험을 손으로 먼저 적는다.

## 핵심 요약

1. 동적 예측은 사건 없이 도달한 현재 시점 $s$에서 위험집합과 관측 정보를 다시 정의한다.
2. 예측 시점 $s$, 예측창 $w$, 끝점 $s+w$는 서로 다른 역할을 하므로 결과와 함께 명시해야 한다.
3. 조건부 Kaplan–Meier는 공변량 없는 동적 예측의 출발점이며, 반복 측정값을 넣은 모형은 개인화를 더한다.
4. 시점별 예측값 차이는 위험집합과 미래 구간이 함께 바뀐 결과이므로 개인 상태 변화로 곧바로 해석하지 않는다.

## 다음 글

다음 글에서는 **동적 예측의 시간별 Brier 점수와 보정**을 살펴보고, 예측 시점마다 “잘 맞는다”는 말을 어떻게 확인하는지 배운다.

## 참고 자료

- van Houwelingen (2007), [*Dynamic Prediction by Landmarking in Event History Analysis*](https://doi.org/10.1111/j.1467-9469.2006.00529.x), 랜드마크 위험집합에서 모형을 갱신하고 여러 시점의 모형을 연결하는 접근을 제시한다.
- Rizopoulos et al. (2014), [*Dynamic Predictions with Time-Dependent Covariates in Survival Analysis using Joint Modeling and Landmarking*](https://arxiv.org/abs/1306.6479), 반복 측정 정보를 사용한 landmarking과 joint model의 예측 가정·성능 평가를 비교한다.
- Signorelli et al. (2024), [*Dynamic prediction of survival using multivariate functional principal component analysis: A strict landmarking approach*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928955/), 랜드마크 이전 사건을 제외하고 그 시점까지의 종단 정보만 쓰는 strict landmarking 절차를 설명한다.
