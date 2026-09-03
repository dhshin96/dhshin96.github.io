---
title: "생존분석 입문 ⑥: 시작–종료 자료로 불멸시간 편향 제대로 막기"
date: 2026-09-03 09:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [생존분석, 시간가변-공변량, 불멸시간-편향, 시작-종료-자료, Cox-모형, Python]
description: "추적 중 노출이 바뀌는 자료를 구간별 행으로 펼치는 법을 배웁니다. 미래의 노출을 기준시점에 붙였을 때 발생률비가 뒤집히는 현상을 손계산과 Python으로 재현합니다."
reading_time: 12
series: "처음 시작하는 생존분석"
level: "입문"
math: true
---

> **이번 글의 목표:** 시간가변 공변량을 $(start, stop]$ 구간으로 표현하고, 미래의 노출 정보를 앞당겨 쓰면 왜 불멸시간 편향이 생기는지 설명한다.

6명의 작은 자료를 두 방식으로 계산했다. 추적 중 한 번이라도 치료받은 사람을 처음부터 치료군으로 둔 잘못된 분석에서는 치료군의 사건 발생률이 비치료군의 0.519배였다. 같은 시간을 실제 치료 시점에서 나누자 치료 후 발생률은 치료 전의 3.100배였다. 이 숫자는 치료 효과를 추정하려는 결과가 아니라, **시간을 어느 집단에 넣느냐만으로 방향이 바뀔 수 있음**을 보여 주는 경고다.

## 1. 한 사람을 여러 행으로 나누는 이유

대상 A는 시간 0에 추적을 시작해 시간 6에 치료받고 시간 10에 사건을 겪었다고 하자. A의 치료 상태는 처음부터 1이 아니다. 시간 6까지는 0이고, 그 뒤에만 1이다. 따라서 한 행 `followup=10, treated=1`로 저장하면 시간 0~6에 아직 없던 치료 정보를 과거로 보낸다.

시작–종료 자료는 A의 추적 시간을 다음처럼 나눈다.

| id | start | stop | treated | event |
|---|---:|---:|---:|---:|
| A | 0 | 6 | 0 | 0 |
| A | 6 | 10 | 1 | 1 |

각 행은 $(start, stop]$ 구간을 나타낸다. 공변량은 구간 동안 적용되고, `event=1`은 그 구간의 오른쪽 끝에서 사건이 일어났다는 뜻이다. Cox 모형은 사건 시점마다 위험집합에 남아 있는 사람들의 **그때 값**을 비교한다. 한 사람이 여러 행으로 보이더라도 같은 시점의 위험집합에는 그 사람에게 해당하는 행 하나만 들어간다.

직전 글에서는 [시간가변 효과와 시간가변 공변량의 차이]({{ '/posts/time-varying-effect-cox/' | relative_url }})를 구분했다. 이번 글의 `treated(t)`는 계수의 변화가 아니라 값 자체가 0에서 1로 바뀌는 시간가변 공변량이다.

## 2. 불멸시간은 어떻게 만들어질까?

불멸시간은 노출군으로 분류되기 위해 사건 없이 살아 있어야 했던 기간이다. A가 시간 6에 치료받았다는 사실은 A가 적어도 시간 6까지 사건을 겪지 않았음을 이미 포함한다. 그런데 “언젠가 치료받음”을 기준으로 A의 시간 0~6까지 치료군에 넣으면, 사건이 일어날 수 없었던 시간을 치료군의 안전한 생존시간처럼 계산한다.

반대로 시간 3에 사건이 발생해 치료 기회가 없었던 C는 비치료군에 남는다. 이렇게 미래의 치료 여부로 기준시점의 집단을 만들면 이른 사건은 비치료군에 몰리고, 치료 전의 무사건 시간은 치료군에 쌓인다. 나이 같은 교란변수를 더 보정해도 잘못 배정한 시간을 원래 자리로 돌려놓지는 못한다.

<figure class="study-figure">
  <img src="{{ '/assets/images/survival-analysis/start-stop-immortal-time-bias.svg' | relative_url }}" alt="미래에 치료받은 대상을 처음부터 치료군으로 분류하는 잘못된 방식과 치료 시점에서 비노출·노출 구간을 나누는 시작–종료 방식을 비교한 시간선">
  <figcaption>그림 1. 잘못된 고정 분류는 치료 전 시간을 치료군에 넣는다. 시작–종료 자료에서는 치료 시점 전후가 서로 다른 행과 상태로 기록된다.</figcaption>
</figure>

## 3. 작은 손계산으로 시간의 주인을 확인한다

A, B, E는 각각 시간 6, 4, 7에 치료받았다. 전체 추적시간과 사건은 A가 10과 1, B가 8과 0, E가 9와 1이다. C, D, F는 치료받지 않았고 추적시간은 각각 3, 5, 6이며 C와 D에서 사건이 발생했다.

미래의 치료 여부로 고정 분류하면 치료군은 27인시간에 사건 2건, 비치료군은 14인시간에 사건 2건이다.

$$
IRR_{\text{잘못된 고정 분류}}
=\frac{2/27}{2/14}=0.519
$$

이제 치료 시점에서 시간을 자른다. 치료 전 시간은 $6+4+7+3+5+6=31$, 치료 후 시간은 $(10-6)+(8-4)+(9-7)=10$이다. 사건은 치료 전 2건, 치료 후 2건이다.

$$
IRR_{\text{시점에 맞춘 분류}}
=\frac{2/10}{2/31}=3.100
$$

두 값은 보정 위험비가 아니라 단순 사건 발생률비다. 작은 자료라 불확실성도 크다. 여기서 볼 것은 효과의 크기가 아니라, 같은 41인시간을 어느 상태에 배정했는지가 계산을 어떻게 바꿨는지다. 위험비의 정의와 위험집합 비교가 낯설다면 [Cox 모형에서 위험비를 해석하는 순서]({{ '/posts/cox-hazard-ratio/' | relative_url }})를 먼저 확인하면 이 차이가 더 선명해진다.

## 4. Python으로 시작–종료 자료 만들기

아래 코드는 `pandas`만 사용한다. 한 사람당 한 행인 원자료를 치료 시점에서 나누고, 잘못된 고정 분류와 시점에 맞춘 분류의 발생률을 함께 계산한다.

```python
import pandas as pd

base = pd.DataFrame([
    ["A", 10, 1, 6],
    ["B", 8, 0, 4],
    ["C", 3, 1, None],
    ["D", 5, 1, None],
    ["E", 9, 1, 7],
    ["F", 6, 0, None],
], columns=["id", "followup", "event", "treat_time"])

# 잘못된 분석: 미래에 치료받을 사람을 시간 0부터 치료군으로 둔다.
base["ever_treated"] = base["treat_time"].notna().astype(int)
naive = base.groupby("ever_treated").agg(
    events=("event", "sum"), person_time=("followup", "sum")
)
naive["rate"] = naive["events"] / naive["person_time"]

# 올바른 시간 배정: 치료 시점에서 한 사람의 행을 둘로 나눈다.
rows = []
for r in base.itertuples(index=False):
    if pd.notna(r.treat_time) and r.treat_time < r.followup:
        rows.append([r.id, 0, r.treat_time, 0, 0])
        rows.append([r.id, r.treat_time, r.followup, 1, r.event])
    else:
        rows.append([r.id, 0, r.followup, 0, r.event])

long = pd.DataFrame(
    rows, columns=["id", "start", "stop", "treated", "event"]
)
long["duration"] = long["stop"] - long["start"]
proper = long.groupby("treated").agg(
    events=("event", "sum"), person_time=("duration", "sum")
)
proper["rate"] = proper["events"] / proper["person_time"]

print(naive.round(3))
print(f"naive rate ratio={naive.loc[1, 'rate'] / naive.loc[0, 'rate']:.3f}")
print(long.to_string(index=False))
print(proper.round(3))
print(f"proper rate ratio={proper.loc[1, 'rate'] / proper.loc[0, 'rate']:.3f}")
```

핵심 출력은 다음과 같다.

```text
              events  person_time   rate
ever_treated
0                  2           14  0.143
1                  2           27  0.074
naive rate ratio=0.519

         events  person_time   rate
treated
0             2         31.0  0.065
1             2         10.0  0.200
proper rate ratio=3.100
```

`long`에는 6명이 9개 구간으로 펼쳐진다. 각 사람의 구간은 겹치거나 비어 있지 않아야 하고, 구간 길이의 합은 원래 추적시간과 같아야 한다. 사건은 사람마다 마지막 구간 끝에만 남는다. 이런 구조를 확인한 뒤 `start`, `stop`, `event`, `id`를 시간가변 Cox 구현에 넘길 수 있다.

## 5. 미래를 보지 않는 세 가지 점검

첫째, 시점 $t$의 공변량은 $t$ 직전까지 실제로 알 수 있던 값이어야 한다. 다음 방문에서 측정한 수치를 앞 구간에 선형 보간하면 미래 정보를 사용하게 된다. 둘째, 노출이 시작된 시점과 사건 시점이 같을 때 어느 구간에 속하는지 규칙을 먼저 정해야 한다. 이 글은 $(start, stop]$을 사용하고 상태 변화는 다음 구간 시작부터 적용했다. 셋째, 구간을 나눈 뒤 각 대상의 총 추적시간과 사건 수가 보존되는지 집계한다.

시작–종료 변환이 모든 문제를 해결하는 것은 아니다. 치료 시작에 영향을 준 건강 상태가 이후 사건에도 영향을 준다면 시간가변 교란이 남을 수 있다. 이번 단계의 목표는 그 문제까지 추정하는 것이 아니라, 적어도 관측되지 않은 미래를 현재의 설명변수로 쓰지 않는 것이다. 누적 사건확률 자체를 읽는 법은 [Kaplan–Meier 곡선과 위험집합 비교]({{ '/posts/km-curve-logrank/' | relative_url }})에서 복습할 수 있다.

<div class="related-reading">
  <strong>이어 읽기</strong>
  <p>먼저 시간가변 효과 글에서 값과 계수의 변화를 구분하고, Cox 위험비 글과 Kaplan–Meier 글로 사건 시점의 위험집합과 누적 생존확률을 함께 복습해 보세요.</p>
</div>

## 직접 바꿔 보는 연습

1. A의 `treat_time`을 6에서 2로 바꾸고 두 발생률비가 얼마나 가까워지는지 확인한다.
2. E의 사건을 0으로 바꾸고 치료 후 사건 수와 발생률비를 다시 계산한다.
3. `long.groupby("id")["duration"].sum()`을 원자료의 `followup`과 비교하는 검증 코드를 추가한다.

## 핵심 요약

1. 시간가변 공변량은 한 사람의 추적을 상태가 일정한 $(start, stop]$ 구간으로 나눠 기록한다.
2. 미래의 노출 여부를 시간 0에 붙이면 치료 전 무사건 시간이 노출군에 들어가 불멸시간 편향이 생긴다.
3. 각 시점에는 그 직전까지 관측된 값만 사용하고, 총 추적시간과 사건 수가 변환 전후 보존되는지 확인한다.
4. 시작–종료 구조는 올바른 시간 배정의 출발점이며 시간가변 교란 같은 별도 문제를 자동으로 없애지는 않는다.

## 다음 글

다음 글에서는 **랜드마크 분석**이 불멸시간 편향을 피하는 원리와, 랜드마크 이전 사건을 제외하면서 생기는 해석 범위를 살펴본다.

## 참고 자료

- Terry Therneau, Cynthia Crowson, Elizabeth Atkinson (2026), [*Using Time Dependent Covariates and Time Dependent Coefficients in the Cox Model*](https://publications.artsci.wustl.edu/web/packages/survival/vignettes/timedep.pdf), 미래를 보지 않는 원칙과 $(start, stop]$ 자료 구조를 설명한다.
- [lifelines 시간가변 생존회귀 공식 문서](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html), `id`, `start`, `stop`, 공변량, 사건으로 구성한 long format 예시를 제공한다.
- [*Immortal time bias in critical care research*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3365557/), 시간 고정 분석과 시간가변 Cox 분석을 비교해 잘못된 시간 분류의 영향을 보여 준다.
