---
title: "생존분석 입문 ⑦: 랜드마크 분석으로 불멸시간 편향 피하기"
date: 2026-09-05 09:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [생존분석, 랜드마크-분석, 불멸시간-편향, 조건부-생존, 시간가변-노출, Python]
description: "추적 중 정해진 시점까지 사건 없이 남은 대상만 다시 비교하는 원리를 배웁니다. 미래 노출을 앞당긴 분석과 랜드마크 분석의 위험을 손계산과 Python으로 재현합니다."
reading_time: 12
series: "처음 시작하는 생존분석"
level: "입문"
math: true
---

> **이번 글의 목표:** 랜드마크 시점에서 분석 대상과 노출 상태를 다시 정의하고, 이 방법이 불멸시간 편향을 피하는 대신 어떤 조건부 질문에 답하는지 설명한다.

10명의 합성 자료를 두 방식으로 계산했다. 추적 중 한 번이라도 치료받으면 처음부터 치료군으로 둔 분석에서는 시간 8까지 사건 위험이 치료군 0.500, 비치료군 0.750이었다. 치료가 위험을 낮춘 것처럼 보인다. 그러나 시간 4를 랜드마크로 정하고 그때까지 사건 없이 남은 사람만 비교하자 두 집단 위험은 모두 0.500이었다. 차이를 만든 것은 치료 효과가 아니라 **누가 어느 시점부터 비교에 들어갔는가**였다.

## 1. 시간 0의 집단을 미래 정보로 만들면 안 된다

치료가 추적 도중 시작된다면 시간 0에는 누가 치료받을지 아직 모른다. 그런데 전체 추적이 끝난 뒤 `언젠가 치료받음`을 기준으로 집단을 만들면, 나중에 치료받은 사람은 치료 전까지 사건 없이 살아 있어야 치료군이 될 수 있다. 그 무사건 시간을 처음부터 치료군에 붙이는 순간 불멸시간 편향이 생긴다.

직전 글에서 [시작–종료 자료가 치료 전후 시간을 제자리에 배정하는 이유]({{ '/posts/survival-start-stop-immortal-time-bias/' | relative_url }})를 확인했다. 시간가변 Cox 모형은 사건 시점마다 노출 상태를 갱신한다. 랜드마크 분석은 다른 길을 택한다. 미리 정한 시점 $s$에서 사진을 한 장 찍고, 그때까지 사건 없이 남은 사람과 그때까지 알려진 정보만으로 새 분석을 시작한다.

## 2. 랜드마크에서 표본과 집단을 다시 만든다

랜드마크가 $s=4$, 이후 관찰 창이 $w=4$라고 하자. 대상 $i$의 사건 또는 검열 시간을 $T_i$, 사건 표시를 $\Delta_i$, 치료 시작 시점을 $E_i$라 쓰면 분석 대상과 집단은 다음처럼 정한다.

$$
R_i(s)=I(T_i>s), \qquad A_i(s)=I(E_i\le s)
$$

$R_i(s)=1$인 사람만 남기고, 시간 4까지 치료를 시작했으면 $A_i(s)=1$로 둔다. 결과는 랜드마크 뒤부터 시간 8까지의 사건이다.

$$
Y_i(s,w)=I(s<T_i\le s+w,\ \Delta_i=1)
$$

따라서 집단 $a$의 단순 위험은 다음과 같다.

$$
\widehat{Risk}_a(s,w)=
\frac{\sum_i R_i(s)I\{A_i(s)=a\}Y_i(s,w)}
{\sum_i R_i(s)I\{A_i(s)=a\}}
$$

여기서 중요한 시계 규칙은 세 가지다. 랜드마크 전 사건이나 검열이 있는 대상은 제외한다. 집단은 시간 4까지 관측된 치료로만 정한다. 시간 4 이후 사건만 결과로 센다. 시간 5에 치료받은 D는 이번 분석에서 비치료군에 남는다. 미래 정보를 다시 들여다보지 않기 위해서다.

<figure class="study-figure">
  <img src="{{ '/assets/images/survival-analysis/landmark-analysis-cohort.svg' | relative_url }}" alt="시간 0의 10명 중 랜드마크 시간 4 이전에 사건이 발생한 2명을 제외하고, 남은 8명을 시간 4까지의 치료 여부로 4명씩 나누어 시간 8까지 추적하는 흐름도">
  <figcaption>그림 1. 랜드마크 분석은 시간 4에 도달한 사람만 남기고 그 시점의 정보로 집단을 고정한다. 이후 치료는 미래 정보이므로 분류를 바꾸지 않는다.</figcaption>
</figure>

## 3. 작은 손계산으로 비교 질문이 바뀌는지 본다

전체 10명을 `언젠가 치료받음`으로 나누면 치료군은 A, B, C, D, H, I의 6명이다. 시간 8까지 A, C, H에서 사건이 생겨 위험은 $3/6=0.500$이다. 비치료군 E, F, G, J에서는 E, G, J가 사건을 겪어 $3/4=0.750$이다. H는 시간 1에 치료받고 시간 2에 사건이 있었지만, G는 시간 3에 사건이 있어 치료 기회가 없었다. 이른 사건이 비치료군에 더 쉽게 남는다.

랜드마크 시간 4를 적용하면 시간 2와 3에 사건을 겪은 H와 G가 함께 제외된다. 남은 8명 가운데 시간 4까지 치료받은 A, B, C, I가 노출군이고, D, E, F, J가 비노출군이다. 시간 8까지 사건은 각 집단에서 2건이다.

$$
\widehat{Risk}_1(4,4)=\frac{2}{4}=0.500, \qquad
\widehat{Risk}_0(4,4)=\frac{2}{4}=0.500
$$

랜드마크 분석은 과거의 안전한 시간을 어느 집단에도 성과처럼 주지 않는다. 하지만 시간 4 전에 사건이 난 두 사람은 질문에서 빠진다. 답은 “처음 등록된 모든 사람에게 치료가 어떤 영향을 주는가?”가 아니라 **“시간 4까지 사건 없이 남은 사람 중, 그때 치료 상태가 다른 두 집단의 이후 위험은 어떤가?”**다.

## 4. Python으로 랜드마크 자료 만들기

아래 코드는 `pandas`만 사용한다. 같은 자료에서 잘못된 전체 추적 분류와 랜드마크 분류를 만들어 시간 8까지의 사건 위험을 비교한다.

```python
import pandas as pd

data = pd.DataFrame([
    ["A", 6, 1, 2],
    ["B", 10, 0, 3],
    ["C", 8, 1, 4],
    ["D", 10, 1, 5],
    ["E", 7, 1, None],
    ["F", 9, 0, None],
    ["G", 3, 1, None],
    ["H", 2, 1, 1],
    ["I", 9, 0, 2],
    ["J", 6, 1, None],
], columns=["id", "followup", "event", "treat_time"])

landmark = 4
horizon = 8
data["event_by_horizon"] = (
    (data["event"] == 1) & (data["followup"] <= horizon)
).astype(int)

# 잘못된 방식: 미래의 치료 여부를 시간 0의 집단처럼 사용한다.
data["ever_treated"] = data["treat_time"].notna().astype(int)
naive = data.groupby("ever_treated").agg(
    n=("id", "size"), events=("event_by_horizon", "sum")
)
naive["risk"] = naive["events"] / naive["n"]

# 랜드마크 방식: 시간 4에 사건 없이 남은 사람과 당시 정보만 사용한다.
lm = data.loc[data["followup"] > landmark].copy()
lm["treated_at_landmark"] = (
    lm["treat_time"].notna() & (lm["treat_time"] <= landmark)
).astype(int)
landmark_result = lm.groupby("treated_at_landmark").agg(
    n=("id", "size"), events=("event_by_horizon", "sum")
)
landmark_result["risk"] = (
    landmark_result["events"] / landmark_result["n"]
)

print("naive")
print(naive.round(3))
print("excluded before landmark:", sorted(set(data.id) - set(lm.id)))
print("landmark")
print(landmark_result.round(3))
```

핵심 출력은 다음과 같다.

```text
naive
              n  events  risk
ever_treated
0             4       3  0.75
1             6       3  0.50
excluded before landmark: ['G', 'H']
landmark
                     n  events  risk
treated_at_landmark
0                    4       2   0.5
1                    4       2   0.5
```

이 예제에는 랜드마크 뒤 조기 검열이 없어 단순 위험을 썼다. 실제 자료에 검열이 있으면 랜드마크에서 다시 시작한 Kaplan–Meier 곡선이나 Cox 모형으로 이후 생존을 분석한다. 곡선과 위험집합을 읽는 법은 [Kaplan–Meier 비교와 log-rank 검정]({{ '/posts/km-curve-logrank/' | relative_url }})에서 복습할 수 있다.

## 5. 편향 하나를 없애도 설계 판단은 남는다

랜드마크 $s$는 결과를 본 뒤 유리한 값으로 고르면 안 된다. 너무 이르면 아직 치료받은 사람이 적어 집단 정의가 불안정하고, 너무 늦으면 많은 사건을 제외해 표본과 질문이 크게 달라진다. 임상적으로 의미 있는 평가 시점이나 미리 정한 관찰 창을 사용하고, 합리적인 여러 시점에서 결과가 얼마나 달라지는지 민감도 분석으로 보여 주는 편이 낫다.

시간 4 이후에 치료받은 D를 끝까지 비노출군으로 두는 것도 기억해야 한다. 이는 미래 정보를 막는 단순한 규칙이지만, 이후 실제 노출 변화를 반영하지 못한다. 모든 사건 시점에서 노출을 갱신하려면 [시간가변 효과와 공변량을 구분한 구간별 Cox 접근]({{ '/posts/time-varying-effect-cox/' | relative_url }})과 시작–종료 자료가 더 직접적이다.

또한 랜드마크 분석은 시간에 따른 교란을 자동으로 없애지 않는다. 시간 4의 건강 상태가 이전 치료와 이후 사건 모두에 관련되면 집단 차이는 여전히 인과효과가 아닐 수 있다. 랜드마크 분석이 해결하는 핵심은 미래로 집단을 정의하는 오류이며, 무작위배정이나 교란 조정을 대신하지 않는다.

<div class="related-reading">
  <strong>이어 읽기</strong>
  <p>시작–종료 자료 글로 시간 전체를 갱신하는 방법을 비교하고, Kaplan–Meier 글과 시간가변 효과 글로 랜드마크 이후의 생존곡선과 모형 해석을 연결해 보세요.</p>
</div>

## 직접 바꿔 보는 연습

1. 랜드마크를 4에서 3으로 바꾸고 제외 대상, 집단 크기, 위험이 어떻게 달라지는지 확인한다.
2. 시간 5에 치료받는 D가 랜드마크 분석에서 왜 비노출군인지 한 문장으로 설명한다.
3. H의 사건 시간을 5로 바꾼 뒤 랜드마크 표본과 두 집단 위험을 다시 계산한다.

## 핵심 요약

1. 랜드마크 분석은 미리 정한 시점까지 사건 없이 남은 대상만 포함하고, 그때까지 알려진 정보로 집단을 고정한다.
2. 랜드마크 이전 사건을 제외하고 이후 사건만 세므로 치료 전 무사건 시간을 치료군의 성과로 붙이지 않는다.
3. 결과는 랜드마크까지 살아남은 사람의 조건부 비교이며, 전체 시작 코호트의 효과로 넓혀 해석할 수 없다.
4. 랜드마크 선택, 이후 노출 변화, 검열과 시간가변 교란은 별도로 점검해야 한다.

## 다음 글

다음 글에서는 **여러 랜드마크 시점에서 생존예측을 갱신하는 동적 예측**과 예측 시점·예측 구간을 구분하는 법을 살펴본다.

## 참고 자료

- Anderson, Cain, Gelber (1983), [*Analysis of Survival by Tumor Response*](https://pubmed.ncbi.nlm.nih.gov/6668489/), 미래에 결정되는 반응군을 처음부터 비교할 때 생기는 편향과 랜드마크 접근의 출발점을 설명한다.
- van Houwelingen (2007), [*Dynamic Prediction by Landmarking in Event History Analysis*](https://doi.org/10.1111/j.1467-9469.2006.00529.x), 랜드마크 시점의 위험집합으로 모형을 다시 적합해 예측을 갱신하는 방법을 제시한다.
- Agarwal et al. (2018), [*Immortal Time Bias in Observational Studies of Time-to-Event Outcomes*](https://doi.org/10.1177/1073274818789355), 랜드마크 분석의 대상 제외·집단 정의·추적 시작 규칙과 시간가변 Cox 모형을 비교한다.
