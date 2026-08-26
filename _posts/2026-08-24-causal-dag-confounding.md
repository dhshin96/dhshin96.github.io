---
title: "인과추론 입문 ②: 평균 차이는 왜 틀릴까? DAG로 교란변수 조정하기"
date: 2026-08-24 09:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [인과추론, causal-DAG, 교란, 뒷문경로, 표준화, Python]
description: "관찰 데이터의 평균 차이는 왜 인과효과와 다를까요? DAG로 교란변수와 뒷문 경로를 찾고 표준화 조정을 손계산과 Python으로 배웁니다."
reading_time: 13
series: "처음 시작하는 인과추론"
level: "입문"
math: true
---

> **이번 글의 목표:** 인과 방향을 DAG로 표현하고, 처리와 결과 사이의 뒷문 경로를 찾아 사전 변수로 조정한 평균처리효과를 계산한다.

지난 글에서는 잠재적 결과 $Y(1)$과 $Y(0)$, 평균처리효과(ATE), 무작위배정을 배웠다. 아직 읽지 않았다면 [인과추론 입문 ①: 상관관계가 인과관계가 아닌 이유]({{ '/posts/causal-inference-roadmap/' | relative_url }})부터 읽는 것을 권한다.

무작위실험에서는 처리 여부를 동전 던지기처럼 정할 수 있다. 그러나 관찰 데이터에서는 사람이 스스로 처리를 선택하거나 어떤 조건에 따라 처리를 받는다. 이때 처리군과 대조군의 단순 평균 차이는 왜 인과효과와 달라질까? 이번 글에서는 **방향성 비순환 그래프(DAG)**와 **뒷문 조정(backdoor adjustment)**으로 답한다.

## 1. 관찰 데이터의 두 집단은 처음부터 다를 수 있다

새 학습법이 시험 점수에 미치는 효과를 알고 싶다고 하자. 학습 준비도가 높은 학생은 새 학습법을 더 적극적으로 선택하고, 준비도 자체도 시험 점수를 높일 수 있다.

- $A$: 새 학습법 사용 여부($1$은 사용, $0$은 미사용)
- $Y$: 시험 점수
- $L$: 학습법 사용 전에 측정한 준비도

처리군 점수가 더 높더라도 그 차이에는 두 가지가 섞여 있다.

1. 새 학습법 $A$가 점수 $Y$를 바꾼 효과
2. 원래 준비도 $L$가 달라서 생긴 차이

두 번째 차이가 첫 번째 효과의 추정을 흐리는 현상이 **교란(confounding)**이다. 표본을 아무리 크게 모아도 두 집단의 구성 차이가 그대로라면 이 편향은 저절로 사라지지 않는다.

## 2. DAG: 인과 가정을 화살표로 쓰는 언어

DAG는 **Directed Acyclic Graph**의 약자다.

- **방향성(directed)**: 원인에서 결과를 향해 화살표를 그린다.
- **비순환(acyclic)**: 화살표를 따라가 출발점으로 돌아오는 순환이 없다.
- **그래프(graph)**: 변수는 노드, 직접적인 인과관계는 화살표로 표현한다.

준비도, 학습법, 점수의 관계를 다음과 같이 가정해 보자.

<figure class="study-figure">
  <img src="{{ '/assets/images/causal-inference/confounding-dag.svg' | relative_url }}" alt="준비도 L이 새 학습법 A와 시험 점수 Y에 모두 영향을 주고 새 학습법 A도 점수 Y에 영향을 주는 인과 DAG">
  <figcaption>그림 1. 준비도 $L$은 처리 $A$와 결과 $Y$의 공통 원인이다. 화살표는 상관관계가 아니라 가정한 직접 인과 방향을 뜻한다.</figcaption>
</figure>

그림에는 $A$와 $Y$를 잇는 두 경로가 있다.

$$
A \rightarrow Y
$$

$$
A \leftarrow L \rightarrow Y
$$

첫 번째는 알고 싶은 인과 경로다. 두 번째는 $A$로 들어오는 화살표로 시작하는 **뒷문 경로(backdoor path)**다. 이 경로가 열려 있으면 $A$와 $Y$의 관찰된 연관성에 $L$의 영향이 함께 섞인다.

### DAG는 데이터가 자동으로 그려 주지 않는다

DAG는 산점도에서 발견한 화살표가 아니다. 시간 순서, 작동 원리, 선행 연구 같은 배경 지식을 바탕으로 분석 전에 세우는 인과 가정이다. 같은 데이터라도 가정한 DAG가 다르면 조정해야 할 변수도 달라질 수 있다. 따라서 화살표 하나하나를 말로 설명할 수 있어야 한다.

## 3. 작은 예제: 실제 효과는 5점인데 평균 차이는 17점

1,000명의 학생이 있고 준비도가 낮은 학생과 높은 학생이 각각 500명이라고 하자. 새 학습법의 효과는 준비도와 관계없이 모두 5점이다.

| 준비도 $L$ | 학습법 $A$ | 학생 수 | 평균 점수 $E[Y\mid A,L]$ |
| --- | ---: | ---: | ---: |
| 낮음(0) | 미사용(0) | 400 | 60 |
| 낮음(0) | 사용(1) | 100 | 65 |
| 높음(1) | 미사용(0) | 100 | 80 |
| 높음(1) | 사용(1) | 400 | 85 |

각 준비도 안에서 비교하면 새 학습법을 쓴 학생의 평균이 정확히 5점 높다.

$$
65-60=5, \qquad 85-80=5
$$

하지만 준비도를 무시하고 전체 처리군과 대조군을 비교해 보자.

$$
E[Y\mid A=1]
=\frac{100\times65+400\times85}{500}
=81
$$

$$
E[Y\mid A=0]
=\frac{400\times60+100\times80}{500}
=64
$$

따라서 조정하지 않은 평균 차이는 $81-64=17$점이다. 실제 효과 5점보다 12점이나 크다. 처리군에는 준비도가 높은 학생이 80%지만 대조군에는 20%뿐이기 때문이다.

<figure class="study-figure">
  <img src="{{ '/assets/images/causal-inference/confounding-mixture.svg' | relative_url }}" alt="처리군은 준비도가 높은 학생 80퍼센트, 대조군은 20퍼센트여서 두 집단의 평균 점수 차이에 구성 차이가 섞이는 그림">
  <figcaption>그림 2. 전체 평균은 학습법 효과뿐 아니라 서로 다른 준비도 구성까지 함께 비교한다.</figcaption>
</figure>

## 4. 뒷문을 막는다는 것은 같은 조건끼리 비교한다는 뜻이다

$L$로 **조건화(conditioning)**하면 낮은 준비도 안에서 처리군과 대조군을 비교하고, 높은 준비도 안에서도 따로 비교한다. 그래프에서는 $L$에 상자를 두어 뒷문 경로를 막았다고 표현할 수 있다.

<figure class="study-figure">
  <img src="{{ '/assets/images/causal-inference/backdoor-adjustment.svg' | relative_url }}" alt="준비도 L을 조건화하여 A에서 L을 거쳐 Y로 가는 뒷문 경로는 막고 A에서 Y로 가는 인과 경로는 남기는 DAG">
  <figcaption>그림 3. $L$을 조정하면 $A\leftarrow L\rightarrow Y$는 막히고, 알고 싶은 $A\rightarrow Y$ 경로는 남는다.</figcaption>
</figure>

조정 집합 $Z$가 다음 두 조건을 만족하면 $A$가 $Y$에 미치는 총효과를 위한 **뒷문 기준**을 만족한다고 한다.

1. $Z$에는 $A$의 결과로 생기는 후손 변수가 들어가지 않는다.
2. $Z$가 $A$와 $Y$ 사이의 모든 뒷문 경로를 막는다.

지금 DAG에서는 $Z=\{L\}$이면 충분하다. 반면 $A$ 이후에 생긴 학습 시간처럼 $A\rightarrow M\rightarrow Y$에 놓인 매개변수 $M$을 조정하면 총효과의 일부까지 제거한다. “측정한 변수를 모두 넣기”가 안전한 원칙은 아니다.

### 무엇을 가정하고 있는가?

$L$로 조정한 비교를 인과적으로 해석하려면 적어도 다음이 필요하다.

- $L$을 고정했을 때 처리군과 대조군 사이에 남은 교란이 없다: $Y(a)\perp\!\!\!\perp A\mid L$
- 각 $L$ 수준에 처리군과 대조군이 모두 존재한다: $0<P(A=1\mid L)<1$
- 관측 결과는 실제로 받은 처리의 잠재적 결과와 일치한다: $Y=Y(A)$

첫 번째는 **조건부 교환가능성**, 두 번째는 **양의 확률성(positivity)**, 세 번째는 **일관성(consistency)**이다. DAG만 그렸다고 이 가정들이 자동으로 참이 되는 것은 아니다.

<aside class="related-reading">
  <strong>잠재적 결과와 ATE가 낯설다면</strong>
  <p>표준화 식은 처리했을 때와 처리하지 않았을 때의 평균 결과를 비교한다. $Y(1)$, $Y(0)$과 ATE의 의미가 바로 떠오르지 않으면 <a href="{{ '/posts/causal-inference-roadmap/' | relative_url }}">인과추론 입문 ①의 잠재적 결과 설명과 가상 실험</a>을 짧게 복습한 뒤 아래 계산으로 돌아오자.</p>
</aside>

<aside class="related-reading">
  <strong>시간 자체가 결과인 연구라면</strong>
  <p>사건 발생까지의 시간이 결과이고 일부 대상의 관찰이 중간에 끝난다면 일반 평균 대신 검열을 고려해야 한다. <a href="{{ '/posts/survival-analysis-roadmap/' | relative_url }}">생존분석 입문 ①의 생존함수와 Kaplan–Meier 실습</a>에서 그 차이를 확인할 수 있다.</p>
</aside>

## 5. 층별 효과를 하나의 모집단 효과로 합치기

준비도별 평균 차이를 구한 뒤, 목표 모집단에서 각 준비도가 차지하는 비율로 다시 합치면 된다. 이 계산을 **표준화(standardization)**라고 한다.

$$
\widehat{ATE}_{std}
=\sum_l
\left\{
E[Y\mid A=1,L=l]-E[Y\mid A=0,L=l]
\right\}P(L=l)
$$

예제에서는 두 준비도 집단이 각각 절반이므로 다음과 같다.

$$
\widehat{ATE}_{std}
=(65-60)\times0.5+(85-80)\times0.5
=5
$$

이 식은 먼저 **같은 $L$ 안에서 공정하게 비교**하고, 그 결과를 **목표 모집단의 $L$ 분포로 평균**낸다. 조정 전 17점이 아니라 데이터 생성 과정의 실제 효과 5점을 회복한다.

## 6. Python 실습: 조정 전후를 직접 비교하기

아래 코드는 손계산 표와 정확히 같은 1,000행 데이터를 만든다. `numpy`만 사용한다.

```python
import numpy as np

# (준비도, 처리, 점수, 학생 수)
cells = [
    (0, 0, 60, 400),
    (0, 1, 65, 100),
    (1, 0, 80, 100),
    (1, 1, 85, 400),
]

readiness = np.concatenate([
    np.full(n, level) for level, treatment, score, n in cells
])
treatment = np.concatenate([
    np.full(n, treatment) for level, treatment, score, n in cells
])
score = np.concatenate([
    np.full(n, score) for level, treatment, score, n in cells
])

# 준비도를 무시한 단순 평균 차이
crude_difference = (
    score[treatment == 1].mean()
    - score[treatment == 0].mean()
)

# 준비도별 평균 차이와 모집단 비중을 이용한 표준화
standardized_ate = 0.0
for level in np.unique(readiness):
    in_level = readiness == level
    treated_mean = score[in_level & (treatment == 1)].mean()
    control_mean = score[in_level & (treatment == 0)].mean()
    weight = in_level.mean()
    effect = treated_mean - control_mean
    standardized_ate += effect * weight

    print(
        f"준비도 {level}: 효과={effect:.1f}, "
        f"모집단 비중={weight:.1f}"
    )

print(f"조정 전 평균 차이: {crude_difference:.1f}")
print(f"표준화한 ATE: {standardized_ate:.1f}")
```

실행 결과는 다음과 같다.

```text
준비도 0: 효과=5.0, 모집단 비중=0.5
준비도 1: 효과=5.0, 모집단 비중=0.5
조정 전 평균 차이: 17.0
표준화한 ATE: 5.0
```

### 결과 해석

1. 준비도가 같은 학생끼리 비교한 효과는 두 층 모두 5점이다.
2. 조정 전 17점에는 준비도 구성의 차이가 섞여 있다.
3. 각 층의 효과를 전체 모집단 비중으로 합치면 ATE는 5점이다.
4. 실제 자료에서는 각 셀의 평균을 표본으로 추정하므로 결과가 이렇게 정확히 떨어지지 않는다.

이 계산이 교란을 해결하려면 DAG에 그리지 못한 공통 원인이 남아 있지 않아야 한다. 측정되지 않은 교란이 있으면 표준화도 그것을 자동으로 제거하지 못한다.

## 7. 직접 바꿔 보는 연습

### 연습 1: 처리 선택의 불균형 줄이기

낮은 준비도의 처리군과 대조군 학생 수를 각각 250명으로, 높은 준비도도 각각 250명으로 바꿔 보자. 조정 전 평균 차이가 5점에 가까워지는 이유를 설명한다.

### 연습 2: 효과가 준비도에 따라 다르게 만들기

높은 준비도·처리군의 점수를 85에서 90으로 바꿔 보자. 두 층의 효과는 각각 5점과 10점이 된다. 표준화한 ATE가 두 값을 어떤 가중치로 합치는지 확인한다.

### 연습 3: 양의 확률성 깨뜨리기

낮은 준비도의 처리군 수를 0으로 바꾼 상황을 생각해 보자. `mean()`이 계산되지 않는 이유와, 낮은 준비도에서 “학습법을 썼다면”의 결과를 데이터만으로 비교할 수 없는 이유를 적어 본다.

## 8. 이번 글에서 꼭 기억할 다섯 문장

1. 교란은 처리와 결과의 공통 원인 때문에 관찰된 연관성과 인과효과가 달라지는 현상이다.
2. DAG의 화살표는 데이터가 아니라 배경 지식으로 세운 직접 인과 가정을 나타낸다.
3. $A$로 들어오는 화살표로 시작해 $Y$로 이어지는 길을 뒷문 경로라고 한다.
4. 적절한 사전 변수로 모든 뒷문 경로를 막으면 같은 조건의 처리군과 대조군을 비교할 수 있다.
5. 표준화는 조건별 효과를 목표 모집단의 조건 분포로 다시 평균내는 조정 방법이다.

## 다음 글

다음 글에서는 DAG의 경로가 언제 열리고 닫히는지 더 정확히 배운다. [인과추론 입문 ③: 콜라이더를 통제하면 왜 선택 편향이 생길까?]({{ '/posts/causal-collider-selection-bias/' | relative_url }})에서 공통 결과를 통제할 때 없던 연관성이 생기는 원리를 작은 표와 Python으로 확인한다.

## 참고 자료

- Miguel A. Hernán, James M. Robins, [*Causal Inference: What If*](https://www.hsph.harvard.edu/miguel-hernan/wp-content/uploads/sites/1268/2024/04/hernanrobins_WhatIf_26apr24.pdf), 6장과 7장에서 인과 DAG, 교란 구조, 뒷문 기준과 조정을 설명한다.
- Judea Pearl (1995), [*Causal Diagrams for Empirical Research*](https://doi.org/10.1093/biomet/82.4.669), DAG를 이용해 관찰 데이터에서 인과효과를 식별하는 그래프 기준을 제시한 원 논문이다.
- Johannes Textor 외 (2016), [*Robust Causal Inference Using Directed Acyclic Graphs: the R package dagitty*](https://doi.org/10.1093/ije/dyw341), DAG에서 최소 충분 조정 집합을 찾고 가정을 점검하는 도구를 설명한다.
