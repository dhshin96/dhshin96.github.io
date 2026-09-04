---
title: "인과추론 입문 ⑦: 성향점수 추정과 SMD 균형 진단 Python 실습"
date: 2026-09-04 09:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [인과추론, 성향점수, 공변량균형, 표준화평균차이, 역확률가중치, Python]
description: "처리 전 변수로 성향점수를 추정하고, 역확률 가중 전후의 표준화 평균 차이를 계산합니다. 작은 자료에서 공변량 불균형이 0.92에서 0.00으로 줄어드는 과정을 재현합니다."
reading_time: 12
series: "처음 시작하는 인과추론"
level: "입문"
math: true
---

> **이번 글의 목표:** 처리 전 공변량으로 성향점수를 추정하고, 역확률 가중 뒤 두 집단이 실제로 비슷해졌는지 표준화 평균 차이(SMD)로 진단한다.

먼저 결과부터 보자. 준비도가 높은 학생의 비율이 새 학습법 사용군에서는 66.7%, 미사용군에서는 25.0%라면 두 집단은 출발부터 다르다. 준비도의 표준화 평균 차이는 **0.92**다. 준비도별 학습법 사용 확률로 가중하면 두 집단의 높은 준비도 비율이 모두 50%가 되고 SMD는 **0.00**이 된다.

앞 글의 [양의 확률성과 겹침 부족 설명]({{ '/posts/causal-positivity-overlap/' | relative_url }})은 성향점수가 0이나 1에 가까울 때 가중치가 불안정해지는 이유를 다뤘다. 이번에는 그 성향점수를 자료에서 구한 뒤, “가중치 계산이 끝났다”가 아니라 “처리 전 변수가 균형을 이루었는가”까지 확인한다.

## 1. 성향점수는 결과가 아니라 처리 확률이다

처리 여부를 $A$, 처리 전에 측정한 공변량을 $L$, 결과를 $Y$라고 하자. 성향점수는 다음 조건부 확률이다.

$$
e(L)=P(A=1\mid L)
$$

핵심은 $Y$를 예측하는 점수가 아니라, 관측된 처리 전 특성에서 $A=1$이 될 확률이라는 점이다. 나이, 기초 점수처럼 처리와 결과에 모두 관련된 사전 변수는 후보가 된다. 반면 처리 뒤에 생긴 만족도나 중간 성취도를 넣으면 처리의 결과를 조건화할 수 있다.

어떤 변수를 넣어야 하는지는 자동으로 정해지지 않는다. 먼저 인과 구조를 생각해야 한다. [DAG로 최소 충분 조정 집합을 고르는 과정]({{ '/posts/causal-minimal-adjustment-set/' | relative_url }})이 필요한 이유다. 성향점수 모형은 그 집합을 숫자 하나로 요약하는 도구이지, 빠진 교란을 스스로 발견하는 장치가 아니다.

이번 예제에서 $L$은 준비도 낮음(0)과 높음(1) 두 값뿐이다. 각 층의 처리 비율을 그대로 사용하면 포화된 범주형 모형과 같은 추정값을 얻는다.

| 준비도 $L$ | 처리군 | 대조군 | 전체 | 추정 성향점수 $\hat e(L)$ |
| --- | ---: | ---: | ---: | ---: |
| 낮음(0) | 40 | 60 | 100 | $40/100=0.40$ |
| 높음(1) | 80 | 20 | 100 | $80/100=0.80$ |

변수가 여러 개이거나 연속형이면 로지스틱 회귀 등으로 각 사람의 확률을 추정할 수 있다. 중요한 것은 처리 예측 정확도만 높이는 일이 아니다. 추정한 점수를 매칭이나 가중에 사용한 뒤, 원래 공변량의 분포가 두 집단에서 비슷해졌는지를 별도로 확인해야 한다.

## 2. 작은 손계산: SMD 0.92를 구하기

이진 변수 $L$의 처리군 평균은 높은 준비도의 비율과 같아서 $80/120=0.667$이다. 대조군 평균은 $20/80=0.250$이다. 두 집단 평균 차이를 합동 표준편차로 나눈 SMD는 다음과 같다.

$$
\mathrm{SMD}
=\frac{\bar L_1-\bar L_0}
{\sqrt{\{s_1^2+s_0^2\}/2}}
$$

이진 변수의 분산을 $p(1-p)$로 계산하면

$$
\frac{0.667-0.250}
{\sqrt{\{0.667(1-0.667)+0.250(1-0.250)\}/2}}
=0.92
$$

이다. 양수는 처리군에 높은 준비도 학생이 더 많다는 방향을 나타낸다. 균형 자체를 볼 때는 보통 절댓값 $|\mathrm{SMD}|$를 비교한다.

이제 처리군은 $1/\hat e(L)$, 대조군은 $1/[1-\hat e(L)]$로 가중한다. 처리군의 낮음 층은 $40\times2.5=100$, 높음 층은 $80\times1.25=100$이 된다. 대조군도 각각 $60\times1.67\approx100$, $20\times5=100$이다. 두 가상 집단에서 높은 준비도 비율이 모두 0.5가 되므로 가중 SMD는 0이다.

<figure class="study-figure">
  <img src="{{ '/assets/images/causal-inference/propensity-score-smd-balance.svg' | relative_url }}" alt="역확률 가중 전에는 처리군의 높은 준비도 비율이 66.7퍼센트이고 대조군은 25퍼센트라 SMD가 0.92이지만, 가중 후에는 두 집단 모두 50퍼센트가 되어 SMD가 0인 비교 그림">
  <figcaption>그림 1. 성향점수 가중 뒤 처리 전 준비도의 분포가 같아졌다. 결과변수가 아니라 공변량 분포의 변화를 확인한다.</figcaption>
</figure>

## 3. SMD는 왜 p값과 다른가

두 평균의 검정 p값은 표본 크기에 크게 좌우된다. 같은 평균 차이라도 표본이 작으면 유의하지 않고, 표본이 매우 크면 사소한 차이도 유의할 수 있다. SMD는 차이를 변수의 표준편차 단위로 표현하므로 가중 전후 불균형의 크기를 같은 척도로 비교하기 쉽다.

$|\mathrm{SMD}|<0.1$은 자주 쓰이는 경험적 기준이지만 합격을 보장하는 법칙은 아니다. 중요한 교란변수의 SMD가 0.09라고 해서 미측정 교란까지 사라지는 것은 아니다. 평균만 같고 분산이나 분포 꼬리가 다를 수도 있다. 연속형 변수라면 SMD와 함께 분산비와 분포 그림을 살피고, 범주형 변수는 각 수준의 비율을 확인하는 편이 안전하다.

또한 균형 진단에는 결과 $Y$를 사용하지 않는다. 결과를 보며 성향점수 모형을 반복해서 바꾸면 연구 설계와 효과 추정이 뒤섞인다. 먼저 처리 전 자료만으로 설계를 정하고 균형을 확인한 다음, 고정된 설계에서 결과를 비교한다.

## 4. Python 실습: 가중 전후 SMD 계산하기

아래 코드는 표준 라이브러리만 사용한다. 200행 자료를 만들고 준비도별 처리 비율을 성향점수로 추정한 뒤, 준비도와 기초 점수의 SMD를 가중 전후로 계산한다.

```python
from math import fsum, sqrt

cells = [
    # 준비도, 처리, 기초 점수, 인원
    (0, 1, 50, 40),
    (0, 0, 50, 60),
    (1, 1, 70, 80),
    (1, 0, 70, 20),
]

rows = [
    {"ready": ready, "a": treatment, "pre": pre_score}
    for ready, treatment, pre_score, count in cells
    for _ in range(count)
]

propensity = {
    level: sum(r["a"] for r in rows if r["ready"] == level)
    / sum(r["ready"] == level for r in rows)
    for level in {0, 1}
}

for row in rows:
    e = propensity[row["ready"]]
    row["w"] = 1 / e if row["a"] else 1 / (1 - e)


def mean_var(values, weights):
    total = fsum(weights)
    mean = fsum(w * x for x, w in zip(values, weights)) / total
    var = fsum(w * (x - mean) ** 2 for x, w in zip(values, weights)) / total
    return mean, var


def smd(variable, weighted=False):
    stats = []
    for group in (1, 0):
        chosen = [r for r in rows if r["a"] == group]
        weights = [r["w"] if weighted else 1 for r in chosen]
        stats.append(mean_var([r[variable] for r in chosen], weights))
    (mean_1, var_1), (mean_0, var_0) = stats
    return (mean_1 - mean_0) / sqrt((var_1 + var_0) / 2)


print(f"성향점수: {propensity}")
for variable in ("ready", "pre"):
    print(
        f"{variable}: 가중 전 SMD={smd(variable):.2f}, "
        f"가중 후 SMD={smd(variable, weighted=True):.2f}"
    )
```

실행 결과는 다음과 같다.

```text
성향점수: {0: 0.4, 1: 0.8}
ready: 가중 전 SMD=0.92, 가중 후 SMD=0.00
pre: 가중 전 SMD=0.92, 가중 후 SMD=0.00
```

### 결과 해석

준비도와 기초 점수 모두 처리 전에 측정한 변수다. 가중 전 SMD 0.92는 학습법 사용군의 출발점이 훨씬 높았다는 뜻이다. 가중 후 0.00은 이 작은 예제에서 두 변수의 가중 평균이 정확히 같아졌음을 뜻한다. 이것은 관측한 변수의 균형에 대한 결과이지, 두 집단이 모든 미측정 특성에서도 같다는 증명은 아니다.

[DAG 표준화 예제에서 평균 차이가 왜 왜곡되는지]({{ '/posts/causal-dag-confounding/' | relative_url }})를 먼저 복습하면, 여기서 맞춘 공변량 분포가 어떤 뒷문 경로를 막으려는 것인지 연결해 볼 수 있다.

## 5. 직접 바꿔 보는 연습

### 연습 1: 불균형 방향 바꾸기

높은 준비도 층의 처리군과 대조군 인원을 각각 20명과 80명으로 바꾼다. SMD 부호가 음수로 바뀌는지, 절댓값은 무엇을 말하는지 확인한다.

### 연습 2: 모형에서 변수 빼기

새로운 이진 공변량을 하나 추가하되 성향점수 추정에는 넣지 않는다. 기존 변수의 균형이 좋아져도 빠진 변수의 SMD가 남을 수 있는지 계산한다.

### 연습 3: 겹침과 균형 함께 보기

높은 준비도 층의 대조군을 1명으로 줄인다. 가중 후 SMD만 보면 0에 가까울 수 있지만 최대 가중치와 유효표본크기는 어떻게 달라지는지 앞 글의 코드로 확인한다.

## 6. 이번 글에서 꼭 기억할 다섯 문장

1. 성향점수는 처리 전 공변량이 주어졌을 때 처리받을 확률이다.
2. 성향점수 모형에는 인과 구조를 바탕으로 선택한 사전 변수를 넣는다.
3. 모형의 처리 예측 성능보다 적용 후 공변량 균형이 핵심 진단 대상이다.
4. SMD는 집단 차이를 표준편차 단위로 나타내며 표본 크기에 덜 끌려간다.
5. 좋은 SMD는 관측 공변량의 균형을 뜻할 뿐 미측정 교란의 부재를 증명하지 않는다.

## 다음 글

다음 글에서는 성향점수가 비슷한 사람끼리 짝짓는 **최근접 이웃 매칭**과 너무 먼 짝을 막는 **캘리퍼**를 배우고, 매칭 뒤 표본 수와 균형이 어떻게 달라지는지 확인한다.

## 참고 자료

- Paul R. Rosenbaum, Donald B. Rubin (1983), [*The Central Role of the Propensity Score in Observational Studies for Causal Effects*](https://doi.org/10.1093/biomet/70.1.41), 성향점수를 관측 공변량의 균형점수로 정의한 원 논문이다.
- Peter C. Austin (2009), [*Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3472075/), 표준화 차이를 이용한 균형 진단과 해석을 다룬다.
- Peter C. Austin (2011), [*An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3144483/), 성향점수 추정·적용·균형 진단의 전체 흐름을 설명한다.
