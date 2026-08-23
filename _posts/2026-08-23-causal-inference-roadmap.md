---
title: "인과추론 로드맵: 상관관계에서 ATE까지, 12단계 커리큘럼"
date: 2026-08-23 09:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [인과추론, causal-inference, ATE, DAG, propensity-score, 커리큘럼]
description: "인과추론을 처음 배우는 사람을 위한 12단계 로드맵. 잠재적 결과, DAG, 성향점수, 준실험, DML을 비즈니스 실습과 함께 정리합니다."
reading_time: 18
series: "인과추론: 질문에서 의사결정까지"
level: "입문 → 심화"
math: true
---

> **이 글의 한 문장 요약:** 인과추론의 출발점은 복잡한 모델이 아니라 “누구에게 무엇을 했을 때, 무엇과 비교해 어떤 결과를 측정할 것인가?”를 명확히 쓰는 일이다.

## 왜 지금 인과추론인가

머신러닝은 “누가 이탈할 것인가?”를 잘 예측할 수 있다. 그러나 실무의 다음 질문은 대개 다르다.

- 이 고객에게 할인 쿠폰을 보내면 **이탈이 실제로 줄어드는가?**
- 설비 점검 주기를 앞당기면 **고장이 감소하는가?**
- 새로운 품질 정책이 **수율을 개선했는가?**

예측 모델은 관측된 패턴을 학습하지만, 의사결정은 아직 관측하지 않은 개입의 결과를 요구한다. 인과추론은 이 간극을 다룬다. Rubin의 잠재적 결과 관점은 각 대상에게 처리했을 때와 처리하지 않았을 때의 결과를 구분했고, 현대 인과추론은 이 비교를 어떤 가정 아래 데이터로 식별할 수 있는지를 체계화한다. 입문부터 실무 적용까지 가장 일관된 참고서는 Hernán과 Robins의 공개 교과서 [*Causal Inference: What If*](https://www.hsph.harvard.edu/miguel-hernan/wp-content/uploads/sites/1268/2024/04/hernanrobins_WhatIf_26apr24.pdf)다.

## 이 커리큘럼의 목표

12개 단계를 마치면 다음을 할 수 있어야 한다.

1. 비즈니스 질문을 **처리, 결과, 대상, 비교 시점, estimand**로 바꾼다.
2. 잠재적 결과와 DAG로 인과 가정을 명시한다.
3. 실험·관측 연구·준실험 중 적절한 설계를 선택한다.
4. 회귀 보정, 성향점수, IPW, AIPW로 평균처리효과를 추정한다.
5. 중첩성, 균형, 위약 검정, 민감도 분석으로 결과를 반박해 본다.
6. CATE와 정책 가치를 구분하고 의사결정 문서로 전달한다.

## 12단계 인과추론 커리큘럼

| 단계 | 핵심 이론 | 실습 | 현업 산출물 |
| --- | --- | --- | --- |
| 1. 질문과 estimand | 예측과 인과, 처리·결과·대상, ATE·ATT | 통신 이탈 캠페인 질문 정의 | 1페이지 인과 질문서 |
| 2. 잠재적 결과 | $Y(1)$, $Y(0)$, 개별·평균 효과 | 합성 데이터로 반사실 비교 | estimand 명세 |
| 3. 인과 그래프 | DAG, 교란·매개·콜라이더 | 변수 관계를 DAG로 표현 | 가정 목록과 조정 변수 집합 |
| 4. 실험 설계 | 무작위배정, 순응도, 간섭 | A/B 테스트 설계와 검정력 | 실험 프로토콜 |
| 5. 식별 가정 | 일관성, 교환가능성, 양의 확률 | 중첩성 시각화 | 식별 가능성 리뷰 |
| 6. 기본 추정 | 표준화, 회귀 보정, g-formula | 결과 모델로 ATE 계산 | 추정 결과표 |
| 7. 성향점수 | 매칭, 층화, IPW, 균형 | SMD와 가중치 진단 | 균형 진단 리포트 |
| 8. 이중강건 추정 | AIPW, TMLE 개념 | 결과·처리 모델 교차검증 | 강건성 비교표 |
| 9. 준실험 | DiD, RDD, IV, synthetic control | 정책 전후 패널 분석 | 설계 타당성 메모 |
| 10. 이질적 효과 | CATE, uplift, 정책 학습 | 세그먼트별 효과 추정 | 타깃팅 정책안 |
| 11. Causal ML | cross-fitting, DML, causal forest | 고차원 공변량 처리 | 모델·추론 분리 보고서 |
| 12. 반박과 운영 | placebo, negative control, 민감도 | 가정 위반 시나리오 | 의사결정 문서와 모니터링 계획 |

이 순서는 “모델 선택”보다 “질문 → 가정 → 식별 → 추정 → 반박 → 결정”을 우선한다. DoWhy 역시 인과 분석을 Model, Identify, Estimate, Refute의 네 단계로 구성한다. 이는 라이브러리 사용법보다 분석 사고방식에 가깝다. [Microsoft Research의 DoWhy 소개](https://www.microsoft.com/en-us/research/project/dowhy/)에서 이 설계 철학을 확인할 수 있다.

---

## 1강: 상관관계가 의사결정의 답이 아닌 이유

### 1. 인과 질문을 다섯 칸으로 적기

예를 들어 통신사의 리텐션 캠페인을 평가한다고 하자.

| 요소 | 정의 예시 |
| --- | --- |
| 대상 | 최근 30일 사용량이 감소한 기존 고객 |
| 처리 $A$ | 해지 방어 쿠폰 발송 여부 |
| 결과 $Y$ | 90일 이내 해지 여부 |
| 비교 | 같은 고객군에서 쿠폰을 보내지 않았을 경우 |
| estimand | 대상 고객의 90일 해지율에 대한 ATE |

좋은 질문은 “쿠폰 효과가 있나?”보다 훨씬 구체적이다. 대상과 시간 창이 달라지면 같은 데이터에서도 다른 인과 효과를 묻게 된다.

### 2. 잠재적 결과와 근본적 문제

고객 $i$의 두 잠재적 결과를 다음처럼 쓴다.

$$
Y_i(1) = \text{쿠폰을 보냈을 때의 결과}, \qquad
Y_i(0) = \text{쿠폰을 보내지 않았을 때의 결과}
$$

개별 처리효과는 $Y_i(1)-Y_i(0)$이지만, 한 고객에게 두 상태를 동시에 관측할 수 없다. 그래서 모집단 평균을 묻는다.

$$
ATE = E\left[Y(1)-Y(0)\right]
$$

관측 자료에는 실제로 선택된 처리 $A_i$에 해당하는 결과 하나만 존재한다.

$$
Y_i = A_iY_i(1) + (1-A_i)Y_i(0)
$$

이것이 인과추론의 근본적 결측 문제다. Rubin의 1974년 논문 [*Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies*](https://www.ets.org/research/policy_research_reports/publications/article/1974/hrbx.html)는 이 틀의 고전적 출발점이다.

### 3. 관측 자료에서 필요한 세 가지 가정

1. **일관성(consistency)**: 실제 받은 처리가 $a$라면 관측 결과는 $Y(a)$다. “쿠폰 발송”의 버전이 제각각이면 가정부터 불명확하다.
2. **조건부 교환가능성(exchangeability)**: 공변량 $X$를 조건으로 처리 배정이 잠재적 결과와 독립이다. 즉 중요한 공통 원인을 충분히 관측했다고 가정한다.
3. **양의 확률(positivity)**: 비교하려는 모든 $X$ 영역에서 처리와 비처리 사례가 모두 존재해야 한다.

세 가정은 알고리즘이 자동으로 보장하지 않는다. 특히 측정하지 않은 교란은 더 복잡한 모델로 해결되지 않는다.

## 첫 실습: 단순 비교의 부호가 뒤집히는 데이터

고위험 고객일수록 처리를 더 자주 받고, 처리는 실제로 결과를 2만큼 낮추는 상황을 만든다. 여기서 `risk`는 처리와 결과의 공통 원인, 즉 교란 변수다.

```python
import numpy as np

rng = np.random.default_rng(42)
n = 4_000

# 고객의 사전 위험도
risk = rng.normal(size=n)

# 고위험 고객일수록 처리를 받을 가능성이 높다.
p_treat = 1 / (1 + np.exp(-(-0.2 + 1.2 * risk)))
treat = rng.binomial(1, p_treat)

# 실제 처리효과는 -2지만 risk가 결과를 크게 높인다.
outcome = 10 + 3 * risk - 2 * treat + rng.normal(scale=1, size=n)

# 잘못된 접근: 처리군과 비처리군의 단순 평균 차이
naive = outcome[treat == 1].mean() - outcome[treat == 0].mean()

# 첫 보정: outcome ~ intercept + treatment + risk
X = np.column_stack([np.ones(n), treat, risk])
beta = np.linalg.lstsq(X, outcome, rcond=None)[0]
adjusted = beta[1]

print(f"단순 평균 차이: {naive:.3f}")
print(f"risk 보정 효과: {adjusted:.3f}")
```

실행 결과는 대략 다음과 같다.

```text
단순 평균 차이: 0.979
risk 보정 효과: -1.981
```

처리는 실제로 결과를 낮췄지만, 고위험 고객이 처리군에 몰려 단순 비교에서는 오히려 결과를 높이는 것처럼 보인다. 회귀 보정이 참값에 가까워진 이유는 이 합성 데이터에서 `risk`가 모든 교란을 담고 있고 선형 결과 모형도 정확하기 때문이다. 현실에서는 이 조건을 먼저 의심해야 한다.

### 실무 리뷰 질문

- 처리 이전에 측정된 변수만 조정했는가?
- 처리의 원인이면서 결과의 원인인 변수를 놓치지 않았는가?
- 매개 변수나 콜라이더를 잘못 조정하지 않았는가?
- 처리군과 비교군의 공변량 분포가 실제로 겹치는가?
- ATE가 필요한가, 처리받은 집단의 ATT가 필요한가?
- 효과 크기뿐 아니라 비용과 부작용을 포함한 정책 가치도 계산했는가?

## 교수 관점과 팀장 관점의 합격 기준

**이론 합격 기준**

- estimand를 수식과 자연어로 모두 설명한다.
- DAG에서 교란·매개·콜라이더를 구분한다.
- 식별 가정과 추정 모형의 가정을 분리한다.
- 표준 오차와 신뢰구간을 포함해 불확실성을 보고한다.

**실무 합격 기준**

- 분석 전에 처리와 결과의 운영 정의를 고정한다.
- 데이터 생성 과정과 정책 배정 로직을 도메인 담당자에게 확인한다.
- 공변량 균형, 중첩성, 민감도 분석을 본 결과와 함께 제시한다.
- “효과가 있다”가 아니라 누구에게, 얼마만큼, 어떤 비용으로 적용할지를 제안한다.

## 심화 단계에서 만나게 될 도구

- **성향점수**: 관측 공변량을 기준으로 처리 배정의 차이를 요약한다. 고전적 근거는 Rosenbaum과 Rubin의 [1983년 논문](https://doi.org/10.1093/biomet/70.1.41)이다.
- **DoWhy**: 인과 그래프, 식별, 추정, 반박을 하나의 분석 흐름으로 관리한다.
- **EconML**: CATE, DML, causal forest처럼 머신러닝을 이용한 효과 이질성 추정을 제공한다. 공식 [CausalForestDML 문서](https://econml.azurewebsites.net/_autosummary/econml.dml.CausalForestDML.html)를 참고할 수 있다.
- **Double Machine Learning**: nuisance function을 유연하게 학습하면서 저차원 인과 모수에 대한 추론을 목표로 한다. 이론은 Chernozhukov 등의 [2018년 논문](https://doi.org/10.1111/ectj.12097)에 정리되어 있다.

## 연습문제

1. 위 실습에서 `risk`를 제거하고 단순 회귀를 적합해 결과가 어떻게 달라지는지 확인한다.
2. 처리 확률의 계수 `1.2`를 `3.0`으로 바꾸고 양의 확률 문제가 어떻게 나타나는지 시각화한다.
3. 반도체 예방 정비를 사례로 대상, 처리, 결과, 시간 창, estimand를 각각 한 문장으로 정의한다.
4. 통신 쿠폰 사례의 DAG를 그리고, 처리 후 사용량을 조정 변수에 넣어도 되는지 설명한다.

## 다음 글

다음 글에서는 잠재적 결과를 ATE·ATT·CATE로 구분하고, DAG를 이용해 “무엇을 조정해야 하는가”를 결정한다. 그 뒤 같은 데이터에서 회귀 보정, IPW, AIPW 결과를 비교할 예정이다.

시간이 지나야 관측되는 이탈·고장 문제도 함께 공부하고 싶다면 [생존분석 로드맵]({{ '/posts/survival-analysis-roadmap/' | relative_url }})으로 이어서 읽을 수 있다.

## 참고 자료

- Miguel A. Hernán, James M. Robins, [*Causal Inference: What If*](https://www.hsph.harvard.edu/miguel-hernan/wp-content/uploads/sites/1268/2024/04/hernanrobins_WhatIf_26apr24.pdf), 공개 교과서.
- Donald B. Rubin (1974), [*Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies*](https://doi.org/10.1037/h0037350).
- Paul R. Rosenbaum, Donald B. Rubin (1983), [*The Central Role of the Propensity Score in Observational Studies for Causal Effects*](https://doi.org/10.1093/biomet/70.1.41).
- Victor Chernozhukov et al. (2018), [*Double/debiased machine learning for treatment and structural parameters*](https://doi.org/10.1111/ectj.12097).
- PyWhy, [DoWhy 공식 프로젝트](https://www.microsoft.com/en-us/research/project/dowhy/) 및 [EconML 공식 문서](https://econml.azurewebsites.net/).
