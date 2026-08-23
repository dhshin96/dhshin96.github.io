---
title: "생존분석 로드맵: 검열 데이터에서 Cox 모델까지, 12단계 커리큘럼"
date: 2026-08-23 09:30:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [생존분석, survival-analysis, Kaplan-Meier, Cox-model, 검열, 커리큘럼]
description: "생존분석 입문자를 위한 12단계 로드맵. 검열, Kaplan–Meier, Cox 모형, 경쟁위험과 Survival ML을 비즈니스 실습으로 배웁니다."
reading_time: 19
series: "생존분석: 사건이 일어날 때까지"
level: "입문 → 심화"
math: true
---

> **이 글의 한 문장 요약:** 생존분석은 “사건이 발생했는가?”뿐 아니라 “언제 발생했으며, 아직 관측되지 않은 대상을 어떻게 다룰 것인가?”에 답하는 통계적 언어다.

## 분류 모델 대신 생존분석이 필요한 순간

다음 문제의 공통점은 무엇일까?

- 통신 고객이 **언제 해지할 것인가?**
- 반도체 설비가 **언제 고장날 것인가?**
- 제조 공정의 부품이 **얼마나 오래 정상 상태를 유지할 것인가?**
- 환자가 치료 후 **얼마 동안 재발 없이 지낼 것인가?**

모두 사건 발생까지의 시간이 핵심이다. 90일 이탈 여부만 분류하면 91일째 이탈과 3년 뒤 이탈을 같은 0으로 취급할 수 있고, 관찰이 끝날 때까지 이탈하지 않은 고객의 정보도 제대로 사용하지 못한다. 생존분석은 시간과 **검열(censoring)**을 함께 모델링한다.

## 이 커리큘럼의 목표

12개 단계를 마치면 다음을 수행할 수 있어야 한다.

1. 이벤트, 시간 원점, 관찰 종료와 검열 규칙을 명확히 정의한다.
2. 생존함수, 위험함수, 누적위험함수의 관계를 설명한다.
3. Kaplan–Meier 곡선과 log-rank 검정을 올바르게 해석한다.
4. Cox 비례위험 모형을 적합하고 비례위험 가정을 진단한다.
5. 시간가변 공변량, 경쟁위험, 반복 사건을 구분한다.
6. Survival ML 모델을 검열을 고려한 지표로 평가한다.
7. 통신 이탈·설비 고장 문제를 운영 가능한 위험 정책으로 전환한다.

## 12단계 생존분석 커리큘럼

| 단계 | 핵심 이론 | 실습 | 현업 산출물 |
| --- | --- | --- | --- |
| 1. 사건과 시간 | time-to-event, 시간 원점, 검열·절단 | 고객 이탈 데이터 스키마 | 관찰 설계서 |
| 2. 세 함수 | $S(t)$, $h(t)$, $H(t)$의 관계 | 지수분포 생존 자료 생성 | 위험 해석 노트 |
| 3. Kaplan–Meier | product-limit 추정량, 중앙생존시간 | KM 곡선 직접 계산 | 코호트 생존 리포트 |
| 4. 집단 비교 | log-rank, 층화, 다중 비교 | 요금제별 이탈 곡선 비교 | 세그먼트 비교표 |
| 5. Cox 모형 | partial likelihood, hazard ratio | 공변량 포함 Cox 회귀 | 효과·불확실성 표 |
| 6. 가정 진단 | 비례위험, Schoenfeld 잔차 | 시간에 따라 변하는 효과 확인 | 모형 진단서 |
| 7. 복잡한 관찰 | 좌측 절단, 시간가변 공변량 | 요금제 변경 이력 long format | 데이터 변환 명세 |
| 8. 경쟁위험 | cause-specific hazard, CIF, Fine–Gray | 해지·요금제 변경 동시 분석 | 경쟁 사건 리포트 |
| 9. 반복·다상태 | recurrent events, multi-state | 정상→경고→고장 전이 | 상태 전이 대시보드 |
| 10. 모수 모형 | Weibull, log-normal, AFT | 수명 분포와 가속계수 | 수명 예측 리포트 |
| 11. Survival ML | RSF, boosting, neural survival | 비선형 위험 예측 | 모델 비교표 |
| 12. 평가와 운영 | C-index, IPCW, Brier, calibration | 시점별 성능·드리프트 | 배치 정책과 모니터링 계획 |

## 1강: 검열을 이해하면 절반은 끝난다

### 1. 최소 데이터 구조

생존 자료의 최소 단위는 보통 다음 두 열이다.

| 변수 | 의미 |
| --- | --- |
| `duration` | 시간 원점부터 사건 또는 마지막 관찰까지의 시간 |
| `event` | 사건이 관찰됐으면 1, 검열됐으면 0 |

통신 이탈 분석이라면 시간 원점은 가입일 또는 분석 코호트 진입일, 사건은 해지, 검열은 관찰 종료일까지 유지 중인 상태가 될 수 있다. 설비 분석이라면 설치 또는 정비 완료 시점, 사건은 고장, 검열은 정상 운전 중인 관찰 종료다.

### 2. 우측 검열과 절단은 다르다

- **우측 검열**: 사건이 마지막 관찰 시점보다 뒤에 있다는 것만 안다.
- **좌측 검열**: 사건이 어떤 시점 이전에 발생했다는 것만 안다.
- **구간 검열**: 사건이 두 관찰 시점 사이에 발생했다.
- **좌측 절단**: 특정 진입 시점까지 살아남은 대상만 표본에 포함된다.

검열은 결과값 0이 아니다. “아직 사건을 보지 못했다”는 부분 정보다. 검열 대상을 단순 비사건으로 바꾸면 관찰 기간이 긴 대상과 짧은 대상을 공정하게 비교할 수 없다.

### 3. 생존함수, 위험함수, 누적위험함수

사건 시간 $T$에 대해 생존함수는 시점 $t$를 넘어 생존할 확률이다.

$$
S(t)=P(T>t)
$$

위험함수는 $t$까지 사건이 없었다는 조건에서 바로 다음 순간 사건이 발생할 순간 위험률이다.

$$
h(t)=\lim_{\Delta t\to 0}
\frac{P(t\le T<t+\Delta t\mid T\ge t)}{\Delta t}
$$

누적위험함수 $H(t)=\int_0^t h(u)du$와 생존함수는 다음 관계를 갖는다.

$$
S(t)=\exp\{-H(t)\}
$$

위험률 0.2를 “20%가 사건을 경험한다”로 해석하면 안 된다. 위험은 확률이 아니라 시간당 순간 발생률이다. Cox 모형의 hazard ratio도 특정 시점의 생존확률 비가 아니다.

## Kaplan–Meier 추정량

사건이 일어난 고유 시점을 $t_1<t_2<\cdots$라 하자. 시점 $t_j$ 직전에 위험집합에 남은 수를 $n_j$, 그 시점의 사건 수를 $d_j$라 하면 Kaplan–Meier 추정량은 다음과 같다.

$$
\widehat S(t)=\prod_{t_j\le t}\left(1-\frac{d_j}{n_j}\right)
$$

검열 대상은 검열 전까지 위험집합에 기여하고 이후 빠진다. Kaplan과 Meier의 1958년 [원 논문](https://doi.org/10.1080/01621459.1958.10501452)은 불완전한 관측에서 분포 형태를 가정하지 않고 생존함수를 추정하는 product-limit 방법을 제시했다.

## 첫 실습: 통신 고객 이탈 시간을 직접 추정하기

먼저 합성 고객 자료를 만든다. `event_time`은 실제 이탈 시간, `censor_time`은 관찰 가능한 마지막 시간이다. 우리가 보는 것은 둘 중 작은 값이다.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500

# 프리미엄 요금제 고객의 이탈률이 더 낮은 합성 상황
premium = rng.binomial(1, 0.45, n)
rate = 0.12 * np.exp(-0.5 * premium)
event_time = rng.exponential(1 / rate)
censor_time = rng.uniform(3, 18, n)

df = pd.DataFrame({
    "time": np.minimum(event_time, censor_time),
    "event": (event_time <= censor_time).astype(int),
    "premium": premium,
})

print(df.head())
print("관찰된 이탈 비율:", df["event"].mean().round(3))
```

이제 Kaplan–Meier 곱을 직접 계산한다.

```python
rows = []
survival = 1.0
event_times = np.sort(df.loc[df["event"] == 1, "time"].unique())

for t in event_times:
    at_risk = (df["time"] >= t).sum()
    events = ((df["time"] == t) & (df["event"] == 1)).sum()
    survival *= 1 - events / at_risk
    rows.append((t, at_risk, events, survival))

km = pd.DataFrame(
    rows,
    columns=["time", "at_risk", "events", "survival"],
)

median_survival = km.loc[km["survival"] <= 0.5, "time"].min()
print(km.head())
print(f"추정 중앙생존시간: {median_survival:.2f}")
```

시드가 같다면 관찰된 사건 비율은 약 `0.598`, 중앙생존시간은 약 `7.01`이다. 중앙생존시간은 생존확률이 처음 0.5 이하가 되는 시점이다. 곡선이 관찰 기간 안에 0.5 아래로 내려가지 않으면 중앙값은 추정되지 않는다.

실무에서는 공식 문서의 API를 사용한다.

```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter(label="전체 고객")
kmf.fit(df["time"], event_observed=df["event"])

print(kmf.median_survival_time_)
kmf.plot_survival_function()
```

`KaplanMeierFitter.fit(durations, event_observed)`와 `median_survival_time_`의 의미는 [lifelines 공식 문서](https://lifelines.readthedocs.io/en/stable/fitters/univariate/KaplanMeierFitter.html)에서 확인할 수 있다.

## 비즈니스 해석에서 자주 틀리는 지점

1. **검열을 비사건으로 처리한다**: 관찰이 일찍 끝난 대상을 장기 생존자로 잘못 분류한다.
2. **시간 원점이 섞여 있다**: 가입일 기준 고객과 캠페인 시작일 기준 고객을 같은 duration으로 비교한다.
3. **미래 정보를 공변량에 넣는다**: 사건 이후 또는 예측 시점 이후의 정보를 사용하면 누수가 발생한다.
4. **hazard ratio를 확률 비로 읽는다**: Cox 계수는 다른 공변량이 같을 때의 순간 위험 비다.
5. **비례위험 가정을 확인하지 않는다**: 효과가 시간에 따라 변하면 단일 hazard ratio가 현상을 숨긴다.
6. **경쟁 사건을 검열로 처리한다**: 요금제 변경이 해지를 막는 별도 사건이라면 단순 검열 가정이 맞지 않을 수 있다.

## 교수 관점과 팀장 관점의 합격 기준

**이론 합격 기준**

- $S(t)$, $h(t)$, $H(t)$의 차이와 관계를 설명한다.
- 검열과 절단을 구분하고 독립 검열 가정을 명시한다.
- Kaplan–Meier 위험집합을 손으로 계산한다.
- Cox 비례위험 가정을 잔차와 시간 상호작용으로 진단한다.

**실무 합격 기준**

- 시간 원점, 사건, 관찰 종료, 재진입 규칙을 데이터 추출 전에 합의한다.
- 전체 곡선뿐 아니라 관심 시점의 생존확률과 신뢰구간을 보고한다.
- 모델 평가 시 일반 분류 AUC만 쓰지 않고 검열을 고려한 지표를 쓴다.
- 위험 점수를 실제 접촉 용량, 정비 슬롯, 비용 함수와 연결한다.

Survival ML에서는 C-index 하나만으로 충분하지 않다. [scikit-survival 공식 평가 가이드](https://scikit-survival.readthedocs.io/en/v0.25.0/user_guide/evaluating-survival-models.html)는 시간의존 AUC와 Brier score를 함께 다루며, Brier score는 생존확률의 판별력과 보정을 시간별로 평가하는 데 유용하다.

## 심화 단계에서 만나게 될 모델

- **Cox 비례위험 모형**: $h(t\mid x)=h_0(t)\exp(x^T\beta)$로, 기저위험함수를 특정하지 않고 partial likelihood로 회귀계수를 추정한다. 출발점은 Cox의 [1972년 논문](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x)이다.
- **비례위험 진단**: `lifelines`의 [`check_assumptions`](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html)은 시간가변 계수에 대한 검정과 scaled Schoenfeld residual plot을 제공한다.
- **경쟁위험**: 사건별 cause-specific hazard와 누적발생함수(CIF)를 구분한다. Fine–Gray 모형은 [1999년 원 논문](https://doi.org/10.1080/01621459.1999.10474144)을 참고한다.
- **Random Survival Forest**: 비선형성과 상호작용을 학습하는 트리 앙상블이다. Ishwaran 등의 [2008년 논문](https://doi.org/10.1214/08-AOAS169)이 고전적 참고자료다.

## 연습문제

1. 합성 데이터에서 프리미엄·일반 요금제별 Kaplan–Meier 곡선을 따로 계산한다.
2. 검열 시간을 모두 6개월 이하로 줄였을 때 중앙생존시간이 어떻게 변하는지 확인한다.
3. 반도체 설비 데이터를 가정해 시간 원점, 사건, 우측 검열, 좌측 절단의 예를 각각 정의한다.
4. “90일 이탈 분류”와 “이탈 시간 생존모형”이 만드는 비즈니스 액션의 차이를 적는다.

## 다음 글

다음 글에서는 Kaplan–Meier 곡선의 신뢰구간과 log-rank 검정을 다룬다. 그 다음 Cox 모형의 hazard ratio를 해석하고, 비례위험 가정이 깨질 때 층화·시간 상호작용·AFT 모형 중 무엇을 선택할지 비교할 예정이다.

정책이나 처치가 사건 발생 시간에 미치는 효과까지 질문하고 싶다면 [인과추론 로드맵]({{ '/posts/causal-inference-roadmap/' | relative_url }})도 함께 읽어 보자.

## 참고 자료

- E. L. Kaplan, Paul Meier (1958), [*Nonparametric Estimation from Incomplete Observations*](https://doi.org/10.1080/01621459.1958.10501452).
- D. R. Cox (1972), [*Regression Models and Life-Tables*](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x).
- Jason P. Fine, Robert J. Gray (1999), [*A Proportional Hazards Model for the Subdistribution of a Competing Risk*](https://doi.org/10.1080/01621459.1999.10474144).
- Hemant Ishwaran et al. (2008), [*Random Survival Forests*](https://doi.org/10.1214/08-AOAS169).
- [lifelines KaplanMeierFitter 공식 문서](https://lifelines.readthedocs.io/en/stable/fitters/univariate/KaplanMeierFitter.html).
- [scikit-survival 모델 평가 공식 가이드](https://scikit-survival.readthedocs.io/en/v0.25.0/user_guide/evaluating-survival-models.html).
