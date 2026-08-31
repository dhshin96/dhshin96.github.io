---
title: "생존분석 입문 ⑤: 시간가변 효과와 구간별 위험비 Python 실습"
date: 2026-08-31 09:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [생존분석, 시간가변-효과, Cox-모형, 위험비, 비례위험-가정, Python]
description: "하나의 상수 위험비가 시간대별 반대 효과를 가리는 과정을 확인합니다. 시간과 공변량의 상호작용을 손계산하고 구간별 위험비를 Python으로 재현합니다."
reading_time: 12
series: "처음 시작하는 생존분석"
level: "입문"
math: true
---

> **이번 글의 목표:** 시간가변 공변량과 시간가변 효과를 구분하고, 시간과 공변량의 상호작용으로 초반·후반 위험비를 계산한다.

같은 합성 자료에 두 Cox 모형을 맞췄다. 효과가 일정하다고 가정한 모형의 위험비는 0.932였다. 1에 가까워 두 집단 차이가 작아 보인다. 그러나 추적 기간을 둘로 나눈 모형에서는 초반 위험비가 2.068, 후반 위험비가 0.364였다. 하나의 숫자가 크기뿐 아니라 **효과 방향의 변화**까지 가린 셈이다.

직전 글에서 [Schoenfeld 잔차가 시간에 따라 기울어지는 이유]({{ '/posts/schoenfeld-residuals-ph-diagnostic/' | relative_url }})를 확인했다면, 이번에는 그 신호를 모형 안에 직접 표현한다. 핵심은 시간이 흐른다고 모든 변수를 바꾸는 것이 아니라, 연구 질문에 맞는 시간 함수와 공변량의 상호작용을 정하는 것이다.

## 1. 값이 변하는 것과 효과가 변하는 것은 다르다

**시간가변 공변량**은 한 대상의 측정값 자체가 시간에 따라 달라지는 경우다. 예를 들어 $x_i(t)$가 0에서 1로 바뀔 수 있다. 이때 계수 $eta$는 일정할 수 있다.

$$
h_i(t)=h_0(t)\exp\{\beta x_i(t)\}
$$

반면 **시간가변 효과**는 공변량 $x_i$가 고정되어 있어도 그 영향인 계수 $eta(t)$가 달라지는 경우다.

$$
h_i(t)=h_0(t)\exp\{\beta(t)x_i\}
$$

두 식은 서로 다른 질문에 답한다. 첫째는 “현재 측정값이 무엇인가?”, 둘째는 “같은 한 단위 차이가 지금 어느 정도의 위험 차이를 만드는가?”다. 이후 코드는 두 번째에 집중한다. [Cox 모형의 기준 위험과 위험비 해석]({{ '/posts/cox-hazard-ratio/' | relative_url }})이 낯설다면 먼저 $e^\beta$가 왜 위험비가 되는지 복습하면 좋다.

## 2. 구간 지시함수로 계수를 바꾼다

추적 시작부터 시간 10까지를 초반, 그 이후를 후반이라고 정하자. 후반이면 1인 지시함수를 $I(t>10)$으로 쓰면 모형은 다음과 같다.

$$
h_i(t)=h_0(t)\exp\{\beta x_i+\gamma x_i I(t>10)\}
$$

초반에는 지시함수가 0이므로 $x=1$과 $x=0$의 위험비가 $e^\beta$다. 후반에는 계수가 $\beta+\gamma$가 되어 위험비가 $e^{\beta+\gamma}$로 바뀐다.

작은 손계산을 해 보자. 합성 자료에서 추정된 값이 $\hat\beta=0.727$, $\hat\gamma=-1.737$이다.

$$
HR_{\text{초반}}=e^{0.727}=2.068
$$

$$
HR_{\text{후반}}=e^{0.727-1.737}=e^{-1.010}=0.364
$$

$\gamma$는 후반 위험비 자체가 아니다. **후반 로그 위험비가 초반보다 얼마나 변했는지** 나타낸다. 따라서 후반 효과는 반드시 두 계수를 더한 뒤 지수화해야 한다.

<figure class="study-figure">
  <img src="{{ '/assets/images/survival-analysis/time-varying-piecewise-hazard-ratio.svg' | relative_url }}" alt="상수 Cox 모형의 위험비 0.932와 시간가변 효과 모형의 초반 위험비 2.068, 후반 위험비 0.364를 비교한 계단형 그래프">
  <figcaption>그림 1. 상수 위험비는 전체 기간을 0.932로 요약하지만, 구간 상호작용은 시간 10을 경계로 효과 방향이 바뀌는 모습을 드러낸다.</figcaption>
</figure>

## 3. 같은 자료에서 왜 답이 달라질까?

Cox 부분우도는 각 사건 시점의 위험집합 안에서 사건 대상과 나머지 대상을 비교한다. 상수 모형은 모든 사건 시점에 같은 $\beta$를 사용한다. 초반에 $x=1$의 사건이 상대적으로 많고 후반에 $x=0$의 사건이 많으면 두 방향의 정보가 하나의 계수 안에서 상쇄될 수 있다.

구간 상호작용 모형은 사건 시점이 10을 넘었는지에 따라 설계변수 $xI(t>10)$을 0 또는 $x$로 바꾼다. 덕분에 초반 비교는 $\beta$, 후반 비교는 $\beta+\gamma$에 기여한다. 여기서 시간 10은 결과를 보고 유리하게 고른 값이 아니라 분석 전에 정한 학습용 경계다. 실제 분석에서 여러 절단점을 시험해 가장 극적인 결과만 고르면 불확실성을 과소평가한다.

구간별 위험비는 각 구간의 **순간적인 사건 발생률 비**다. 초반 위험비 2.068은 누적 사건확률이 2.068배라는 뜻이 아니며, 후반 위험비 0.364도 모든 대상의 위험이 갑자기 같은 폭으로 감소한다는 뜻이 아니다. 위험집합의 구성은 사건과 검열에 따라 계속 달라진다.

경계 시점의 정의도 명시해야 한다. 이 글에서는 $t=10$인 사건을 초반에 포함하고 $t>10$부터 후반으로 처리했다. 분석 도구가 $(start, stop]$ 구간을 사용한다면 측정값이 어느 구간에 들어가는지 확인해야 같은 계산을 재현할 수 있다. 결과를 보고할 때는 절단점, 각 구간의 사건 수, 두 위험비와 신뢰구간을 함께 제시한다. 그래야 후반 위험비가 소수 사건에 의해 크게 흔들린 결과인지 독자가 판단할 수 있다.

## 4. Python으로 상수 모형과 구간 모형 비교하기

아래 코드는 `numpy`만 사용한다. 20명의 사건 순서를 담은 합성 자료에 상수 계수 하나와 구간 상호작용 계수 두 개를 각각 Newton–Raphson 방법으로 추정한다.

```python
import numpy as np

time = np.arange(1, 21, dtype=float)
event = np.ones(20, dtype=int)
x = np.array([
    1, 1, 1, 1, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 0, 1, 0, 1, 1, 0, 1,
], dtype=float)
cutoff = 10.0


def fit_cox(piecewise):
    theta = np.zeros(2 if piecewise else 1)

    for _ in range(50):
        score = np.zeros_like(theta)
        information = np.zeros((len(theta), len(theta)))

        for t in time[event == 1]:
            risk = time >= t
            if piecewise:
                late = float(t > cutoff)
                z = np.column_stack([x[risk], x[risk] * late])
                event_z = np.array([
                    x[time == t][0], x[time == t][0] * late
                ])
            else:
                z = x[risk, None]
                event_z = np.array([x[time == t][0]])

            weight = np.exp(z @ theta)
            mean_z = np.sum(weight[:, None] * z, axis=0) / weight.sum()
            centered = z - mean_z
            information += (
                (weight[:, None] * centered).T @ centered / weight.sum()
            )
            score += event_z - mean_z

        step = np.linalg.solve(information, score)
        theta += step
        if np.max(np.abs(step)) < 1e-10:
            break

    return theta


constant_beta = fit_cox(piecewise=False)[0]
beta, gamma = fit_cox(piecewise=True)

print(f"constant HR={np.exp(constant_beta):.3f}")
print(f"beta={beta:.3f}, gamma={gamma:.3f}")
print(f"early HR={np.exp(beta):.3f}")
print(f"late HR={np.exp(beta + gamma):.3f}")
```

실행 결과는 다음과 같다.

```text
constant HR=0.932
beta=0.727, gamma=-1.737
early HR=2.068
late HR=0.364
```

상수 모형의 0.932만 보면 $x=1$과 $x=0$의 위험이 거의 비슷하다고 요약할 수 있다. 구간 모형은 초반에는 $x=1$의 위험이 더 높고 후반에는 더 낮다는 패턴을 보여준다. 그렇다고 구간 모형이 자동으로 더 옳은 것은 아니다. 계수가 하나 늘었으므로 신뢰구간, 모형 적합도, 잔차 패턴과 사전에 세운 질문을 함께 확인해야 한다.

## 5. 시간 함수를 정할 때 주의할 점

계단함수는 해석이 쉽지만 경계에서 효과가 갑자기 뛰는 모양을 가정한다. 변화가 부드럽다고 예상하면 $x\log(t)$ 같은 상호작용이나 스플라인을 사용할 수 있다. 시간 함수를 바꾸면 $\beta$의 기준 시점과 해석도 달라지므로 식과 기준을 함께 보고해야 한다.

또한 시간가변 효과를 넣었다고 비례위험 위반 문제가 모두 해결되는 것은 아니다. 잘못된 함수 형태, 빠진 상호작용, 소수 관측의 영향도 비슷한 잔차 추세를 만들 수 있다. [Kaplan–Meier 곡선 비교에서 위험집합과 불확실성을 함께 읽는 법]({{ '/posts/km-curve-logrank/' | relative_url }})처럼, 구간 뒤쪽에 남은 대상과 사건 수가 충분한지도 살펴야 한다.

마지막으로 $x$의 주효과를 빼고 $xI(t>10)$만 넣으면 초반 효과를 0으로 고정하는 다른 모형이 된다. “상호작용을 넣었다”는 말만으로는 식을 알 수 없다. 주효과와 상호작용을 함께 적고, 어느 시점 또는 구간이 기준인지 밝혀야 계수를 위험비로 올바르게 바꿀 수 있다.

## 직접 바꿔 보는 연습

1. `cutoff`를 8과 12로 바꾸고 초반·후반 위험비가 얼마나 민감한지 비교한다.
2. `x`의 첫 번째 값을 1에서 0으로 바꿔 초반 위험비와 상수 위험비의 변화를 확인한다.
3. `x`와 `x * np.log(t)`를 사용하는 부드러운 시간 상호작용으로 설계행렬을 바꾸고 시점별 위험비 식을 적어 본다.

## 핵심 요약

1. 시간가변 공변량은 $x(t)$가 변하고, 시간가변 효과는 같은 $x$의 계수 $\beta(t)$가 변한다.
2. 구간 상호작용 모형의 초반 위험비는 $e^\beta$, 후반 위험비는 $e^{\beta+\gamma}$다.
3. 하나의 상수 위험비는 시간대별로 방향이 다른 효과를 상쇄해 보여 줄 수 있다.
4. 시간 함수와 절단점은 연구 질문에 따라 정하고, 신뢰구간·적합도·잔차와 위험집합을 함께 확인한다.

## 다음 글

다음 글에서는 **시간가변 공변량의 시작–종료 자료 구조**를 만들고, 아직 관측되지 않은 미래 값을 사용하면 왜 불멸시간 편향이 생기는지 살펴본다.

## 참고 자료

- Terry Therneau, Cynthia Crowson, Elizabeth Atkinson (2026), [*Using Time Dependent Covariates and Time Dependent Coefficients in the Cox Model*](https://publications.artsci.wustl.edu/web/packages/survival/vignettes/timedep.pdf), 시간가변 공변량과 시간가변 계수를 서로 다른 Cox 모형 확장으로 설명한다.
- [lifelines 비례위험 가정 진단 공식 안내](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html), 시간가변 계수와 Schoenfeld 잔차의 관계 및 시간 상호작용 예제를 제공한다.
- [lifelines 시간가변 생존회귀 공식 문서](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html), 시작–종료 형식과 시점별 공변량을 포함한 Cox 모형을 설명한다.
