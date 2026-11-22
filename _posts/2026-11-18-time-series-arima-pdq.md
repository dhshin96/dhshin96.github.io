---
title: "시계열 예측 입문 ⑦: ARIMA p·d·q로 값과 오차 연결하기"
date: 2026-11-18 20:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [시계열, 시계열-예측, ARIMA, 자기회귀, 이동평균, Python]
description: "ARIMA의 p·d·q가 각각 과거 변화량, 차분 횟수, 과거 예측 오차를 뜻하는 이유를 배웁니다. 순수 Python으로 ARIMA(1,1,1)의 한 단계 예측값 140.97을 재현합니다."
reading_time: 12
series: "처음 시작하는 시계열 예측"
level: "입문"
math: true
---

> **이번 글의 목표:** ARIMA의 세 숫자 $p$, $d$, $q$를 계산 순서와 연결하고, ARIMA(1,1,1)의 한 단계 예측을 손과 Python으로 재현한다.

먼저 결과부터 보자. 값이 136, 139, 144로 이어질 때 1차 차분의 최근 두 값은 3과 5다. 계수를 $phi=0.6$, $	heta=0.3$으로 둔 ARIMA(1,1,1)은 직전 변화량 3과 직전 예측 오차 0.57을 사용해 다음 변화량을 **1.97**로 예측한다. 이를 직전 수준 139에 다시 더하면 수준 예측은 **140.97**이다. 세 숫자는 모델 이름을 꾸미는 번호가 아니라, 이 계산에 얼마만큼의 과거를 넣을지 정하는 규칙이다.

## 1. 직관: 세 숫자는 계산의 역할표다

ARIMA는 AutoRegressive Integrated Moving Average의 머리글자다. 한국어로는 자기회귀 누적이동평균 모형이라고 부르지만, 처음에는 다음 역할로 기억하는 편이 낫다.

- $p$: 차분한 값의 과거를 몇 시점까지 사용하는가
- $d$: 원계열을 몇 번 차분하는가
- $q$: 과거의 예측 오차를 몇 시점까지 사용하는가

예를 들어 ARIMA(1,1,1)은 원계열을 한 번 차분하고, 직전 변화량 한 개와 직전 예측 오차 한 개를 사용한다. $p=0$이면 과거 변화량 항이 없고, $q=0$이면 과거 오차 항이 없다. $d=0$이면 차분하지 않으므로 ARIMA($p$,0,$q$)는 ARMA($p$,$q$)가 된다.

[정상성과 1차 차분 글]({{ '/posts/time-series-stationarity-differencing/' | relative_url }})에서 원계열의 수준을 변화량으로 바꾼 이유가 바로 $d$에 해당한다. 이번 예제는 같은 계열을 한 번 차분하므로 $d=1$이다.

## 2. 기호와 수식: 차분값을 AR과 MA로 설명한다

$d$번 차분한 계열을 $w_t=\Delta^d y_t$라고 하자. ARIMA($p$,$d$,$q$)의 기본 형태는 다음처럼 쓸 수 있다.

$$
w_t=c+\phi_1w_{t-1}+\cdots+\phi_pw_{t-p}
 +\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}
 +\varepsilon_t
$$

$\phi$가 붙은 부분은 과거의 관측된 변화량을 가져오는 AR 항이다. $\theta$가 붙은 부분은 과거 예측에서 남은 오차를 가져오는 MA 항이다. 여기서 이동평균은 최근 관측값을 단순 평균한다는 뜻이 아니다. 현재 충격 $\varepsilon_t$와 과거 충격의 가중합으로 현재를 설명하는 모형 이름이다.

$p=1$, $d=1$, $q=1$, $c=0$이면 한 단계 예측은 더 단순해진다.

$$
\hat w_t=\phi_1w_{t-1}+\theta_1\varepsilon_{t-1},\qquad
\hat y_t=y_{t-1}+\hat w_t
$$

첫 식은 변화량을 예측하고, 둘째 식은 차분을 되돌려 원래 수준으로 복원한다.

## 3. 작은 손계산: 139 다음 수준 예측하기

직전 변화량이 $w_{t-1}=3$, 직전 오차가 $\varepsilon_{t-1}=0.57$이라고 하자. $\phi_1=0.6$, $\theta_1=0.3$을 대입하면 다음과 같다.

$$
\hat w_t=0.6\times3+0.3\times0.57=1.971\approx1.97
$$

따라서 직전 수준 139에서 다음 수준은 $139+1.97=140.97$로 예측한다. 실제 값이 144라면 변화량 오차는 $5-1.97=3.03$이다. 이 오차는 다음 시점의 MA 항에 들어간다. 예측이 빗나간 흔적을 버리지 않고 다음 계산에 반영하는 셈이다.

<figure class="study-figure">
  <img src="{{ '/assets/images/time-series/arima-pdq.svg' | relative_url }}" alt="ARIMA(1,1,1)의 계산 흐름. 원계열 136, 139, 144를 한 번 차분해 변화량 3과 5를 만들고, p=1이 직전 변화량 3, q=1이 직전 오차 0.57을 선택한다. 두 값에 계수 0.6과 0.3을 곱해 변화량 예측 1.97을 얻고 직전 수준 139에 더해 수준 예측 140.97을 만든다.">
  <figcaption>그림 1. $d$는 계열을 차분하고, $p$와 $q$는 각각 과거 변화량과 과거 오차를 선택한다. 마지막에는 예측 변화량을 직전 수준에 더한다.</figcaption>
</figure>

## 4. 실행 가능한 Python 실습

다음 코드는 외부 패키지 없이 같은 계산을 처음부터 반복한다. 첫 예측에는 이전 오차가 없으므로 0으로 시작한다.

```python
values = [100, 103, 108, 112, 115, 120,
          124, 127, 132, 136, 139, 144]
phi, theta = 0.6, 0.3
differences = [
    current - previous
    for previous, current in zip(values, values[1:])
]

predictions = [None]
errors = [0.0]
level_forecasts = [None]

for index in range(1, len(differences)):
    prediction = (
        phi * differences[index - 1]
        + theta * errors[index - 1]
    )
    error = differences[index] - prediction
    predictions.append(prediction)
    errors.append(error)
    level_forecasts.append(values[index] + prediction)

print("1차 차분:", differences)
print(
    f"마지막 계산: 0.6×{differences[-2]} + "
    f"0.3×{errors[-2]:.2f} = {predictions[-1]:.2f}"
)
print(f"마지막 실제 변화량: {differences[-1]:.2f}")
print(f"마지막 변화량 오차: {errors[-1]:.2f}")
print(
    f"수준 예측: {values[-2]} + {predictions[-1]:.2f} "
    f"= {level_forecasts[-1]:.2f}"
)
```

실행 출력은 다음과 같다.

```text
1차 차분: [3, 5, 4, 3, 5, 4, 3, 5, 4, 3, 5]
마지막 계산: 0.6×3 + 0.3×0.57 = 1.97
마지막 실제 변화량: 5.00
마지막 변화량 오차: 3.03
수준 예측: 139 + 1.97 = 140.97
```

## 5. 결과 해석: 차수와 계수는 다르다

ARIMA(1,1,1)의 1, 1, 1은 계수값이 아니다. 사용할 항의 개수와 차분 횟수다. 실제 영향의 크기는 학습으로 추정하는 $\phi_1$과 $\theta_1$이 결정한다. 이번 글에서는 계산을 투명하게 보이기 위해 두 계수를 미리 정했다. 따라서 이 결과는 모델 적합의 성능 비교가 아니라, 정해진 ARIMA 식이 예측값을 만드는 과정의 재현이다.

또한 $p=1$은 직전 값 하나만 중요하다는 결론이 아니라, 이 모형이 직전 차분값까지 직접 포함한다는 뜻이다. 더 오래된 값의 영향은 직전 값에 이미 전달되어 간접적으로 남을 수 있다. $q=1$도 같은 방식으로 직전 오차 한 개를 직접 포함한다. 차수가 커질수록 무조건 좋아지는 것이 아니라 추정할 계수가 늘고 짧은 데이터에 과하게 맞을 위험도 커진다.

[자기상관과 지연값 글]({{ '/posts/time-series-autocorrelation-lags/' | relative_url }})의 ACF와 PACF는 $p$와 $q$ 후보를 좁히는 단서가 된다. 다만 둘이 모두 0보다 큰 혼합 모형에서는 그래프만으로 차수를 확정하기 어렵다. 먼저 필요한 $d$를 정하고, 여러 $p$, $q$ 후보를 적합해 AICc 같은 정보 기준과 잔차를 확인해야 한다. 차분 차수가 다른 모델의 AIC를 그대로 비교하는 것도 피해야 한다. 차분하면 우도가 계산되는 데이터 자체가 달라지기 때문이다.

후보를 골랐다고 검증이 끝난 것은 아니다. 잔차에 자기상관이 남으면 모델이 시간 구조를 충분히 설명하지 못한 것이다. 그리고 최종 예측력은 [롤링 검증 글]({{ '/posts/time-series-rolling-origin-validation/' | relative_url }})처럼 시간 순서를 지킨 여러 출발점에서 확인해야 한다.

좋은 잔차는 다음 시점에 활용할 규칙이 남지 않은 백색잡음에 가깝다. 따라서 잔차의 평균이 0 주변인지, ACF에 뚜렷한 봉우리가 남는지, 분산이 시간에 따라 달라지는지를 함께 살핀다. 잔차에 패턴이 보이면 차수를 무작정 늘리기보다 빠진 계절성, 이상치, 변환 필요성을 먼저 점검한다.

## 6. 직접 바꿔 보는 연습

1. `phi`를 0으로 바꿔 AR 항이 사라질 때 마지막 예측을 계산한다.
2. `theta`를 0으로 바꿔 MA 항이 사라질 때 오차가 다음 예측에 전달되지 않는지 확인한다.
3. `values`의 모든 값에 1,000을 더해도 변화량 예측은 같고 수준 예측만 1,000 커지는지 확인한다.

## 핵심 요약

- ARIMA의 $p$는 과거 차분값, $d$는 차분 횟수, $q$는 과거 예측 오차의 사용 범위를 뜻한다.
- ARIMA(1,1,1)은 한 번 차분한 뒤 직전 변화량과 직전 오차를 한 개씩 사용한다.
- 예제의 변화량 예측은 1.97, 이를 원래 수준으로 복원한 예측은 140.97이었다.
- 차수 $p,d,q$와 학습되는 계수 $\phi,\theta$는 서로 다른 값이다.
- 후보 모델은 정보 기준뿐 아니라 잔차 자기상관과 시간 순서 검증으로 확인해야 한다.

## 다음 글 예고

다음 글에서는 **지수평활**로 넘어가 최근 관측값에 더 큰 가중치를 주는 이유와 단순 지수평활의 갱신식을 살펴본다.

## 참고 자료

1. [Forecasting: Principles and Practice, ARIMA models](https://otexts.com/fpp3/arima.html) — ARIMA($p$,$d$,$q$)의 정의, ACF·PACF를 이용한 후보 식별, 정보 기준과 잔차 진단 절차.
2. [statsmodels 0.15.0, ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA) — `order=(p,d,q)` 인수와 AR·차분·MA 구성요소의 공식 정의.
