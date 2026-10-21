---
title: "시계열 예측 입문 ④: MAE·RMSE·MAPE 차이와 선택 기준"
date: 2026-10-21 20:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [시계열, 시계열-예측, 예측-오차, MAE, RMSE, MAPE, Python]
description: "같은 예측 오차에 MAE·RMSE·MAPE를 적용해 값과 해석의 차이를 확인합니다. 순수 Python으로 4.67, 4.73, 3.91%를 재현하고 지표 선택 기준을 익힙니다."
reading_time: 11
series: "처음 시작하는 시계열 예측"
level: "입문"
math: true
---

> **이번 글의 목표:** 같은 실제값과 예측값으로 MAE·RMSE·MAPE를 직접 계산하고, 큰 오차와 0에 가까운 실제값이 각 지표에 어떤 영향을 주는지 구분한다.

먼저 결과부터 보자. [앞 글의 롤링 검증]({{ '/posts/time-series-rolling-origin-validation/' | relative_url }})에서 얻은 여섯 예측에 세 지표를 적용하면 **MAE 4.67, RMSE 4.73, MAPE 3.91%**가 나온다. 한 자료를 평가했는데도 숫자가 다른 이유는 각 지표가 오차를 모으는 규칙이 다르기 때문이다.

## 1. 직관: 같은 오차표를 서로 다른 렌즈로 본다

예측값과 실제값의 차이를 한 줄로 세우면 `[5, -5, 3, 5, 5, -5]`다. 양수는 실제값보다 낮게 예측했고, 음수는 높게 예측했다는 뜻이다. 부호를 그대로 평균 내면 서로 상쇄되므로 예측이 정확해 보일 수 있다. 세 지표는 이 상쇄를 막되 서로 다른 질문을 던진다.

- **MAE**는 오차의 절댓값을 평균 낸다. “보통 원래 단위로 얼마나 빗나갔나?”에 답하기 쉽다.
- **RMSE**는 오차를 제곱해 평균 낸 뒤 제곱근을 취한다. 큰 오차가 생기면 더 민감하게 커진다.
- **MAPE**는 실제값 대비 절대오차의 비율을 평균 낸다. 단위가 다른 자료를 백분율로 읽기 쉽지만, 실제값이 0이거나 0에 가까우면 불안정하다.

세 값은 낮을수록 좋지만 서로 바꿔 읽을 수 없다. MAE 4.67은 원래 자료 단위의 평균 절대오차이고, MAPE 3.91%는 실제값을 분모로 둔 평균 상대오차다. 평가할 모델들은 반드시 같은 테스트 시점과 같은 지표로 비교해야 한다.

## 2. 기호와 수식: 절댓값, 제곱, 비율의 차이

시점 $t$의 실제값을 $y_t$, 예측값을 $\hat{y}_t$, 오차를 $e_t=y_t-\hat{y}_t$라고 하자. 평가 시점이 $n$개라면 MAE는 다음과 같다.

$$
\mathrm{MAE}=\frac{1}{n}\sum_{t=1}^{n}|e_t|
$$

RMSE는 큰 오차를 제곱한 뒤 평균 내므로, 다른 오차가 같다면 한 번의 큰 실패가 점수를 더 많이 끌어올린다. 마지막에 제곱근을 취해 값의 단위를 원래 자료와 맞춘다.

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{t=1}^{n}e_t^2}
$$

MAPE는 각 절대오차를 해당 시점의 실제값으로 나눈다.

$$
\mathrm{MAPE}=\frac{100}{n}\sum_{t=1}^{n}\left|\frac{e_t}{y_t}\right|
$$

이 식은 $y_t=0$이면 정의할 수 없다. $y_t$가 아주 작아도 작은 절대오차가 매우 큰 백분율로 바뀐다. 섭씨처럼 0이 임의의 기준점인 척도에도 “실제값 대비 몇 퍼센트”라는 해석이 맞지 않는다. MAPE를 쓰기 전에는 0과 0 근처 값, 그리고 0의 의미부터 확인해야 한다.

## 3. 작은 손계산: MAE는 같아도 RMSE는 달라진다

이번 여섯 예측의 절대오차는 `[5, 5, 3, 5, 5, 5]`다. 합이 28이므로 MAE는 `28 ÷ 6 = 4.67`이다. 제곱오차의 합은 `25+25+9+25+25+25=134`이므로 RMSE는 다음처럼 4.73이다.

$$
\mathrm{RMSE}=\sqrt{134/6}=4.73
$$

총절대오차 28을 마지막 한 시점에만 몰아 `[0, 0, 0, 0, 0, 28]`로 바꿔 보자. MAE는 여전히 4.67이지만 RMSE는 $\sqrt{784/6}=11.43$으로 커진다. 제곱이 큰 오차 하나를 강조했기 때문이다.

<figure class="study-figure">
  <img src="{{ '/assets/images/time-series/mae-rmse-mape.svg' | relative_url }}" alt="총절대오차가 28로 같은 두 예측의 오차 분포를 비교한다. 고른 오차는 MAE 4.67과 RMSE 4.73이고, 한 시점에 28이 몰린 오차는 MAE 4.67과 RMSE 11.43이다. 아래에는 실제값 100, 10, 1에서 같은 절대오차 1이 각각 1퍼센트, 10퍼센트, 100퍼센트가 되는 MAPE 예가 있다.">
  <figcaption>그림 1. MAE는 총절대오차가 같으면 같지만 RMSE는 큰 오차가 몰릴수록 커진다. MAPE는 실제값이 작아질수록 같은 절대오차를 크게 본다.</figcaption>
</figure>

MAPE의 분모 효과도 간단하다. 실제값이 `[100, 10, 1]`이고 모두 1만큼 빗나가면 절대오차는 같지만 백분율 오차는 `[1%, 10%, 100%]`다. 평균은 37%다. 작은 실제값 하나가 전체 MAPE를 지배할 수 있다는 뜻이다.

## 4. 실행 가능한 Python 실습

외부 패키지 없이 세 지표와 큰 오차 비교를 재현한다. MAPE는 0으로 나누는 상황을 숨기지 않도록 실제값이 0이면 오류를 내게 했다.

```python
from math import sqrt

actual = [118, 113, 116, 121, 126, 121]
predicted = [113, 118, 113, 116, 121, 126]

errors = [y - y_hat for y, y_hat in zip(actual, predicted)]
absolute_errors = [abs(error) for error in errors]

mae = sum(absolute_errors) / len(errors)
rmse = sqrt(sum(error ** 2 for error in errors) / len(errors))

if any(y == 0 for y in actual):
    raise ValueError("MAPE는 실제값이 0이면 정의할 수 없습니다.")
mape = 100 * sum(error / y for error, y in zip(absolute_errors, actual)) / len(errors)

concentrated_errors = [0, 0, 0, 0, 0, 28]
concentrated_mae = sum(concentrated_errors) / len(concentrated_errors)
concentrated_rmse = sqrt(
    sum(error ** 2 for error in concentrated_errors) / len(concentrated_errors)
)

print("오차:", errors)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"집중 오차 MAE: {concentrated_mae:.2f}")
print(f"집중 오차 RMSE: {concentrated_rmse:.2f}")
```

실행 출력은 다음과 같다.

```text
오차: [5, -5, 3, 5, 5, -5]
MAE: 4.67
RMSE: 4.73
MAPE: 3.91%
집중 오차 MAE: 4.67
집중 오차 RMSE: 11.43
```

## 5. 결과 해석: 지표는 평가 목적을 숫자로 바꾼다

이번 기본 오차는 3과 5 사이에 모여 있어 MAE와 RMSE가 비슷하다. 반면 집중 오차 예에서는 RMSE만 크게 늘었다. 드문 큰 오차를 강하게 경계해야 한다면 RMSE가 그 요구를 더 잘 드러낸다. 각 시점의 절대적인 빗나감 크기를 단순하게 설명하려면 MAE가 편하다.

MAPE 3.91%는 이 여섯 실제값이 모두 113 이상이어서 안정적으로 계산됐다. 그렇다고 MAPE가 언제나 더 공정한 것은 아니다. 0이 포함된 자료에는 수식 자체를 적용할 수 없고, 0 근처 값이 많으면 지표가 급격히 커진다. 서로 규모가 다른 여러 시계열을 비교해야 한다면 MAPE의 분모 조건을 점검하고, 맞지 않으면 나이브 기준선으로 오차를 조정하는 MASE 같은 대안을 검토한다.

지표를 고른 뒤에는 [나이브 기준선 글]({{ '/posts/time-series-naive-forecast-baseline/' | relative_url }})에서처럼 단순한 예측과 같은 조건으로 비교해야 한다. 지표를 여러 개 계산할 수는 있지만 결과를 본 뒤 유리한 지표만 고르면 평가 기준이 흔들린다. 먼저 “평균적인 절대오차”와 “큰 오차 회피” 중 무엇이 중요한지 정하고 주지표를 고르는 편이 명확하다.

## 6. 직접 바꿔 보는 연습

1. `predicted`의 마지막 값을 126에서 149로 바꾸고 MAE와 RMSE가 각각 얼마나 늘어나는지 확인한다.
2. 실제값 `[100, 10, 1]`, 예측값 `[99, 9, 0]`으로 MAPE 37%를 재현한다.
3. `actual`에 0을 하나 넣어 방어 코드가 `ValueError`를 내는지 확인하고, 그 자료에 MAPE를 쓰면 안 되는 이유를 한 문장으로 적는다.

## 핵심 요약

- MAE는 절대오차의 평균이라 원래 단위로 읽기 쉽다.
- RMSE는 오차를 제곱해 큰 오차에 더 민감하고, 제곱근으로 원래 단위를 회복한다.
- MAPE는 상대오차를 백분율로 보여 주지만 실제값 0에서는 정의되지 않고 0 근처에서 불안정하다.
- 같은 롤링 예측에서 MAE는 4.67, RMSE는 4.73, MAPE는 3.91%였다.
- 지표는 결과를 본 뒤 고르는 장식이 아니라, 어떤 예측 실패를 더 중요하게 볼지 정하는 평가 규칙이다.

## 다음 글 예고

다음 글에서는 **자기상관과 지연값**을 다룬다. 현재 값이 과거 값과 얼마나 연결되어 있는지 자기상관계수와 간단한 ACF 계산으로 확인한다.

## 참고 자료

1. [Forecasting: Principles and Practice, Evaluating point forecast accuracy](https://otexts.com/fpp3/accuracy.html) — MAE·RMSE의 정의와 해석, MAPE의 0 및 척도 문제.
2. [scikit-learn 공식 문서, Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) — MAE·MSE·RMSE·MAPE 공식과 Python 지표 정의.
