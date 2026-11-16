---
title: "머신러닝 입문 ⑨: K-겹 교차검증 평균과 표준편차 손계산"
date: 2026-11-16 20:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [머신러닝, 교차검증, K-겹, 모델-평가, MAE, Python]
description: "K-겹 교차검증에서 모든 자료가 한 번씩 검증에 쓰이는 과정을 배웁니다. 4개 폴드의 MAE를 순수 Python으로 계산해 평균 0.398과 표준편차 0.095를 재현합니다."
reading_time: 12
series: "처음 시작하는 머신러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** K-겹 교차검증에서 학습과 검증을 반복하는 이유를 이해하고, 폴드별 점수의 평균과 표준편차를 직접 계산한다.

12개 자료를 4개 폴드로 나눠 선형회귀를 네 번 새로 학습했다. 검증 평균절대오차(MAE)는 차례로 **0.322, 0.289, 0.517, 0.464**였다. 평균은 **0.398**, 표준편차는 **0.095**다. 한 번의 분할에서 나온 점수 하나 대신, 네 검증 조각에서 성능과 흔들림을 함께 확인한 결과다.

앞 글에서는 [순열 중요도가 평가 자료와 점수에 따라 달라지는 이유]({{ '/posts/machine-learning-permutation-feature-importance/' | relative_url }})를 살펴봤다. 그때처럼 모델을 비교하는 점수도 어떤 자료를 검증에 두느냐에 따라 달라질 수 있다. 이번에는 검증 역할을 여러 조각이 번갈아 맡게 한다.

## 1. 직관: 검증석을 한 번씩 돌아가며 맡는다

자료를 한 번만 학습 세트와 검증 세트로 나누면 결과가 간단하다. 그러나 우연히 쉬운 표본이 검증에 모이면 점수가 좋아지고, 어려운 표본이 모이면 나빠질 수 있다. **K-겹 교차검증(K-fold cross-validation)**은 전체 학습 자료를 겹치지 않는 $K$개 조각, 즉 폴드로 나눈다.

첫 번째 반복에서는 1번 폴드만 검증에 두고 나머지로 모델을 학습한다. 다음 반복에서는 2번 폴드가 검증 역할을 맡는다. 이 과정을 $K$번 반복하면 모든 자료가 정확히 한 번 검증에 쓰이고, 매 반복마다 모델은 처음부터 다시 학습된다.

<figure class="study-figure">
  <img src="{{ '/assets/images/machine-learning/k-fold-cross-validation.svg' | relative_url }}" alt="12개 자료를 네 폴드로 나누고 한 폴드씩 번갈아 검증에 사용하는 4겹 교차검증. 폴드별 MAE는 0.322, 0.289, 0.517, 0.464이며 평균은 0.398, 표준편차는 0.095다">
  <figcaption>그림 1. 네 번 모두 모델을 새로 학습하고, 검증 폴드만 차례로 바꾼다.</figcaption>
</figure>

## 2. 기호와 수식: K개 오차를 한 숫자로 모은다

$k$번째 반복의 검증 오차를 $e_k$라 하자. K개 폴드의 평균 오차 $\bar e$는 다음과 같다.

$$
\bar e=\frac{1}{K}\sum_{k=1}^{K}e_k
$$

폴드에 따라 점수가 얼마나 달라졌는지는 표준편차로 요약할 수 있다. 이 글에서는 K개 점수 자체의 흩어짐을 나타내는 모집단 표준편차를 계산한다.

$$
s=\sqrt{\frac{1}{K}\sum_{k=1}^{K}(e_k-\bar e)^2}
$$

평균 오차가 작을수록 네 검증 조각에서 대체로 예측이 정확했다는 뜻이다. 표준편차가 작으면 폴드를 바꿔도 점수가 비교적 일정했다는 뜻이다. 다만 이 표준편차를 곧바로 일반화 오차의 신뢰구간으로 해석해서는 안 된다. 각 반복의 학습 자료가 서로 겹쳐 폴드 점수들이 완전히 독립적이지 않기 때문이다.

## 3. 작은 손계산: 네 MAE의 평균과 흩어짐

네 폴드의 MAE를 더해 4로 나누면 다음과 같다.

$$
\bar e\approx\frac{0.322+0.289+0.517+0.464}{4}=0.398
$$

각 오차가 평균 0.398에서 떨어진 거리를 제곱해 평균낸 뒤 제곱근을 취하면 표준편차는 약 0.095다. 3번 폴드의 MAE 0.517이 가장 크므로, 그 검증 조각에 상대적으로 예측하기 어려운 자료가 있었음을 알 수 있다. 그렇다고 3번 폴드를 빼면 안 된다. 불편한 검증 결과도 새 자료에서 만날 수 있는 변동의 일부다.

## 4. 실행 가능한 Python 실습

외부 패키지 없이 12개 $(x,y)$ 자료를 네 폴드로 나눈다. 각 폴드에는 전체 범위가 섞이도록 4칸 간격의 인덱스를 넣었다. 매 반복에서 검증 폴드를 제외한 9개로 최소제곱 직선을 새로 구하고, 남겨 둔 3개의 MAE를 계산한다.

```python
from math import sqrt

x = list(range(1, 13))
y = [3, 5, 8, 9, 11, 13, 14, 17, 18, 20, 22, 23]
folds = [list(range(start, len(x), 4)) for start in range(4)]

def fit_line(train_idx):
    mean_x = sum(x[i] for i in train_idx) / len(train_idx)
    mean_y = sum(y[i] for i in train_idx) / len(train_idx)
    numerator = sum(
        (x[i] - mean_x) * (y[i] - mean_y) for i in train_idx
    )
    denominator = sum((x[i] - mean_x) ** 2 for i in train_idx)
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept

def mae(valid_idx, slope, intercept):
    errors = [
        abs(y[i] - (slope * x[i] + intercept)) for i in valid_idx
    ]
    return sum(errors) / len(errors)

scores = []
for fold_no, valid_idx in enumerate(folds, 1):
    train_idx = [i for i in range(len(x)) if i not in valid_idx]
    slope, intercept = fit_line(train_idx)
    score = mae(valid_idx, slope, intercept)
    scores.append(score)
    valid_x = [x[i] for i in valid_idx]
    print(f"{fold_no}겹: 검증 x={valid_x}, MAE={score:.3f}")

mean_score = sum(scores) / len(scores)
std_score = sqrt(
    sum((score - mean_score) ** 2 for score in scores) / len(scores)
)
print(f"평균 MAE: {mean_score:.3f}")
print(f"표준편차: {std_score:.3f}")
```

실행 출력은 다음과 같다.

```text
1겹: 검증 x=[1, 5, 9], MAE=0.322
2겹: 검증 x=[2, 6, 10], MAE=0.289
3겹: 검증 x=[3, 7, 11], MAE=0.517
4겹: 검증 x=[4, 8, 12], MAE=0.464
평균 MAE: 0.398
표준편차: 0.095
```

## 5. 결과 해석: 평균만 보고 끝내지 않는다

평균 MAE 0.398은 네 번의 검증 오차를 같은 비중으로 모은 값이다. 모델 후보가 둘이라면 같은 폴드를 사용해 평균 MAE를 비교할 수 있다. 동시에 폴드별 점수를 나란히 보면 특정 조각에서만 크게 무너지는 후보도 찾을 수 있다. 표준편차 0.095는 이 네 점수의 흩어짐이지, 모델 품질을 보증하는 합격선은 아니다.

두 후보를 비교할 때는 폴드 구성을 같게 고정하는 편이 좋다. 후보 A와 B가 서로 다른 검증 조각에서 점수를 받으면 모델 차이와 분할 차이가 섞여 어느 쪽이 나은지 읽기 어려워진다. 같은 네 폴드에서 얻은 A와 B의 점수를 폴드별로 나란히 놓으면, 두 후보가 같은 어려운 자료를 만났을 때 어떻게 달라지는지 확인할 수 있다. 평균이 조금 좋은 후보가 특정 폴드에서는 크게 나빠진다면 평균 하나만으로 선택하기보다 자료 구성과 오차 사례를 함께 살펴야 한다.

교차검증은 테스트 세트를 없애는 절차도 아니다. [학습·검증·테스트 자료의 역할]({{ '/posts/machine-learning-train-validation-test-split/' | relative_url }})에서처럼 먼저 테스트 자료를 따로 남겨 둔다. 남은 학습 자료 안에서 교차검증으로 모델이나 설정을 고른 뒤, 선택이 끝났을 때 테스트 자료를 한 번 사용한다. 테스트 자료까지 폴드에 넣어 선택하면 마지막 독립 평가를 잃는다.

교차검증 평균을 보고 여러 후보 중 하나를 골랐다면 그 평균도 이미 선택에 사용된 값이다. 후보를 계속 늘리면 우연히 그 폴드에 잘 맞은 설정을 고를 가능성도 커진다. 따라서 선택된 설정을 모든 학습 자료로 다시 학습한 뒤, 손대지 않은 테스트 세트에서 최종 성능을 확인해야 한다. 교차검증은 자료를 효율적으로 돌려 쓰는 선택 도구이지, 새 자료에서의 성능을 확정하는 장치는 아니다.

전처리에도 같은 경계를 적용해야 한다. 결측값 대치, 표준화, 특성 선택은 전체 자료에서 미리 맞추지 말고 각 반복의 학습 폴드에서만 학습해야 한다. 검증 폴드의 평균이나 레이블을 전처리에 사용하면 정보 누수가 생겨 점수가 실제보다 좋아질 수 있다. 라이브러리를 쓸 때는 전처리와 모델을 하나의 파이프라인으로 묶어 교차검증하는 이유가 여기에 있다.

폴드 수 $K$가 크면 매번 더 많은 자료로 학습하지만 모델을 더 많이 만들어야 한다. 자료가 12개인 이 예제에서는 구조를 눈으로 확인하려고 4겹을 썼다. 실제 분류에서는 각 폴드의 클래스 비율을 비슷하게 유지하는 층화 분할이 필요할 수 있고, 같은 사람에게서 나온 여러 행처럼 묶음이 있으면 한 묶음이 학습과 검증에 동시에 들어가지 않게 해야 한다.

자료에 시간 순서가 있다면 일반 K-겹을 그대로 쓰지 않는다. 미래가 학습 폴드에 들어가 과거를 검증하는 역전이 생길 수 있다. 이런 경우에는 [시간 순서를 지키는 롤링 검증 방식]({{ '/posts/time-series-rolling-origin-validation/' | relative_url }})처럼 예측 시점보다 앞선 자료만 학습에 사용해야 한다. 분할 방법은 알고리즘 이름이 아니라 새 자료가 들어오는 구조에 맞춰 고른다.

## 6. 직접 바꿔 보는 연습

1. `folds`를 3개 폴드로 바꾸고 평균 MAE와 표준편차가 어떻게 달라지는지 확인한다.
2. 3번 폴드의 세 $y$ 값에 각각 2를 더한 뒤 어떤 폴드 점수가 가장 크게 변하는지 살펴본다.
3. MAE 대신 평균제곱오차를 계산하고 큰 오차를 더 강하게 반영하는지 비교한다.

## 핵심 요약

- K-겹 교차검증은 K개 폴드가 한 번씩 검증을 맡도록 모델 학습과 평가를 K번 반복한다.
- 평균 점수는 전반적인 성능을, 폴드별 점수와 표준편차는 분할에 따른 흔들림을 보여 준다.
- 테스트 자료와 전처리 경계를 지켜야 하며, 시간·그룹·불균형 구조에 맞는 분할법을 골라야 한다.

## 다음 글 예고

다음 글에서는 **과대적합과 규제**를 다룬다. 모델이 학습 자료의 잡음까지 따라갈 때 검증 오차가 어떻게 달라지는지 보고, 규제가 복잡도를 누르는 원리를 계산한다.

## 참고 자료

- scikit-learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — K-겹 절차와 시간·그룹 구조에 맞는 분할 원칙을 설명하는 공식 사용자 안내서.
- scikit-learn, [KFold API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) — `n_splits`, `shuffle`, `random_state`와 반환 인덱스를 정의한 공식 API 문서.
- scikit-learn, [cross_validate API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) — 교차검증 분할과 평가 점수 설정을 정의한 공식 API 문서.
- scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — 전처리 누수와 파이프라인 사용 원칙을 설명하는 공식 지침.
