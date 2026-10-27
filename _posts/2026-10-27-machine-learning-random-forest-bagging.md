---
title: "머신러닝 입문 ⑥: 배깅과 랜덤 포레스트 부트스트랩·특성 무작위화"
date: 2026-10-27 20:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [머신러닝, 앙상블, 배깅, 랜덤-포레스트, 부트스트랩, Python]
description: "배깅이 부트스트랩 표본으로 여러 결정트리를 만드는 원리와 랜덤 포레스트가 특성 무작위화를 더하는 이유를 배웁니다. 다섯 나무의 서로 다른 예측을 다수결로 합쳐 최종 예측 0을 재현합니다."
reading_time: 12
series: "처음 시작하는 머신러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 부트스트랩 표본과 특성 무작위화가 서로 다른 나무를 만드는 과정을 이해하고, 여러 분류 결과를 하나의 예측으로 모으는 방법을 직접 계산한다.

여덟 개 학습 자료에서 복원추출한 표본으로 깊이 1의 나무 다섯 개를 만들었다. 입력 `(공부 시간 3.5, 결석 횟수 6)`에 대한 각 나무의 예측은 **`[1, 0, 1, 0, 0]`**으로 달랐다. 하지만 다수결을 적용하면 클래스 0이 3표를 얻어 최종 예측은 **0**, 클래스 1의 투표 비율은 **0.4**로 고정된다.

앞 글에서 배운 [결정트리의 분할 기준과 지니 불순도]({{ '/posts/machine-learning-decision-tree-gini-split/' | relative_url }})를 그대로 사용한다. 이번에는 한 나무를 더 깊게 만드는 대신, 학습 자료와 분할 특성을 조금씩 바꾼 여러 나무의 판단을 모은다.

## 1. 직관: 같은 문제를 조금씩 다른 모둠에 맡기기

결정트리는 학습 자료가 조금만 달라져도 첫 분할과 뒤따르는 가지가 크게 달라질 수 있다. 이런 성질을 **높은 분산**이라고 한다. 한 나무의 예측은 명확하지만, 어떤 자료가 학습에 들어왔는지에 민감하다는 뜻이다.

**배깅(bagging)**은 원래 학습 자료에서 같은 크기의 표본을 여러 번 뽑아 모델을 각각 학습하고, 예측을 합친다. 표본을 뽑은 뒤에도 다시 넣는 **복원추출**이므로 한 표본 안에 같은 자료가 두 번 이상 들어갈 수 있고 어떤 자료는 빠질 수 있다. 이렇게 얻은 표본을 부트스트랩 표본이라고 한다.

첫 번째 실습 표본 `[8, 5, 6, 3, 7, 1, 6, 8]`을 보자. 원래 자료 8개와 크기는 같지만 6번과 8번은 두 번씩 들어가고 2번과 4번은 빠졌다. 반복된 자료는 그 나무의 분할 기준을 정할 때 더 큰 영향을 주며, 빠진 자료는 영향을 주지 않는다. 다음 나무는 다시 독립적으로 뽑기 때문에 중복과 누락의 조합도 바뀐다. 배깅은 원본을 다섯 조각으로 나누는 방법이 아니라, 원본에서 크기가 같은 새 학습 세트를 다섯 번 만드는 방법이다. 따라서 각 나무는 전체 규모의 자료로 학습하면서도 조금씩 다른 경계를 얻게 된다.

**랜덤 포레스트**는 결정트리 배깅에 한 가지 무작위성을 더한다. 각 노드에서 모든 특성을 비교하지 않고 무작위로 고른 일부 특성만 분할 후보로 본다. 강한 특성 하나가 모든 나무의 첫 질문을 차지하는 현상을 줄여 나무들이 서로 다른 실수를 하도록 유도한다. 서로 같은 실수를 반복하는 나무를 여러 개 모으는 것보다, 판단이 다양한 나무를 모을 때 평균이나 투표가 흔들림을 더 잘 상쇄한다.

<figure class="study-figure">
  <img src="{{ '/assets/images/machine-learning/random-forest-bagging.svg' | relative_url }}" alt="여덟 개 학습 자료에서 복원추출한 서로 다른 표본과 무작위 특성으로 다섯 결정트리를 만든다. 새 입력에 대한 예측은 1, 0, 1, 0, 0이며 클래스 0이 세 표를 얻어 최종 예측 0이 된다">
  <figcaption>그림 1. 부트스트랩 표본과 무작위 특성으로 나무를 다양하게 만들고, 다섯 예측을 다수결로 합친다.</figcaption>
</figure>

## 2. 기호와 수식: 복원추출과 다수결

학습 자료가 $D=\{(x_i,y_i)\}_{i=1}^{n}$일 때, $b$번째 나무는 $D$에서 $n$개를 복원추출한 부트스트랩 표본 $D_b$로 학습한다. 같은 인덱스가 반복되어도 허용한다.

$$
D_b=\{(x_{i_1},y_{i_1}),\ldots,(x_{i_n},y_{i_n})\},\qquad i_j\sim\text{Uniform}\{1,\ldots,n\}
$$

$B$개 분류 나무의 예측을 $h_b(x)$라 하면, 가장 많은 표를 받은 클래스를 최종 예측으로 선택할 수 있다.

$$
\hat y(x)=\underset{k}{\operatorname{argmax}}\sum_{b=1}^{B}\mathbf{1}\bigl(h_b(x)=k\bigr)
$$

회귀라면 다수결 대신 $\hat y(x)=\frac{1}{B}\sum_{b=1}^{B}h_b(x)$처럼 평균을 낸다. 랜덤 포레스트의 핵심은 이 집계식보다 각 나무를 다르게 만드는 과정에 있다. 부트스트랩으로 행을 바꾸고, 각 분할에서 살펴볼 특성 열도 제한한다.

## 3. 작은 손계산: 다섯 표를 하나로 모으기

새 입력에 대한 다섯 나무의 예측이 `1, 0, 1, 0, 0`이라고 하자. 클래스 0의 표는 3개, 클래스 1의 표는 2개이므로 다수결 결과는 0이다.

클래스 1에 투표한 비율도 계산할 수 있다.

$$
\frac{1+0+1+0+0}{5}=\frac{2}{5}=0.4
$$

이 값은 다섯 개의 이진 투표를 평균 낸 결과이지, 언제나 보정된 확률이라는 뜻은 아니다. 또한 다수결이 자동으로 정답을 보장하지도 않는다. 나무들이 같은 편향을 공유하면 함께 틀릴 수 있으므로, 보지 않은 자료에서 단일 나무와 앙상블을 공정하게 비교해야 한다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 패키지 없이 두 특성 중 하나를 무작위로 고르고, 부트스트랩 표본에서 지니 불순도가 가장 작은 기준을 찾는 결정 그루터기 다섯 개를 만든다. `random.Random(12)`로 표본과 특성 선택을 고정해 출력도 재현할 수 있다.

```python
import random
from collections import Counter

X = [(1, 9), (2, 8), (3, 3), (4, 7),
     (5, 4), (6, 6), (7, 2), (8, 5)]
y = [0, 0, 0, 1, 1, 1, 1, 1]
query = (3.5, 6)

def gini(labels):
    counts = Counter(labels)
    n = len(labels)
    return 1 - sum((count / n) ** 2 for count in counts.values())

def train_stump(indices, feature):
    values = sorted({X[i][feature] for i in indices})
    thresholds = [(a + b) / 2 for a, b in zip(values, values[1:])]
    best = None
    for threshold in thresholds:
        left = [y[i] for i in indices if X[i][feature] <= threshold]
        right = [y[i] for i in indices if X[i][feature] > threshold]
        if not left or not right:
            continue
        score = (len(left) * gini(left) + len(right) * gini(right)) / len(indices)
        candidate = (score, threshold)
        if best is None or candidate < best:
            best = candidate
    _, threshold = best
    left_y = [y[i] for i in indices if X[i][feature] <= threshold]
    right_y = [y[i] for i in indices if X[i][feature] > threshold]
    return (threshold,
            Counter(left_y).most_common(1)[0][0],
            Counter(right_y).most_common(1)[0][0])

rng = random.Random(12)
votes = []
for tree in range(1, 6):
    sample = [rng.randrange(len(X)) for _ in X]
    feature = rng.randrange(2)
    threshold, left_pred, right_pred = train_stump(sample, feature)
    prediction = left_pred if query[feature] <= threshold else right_pred
    votes.append(prediction)
    shown = [i + 1 for i in sample]
    print(f"트리 {tree}: 표본 {shown}, 특성 {feature + 1}, "
          f"기준 {threshold:.1f}, 예측 {prediction}")

print("투표:", votes)
print("최종 예측:", Counter(votes).most_common(1)[0][0])
print(f"클래스 1 투표 비율: {sum(votes) / len(votes):.1f}")
```

실행 출력은 다음과 같다.

```text
트리 1: 표본 [8, 5, 6, 3, 7, 1, 6, 8], 특성 2, 기준 7.5, 예측 1
트리 2: 표본 [8, 4, 1, 3, 8, 6, 3, 6], 특성 1, 기준 3.5, 예측 0
트리 3: 표본 [1, 4, 2, 6, 7, 2, 1, 1], 특성 1, 기준 3.0, 예측 1
트리 4: 표본 [2, 7, 8, 2, 7, 3, 6, 3], 특성 1, 기준 4.5, 예측 0
트리 5: 표본 [3, 2, 7, 7, 8, 8, 7, 1], 특성 1, 기준 5.0, 예측 0
투표: [1, 0, 1, 0, 0]
최종 예측: 0
클래스 1 투표 비율: 0.4
```

## 5. 결과 해석: 차이가 있어야 모으는 효과가 생긴다

각 표본에는 중복과 누락이 있고, 첫 번째 나무만 특성 2를 선택했다. 그 결과 기준이 3.0부터 7.5까지 달라지고 같은 입력에 대한 예측도 둘로 갈렸다. 개별 나무 하나를 임의로 고르면 결과가 0일 수도 1일 수도 있지만, 정해진 다섯 나무를 모두 사용하면 집계 규칙은 0을 반환한다.

이 실습은 원리를 드러내려고 각 나무를 분할 한 번짜리 그루터기로 제한했다. 실제 랜덤 포레스트는 각 노드에서 특성 후보를 다시 뽑으며 여러 층의 트리를 만든다. 나무 수, 특성 후보 수, 깊이 같은 값은 [학습·검증·테스트 분리]({{ '/posts/machine-learning-train-validation-test-split/' | relative_url }})에서처럼 검증 자료로 선택하고, 마지막 성능은 한 번 남겨 둔 테스트 자료로 확인해야 한다.

표본 복원추출과 특성 후보 제한은 목적이 조금 다르다. 전자는 나무마다 보는 사례를 바꾸고, 후자는 나무마다 고려하는 질문을 바꾼다. 랜덤 포레스트는 두 장치를 함께 써서 나무 사이의 상관을 낮추려 한다. 다만 특성 후보를 지나치게 줄이면 각 나무가 유용한 신호를 놓쳐 편향이 커질 수 있으므로, 무작위성이 많을수록 무조건 좋다고 해석하면 안 된다.

여기서 “분산을 줄인다”는 말은 다섯 예측값이 서로 같아진다는 뜻이 아니다. 학습 자료를 조금 바꿔 다시 학습했을 때 **최종 앙상블의 예측**이 단일 나무보다 덜 바뀌기를 기대한다는 뜻이다. 이 효과는 나무 수만으로 결정되지 않는다. 나무들이 거의 같은 자료와 같은 특성으로 같은 경계를 만들면 오류도 함께 움직인다. 랜덤 포레스트가 표본과 특성이라는 두 축에 무작위성을 넣는 이유가 여기에 있다.

## 6. 직접 바꿔 보는 연습

1. `random.Random(12)`의 12를 7로 바꾸고 표본·특성·투표가 어떻게 달라지는지 기록한다.
2. `range(1, 6)`을 `range(1, 12)`로 바꿔 나무를 11개로 늘리고 최종 예측과 투표 비율을 비교한다.
3. `feature = rng.randrange(2)`를 `feature = 0`으로 고정해 특성 무작위화가 없는 배깅을 만들고, 기준과 예측의 다양성을 비교한다.

## 핵심 요약

- 배깅은 학습 자료를 복원추출해 여러 모델을 학습하고, 분류는 투표·회귀는 평균으로 예측을 합친다.
- 랜덤 포레스트는 결정트리 배깅에 분할 특성의 무작위 선택을 더해 나무들이 같은 실수를 반복하는 정도를 줄인다.
- 앙상블도 자동으로 정답이 되지는 않는다. 나무 수와 특성 수 같은 설정은 검증 자료로 고르고 테스트 자료는 마지막 평가에만 사용한다.

## 다음 글 예고

다음 글에서는 **불균형 분류**를 다룬다. 정확도만 보면 놓치기 쉬운 소수 클래스를 혼동행렬·정밀도·재현율로 평가하고, 기준값을 바꿀 때 두 지표가 어떻게 달라지는지 계산한다.

## 참고 자료

- [Breiman, Bagging Predictors](https://statistics.berkeley.edu/tech-reports/421) — 부트스트랩 표본으로 여러 예측기를 만들고 분류 투표·회귀 평균으로 결합하는 원 논문 기록.
- [Breiman, Random Forests—Random Features](https://statistics.berkeley.edu/tech-reports/567) — 나무의 강도와 상관, 분할 특성 무작위화의 관계를 설명한 원 논문 기록.
- [scikit-learn User Guide, Random forests and bagging](https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles) — 랜덤 포레스트와 배깅의 학습 절차 및 주요 설정에 대한 공식 문서.
