---
title: "머신러닝 입문 ⑤: 결정트리 분할 기준과 지니 불순도 손계산"
date: 2026-10-19 20:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [머신러닝, 결정트리, 지니-불순도, 분할-기준, 분류, Python]
description: "결정트리가 한 개의 질문으로 자료를 나누는 원리와 지니 불순도를 배웁니다. 일곱 개의 합성 자료에서 모든 후보를 계산해 최적 기준 4.5와 불순도 감소 0.276을 재현합니다."
reading_time: 12
series: "처음 시작하는 머신러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 결정트리가 수치 특성의 분할 기준을 만드는 방법과, 지니 불순도 감소로 더 나은 질문을 고르는 원리를 연결한다.

입력 `1, 2, 3, 4, 5, 6, 7`과 이진 레이블 `0, 0, 1, 0, 1, 1, 1`을 한 번만 나눠 보았다. 여섯 후보를 순수 Python으로 모두 비교한 결과, 최적 질문은 **`x <= 4.5`**, 분할 전후의 지니 불순도 감소는 **0.276**이었다. 이 질문 하나로 만든 깊이 1의 트리는 일곱 개 중 여섯 개를 맞혀 학습 정확도 **0.857**을 얻었다.

앞 글의 [로지스틱 회귀]({{ '/posts/machine-learning-logistic-regression-sigmoid-log-loss/' | relative_url }})는 하나의 식으로 모든 입력의 확률을 계산했다. 결정트리는 대신 “이 값 이하인가?”라는 질문을 이어 붙여 입력 공간을 여러 구역으로 나눈다.

## 1. 직관: 섞인 상자에서 한 종류씩 모으기

결정트리의 맨 위를 **루트 노드**, 질문 뒤에 생긴 집합을 **자식 노드**, 더 나누지 않고 예측을 내는 끝점을 **리프 노드**라고 한다. 분류 리프는 보통 그 안에서 가장 많은 클래스를 예측한다.

좋은 질문은 레이블이 뒤섞인 부모를 한쪽에는 0이, 다른 쪽에는 1이 더 많이 모이도록 나눈다. `x <= 4.5`를 적용하면 왼쪽 레이블은 `[0, 0, 1, 0]`, 오른쪽은 `[1, 1, 1]`이 된다. 오른쪽은 한 클래스만 남은 순수한 노드이고, 왼쪽도 0이 우세하다.

<figure class="study-figure">
  <img src="{{ '/assets/images/machine-learning/decision-tree-gini-split.svg' | relative_url }}" alt="레이블 0이 세 개, 1이 네 개인 부모 노드가 x가 4.5 이하인지 묻는 조건으로 나뉜다. 왼쪽 자식에는 0 세 개와 1 한 개가 있어 지니 불순도가 0.375이고, 오른쪽 자식에는 1 세 개만 있어 지니 불순도가 0이다. 가중 자식 불순도는 0.214, 감소량은 0.276이다">
  <figcaption>그림 1. `x <= 4.5`는 부모의 지니 불순도 0.490을 가중 자식 불순도 0.214로 낮춘다.</figcaption>
</figure>

## 2. 기호와 수식: 지니 불순도와 감소량

노드 $m$에서 클래스 $k$의 비율을 $p_{mk}$라 하면 지니 불순도는 다음과 같다.

$$
G(m)=1-\sum_{k=1}^{K}p_{mk}^{2}
$$

한 클래스만 있으면 그 비율의 제곱 합이 1이므로 $G=0$이다. 이진 분류에서 두 클래스가 절반씩 있으면 $G=1-(0.5^2+0.5^2)=0.5$로 가장 많이 섞인 상태가 된다. 값이 작을수록 한 클래스가 더 뚜렷하다.

부모 자료 수를 $N$, 왼쪽과 오른쪽 자식의 자료 수를 $N_L$, $N_R$이라 하자. 분할 뒤 불순도는 두 자식의 크기를 반영한 가중 평균이다. 분할의 개선 정도는 부모 불순도에서 이를 뺀 값으로 계산한다.

$$
G_{split}=\frac{N_L}{N}G(L)+\frac{N_R}{N}G(R)
$$

$$
\Delta G=G(parent)-G_{split}
$$

$\Delta G$가 클수록 한 번의 질문이 레이블을 더 잘 분리했다는 뜻이다. 실제 트리는 현재 노드에서 이 값을 크게 만드는 질문을 고른 뒤, 자식 노드에서도 같은 절차를 반복한다.

여기서 자식 불순도를 단순히 둘로 나누어 평균 내면 안 된다. 자료 1개가 든 순수한 자식과 자료 99개가 든 뒤섞인 자식을 같은 비중으로 취급하면, 전체 자료가 얼마나 잘 정리됐는지를 왜곡한다. 그래서 각 자식이 차지하는 자료 수 비율 $N_L/N$, $N_R/N$을 곱한다. 작은 노드를 순수하게 만드는 것보다 많은 자료의 섞임을 줄인 분할이 더 큰 개선으로 반영된다.

## 3. 작은 손계산: `x <= 4.5`가 만든 변화

부모에는 클래스 0이 3개, 클래스 1이 4개 있다. 따라서 부모 지니 불순도는 다음과 같다.

$$
G(parent)=1-\left(\frac{3}{7}\right)^2-\left(\frac{4}{7}\right)^2=\frac{24}{49}\approx0.490
$$

`x <= 4.5`인 왼쪽에는 0이 3개, 1이 1개 있으므로 $G(L)=1-(3/4)^2-(1/4)^2=0.375$다. 오른쪽에는 1만 3개 있어 $G(R)=0$이다. 자료 수까지 반영하면 다음 결과를 얻는다.

$$
G_{split}=\frac{4}{7}\times0.375+\frac{3}{7}\times0=0.214
$$

$$
\Delta G=0.490-0.214=0.276
$$

수치 특성에서는 정렬된 서로 다른 값 사이의 중간점을 후보로 둘 수 있다. 이 예의 후보는 1.5부터 6.5까지 여섯 개다. 손계산은 4.5만 확인했지만, 최적이라고 말하려면 나머지도 같은 방식으로 비교해야 한다.

중간점을 쓰는 이유도 확인해 보자. 정렬된 두 값이 4와 5라면 4.1, 4.5, 4.9는 모두 4까지를 왼쪽에, 5부터를 오른쪽에 보낸다. 자료의 소속이 같으므로 불순도도 같다. 이 구간을 대표하는 값 하나인 4.5만 시험하면 같은 분할을 중복 계산하지 않을 수 있다. 같은 입력값끼리는 서로 다른 자식으로 가를 수 없으므로, 서로 다른 인접값 사이만 후보가 된다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 패키지 없이 모든 후보의 가중 자식 불순도와 감소량을 계산한다. 마지막에는 최적 질문의 각 리프에서 다수 클래스를 골라 예측한다.

```python
from collections import Counter

x = [1, 2, 3, 4, 5, 6, 7]
y = [0, 0, 1, 0, 1, 1, 1]

def gini(labels):
    counts = Counter(labels)
    n = len(labels)
    return 1 - sum((count / n) ** 2 for count in counts.values())

parent = gini(y)
best = None
print(f"부모 지니: {parent:.3f}")

for left_x, right_x in zip(x, x[1:]):
    threshold = (left_x + right_x) / 2
    left = [label for value, label in zip(x, y) if value <= threshold]
    right = [label for value, label in zip(x, y) if value > threshold]
    weighted = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
    decrease = parent - weighted
    print(f"기준 {threshold:.1f}: 자식 지니 {weighted:.3f}, 감소 {decrease:.3f}")
    if best is None or decrease > best[0]:
        best = (decrease, threshold)

_, threshold = best
left_labels = [label for value, label in zip(x, y) if value <= threshold]
right_labels = [label for value, label in zip(x, y) if value > threshold]
left_prediction = Counter(left_labels).most_common(1)[0][0]
right_prediction = Counter(right_labels).most_common(1)[0][0]
predictions = [
    left_prediction if value <= threshold else right_prediction
    for value in x
]
accuracy = sum(pred == label for pred, label in zip(predictions, y)) / len(y)

print(f"최적 기준: x <= {threshold:.1f}")
print("예측:", predictions)
print(f"학습 정확도: {accuracy:.3f}")
```

실행 출력은 다음과 같다.

```text
부모 지니: 0.490
기준 1.5: 자식 지니 0.381, 감소 0.109
기준 2.5: 자식 지니 0.229, 감소 0.261
기준 3.5: 자식 지니 0.405, 감소 0.085
기준 4.5: 자식 지니 0.214, 감소 0.276
기준 5.5: 자식 지니 0.343, 감소 0.147
기준 6.5: 자식 지니 0.429, 감소 0.061
최적 기준: x <= 4.5
예측: [0, 0, 0, 0, 1, 1, 1]
학습 정확도: 0.857
```

## 5. 결과 해석: 가장 순수한 분할도 정답 암기는 아니다

4.5의 감소량 0.276이 여섯 후보 중 가장 크므로 첫 질문으로 선택된다. 왼쪽 리프는 다수인 0, 오른쪽 리프는 1을 예측한다. `x=3`의 실제 레이블 1만 틀려 정확도는 $6/7\approx0.857$이다. 한 단계만 나눈 트리는 모든 학습 자료를 외우지 않고 가장 큰 구조 하나를 포착했다.

자식 노드를 계속 나누면 이 한 개의 오류도 맞힐 수 있다. 그러나 깊이가 커질수록 학습 자료의 작은 예외까지 따라갈 수 있어 보지 않은 자료에서는 성능이 나빠질 수 있다. `max_depth`, 리프의 최소 자료 수 같은 제한은 이를 조절한다. 어느 깊이가 나은지는 [학습·검증·테스트 분리]({{ '/posts/machine-learning-train-validation-test-split/' | relative_url }})에서 배운 것처럼 검증 자료로 선택하고 테스트 자료로 마지막에 확인해야 한다.

결정트리의 질문은 확률을 바로 출력하는 로지스틱 회귀와 표현 방식이 다르다. 그래도 둘 다 [특성과 레이블로 지도학습 문제를 정의]({{ '/posts/machine-learning-supervised-features-labels/' | relative_url }})한 뒤, 정답과 예측의 차이를 줄이는 규칙을 학습한다는 점은 같다.

## 6. 직접 바꿔 보는 연습

1. `y`의 네 번째 값을 0에서 1로 바꾸고 최적 기준과 감소량을 기록한다.
2. `x=3`이 있는 왼쪽 자식을 다시 나누어 학습 자료를 모두 맞힐 수 있는지 확인한다.
3. 후보 하나를 골라 왼쪽과 오른쪽의 클래스 개수, 지니 불순도, 가중 평균을 손으로 계산해 코드 출력과 비교한다.

## 핵심 요약

- 결정트리는 하나의 조건으로 자료를 나누고, 리프에서는 그 구역의 다수 클래스를 예측할 수 있다.
- 지니 불순도는 클래스가 얼마나 섞였는지 나타내며, 0이면 한 클래스만 있는 순수한 노드다.
- 수치 특성의 후보 기준을 비교해 가중 자식 불순도를 가장 많이 낮추는 분할을 고르고, 트리의 깊이는 검증 자료로 조절한다.

## 다음 글 예고

다음 글에서는 **앙상블**을 다룬다. 서로 다른 여러 결정트리의 예측을 모으면 왜 한 트리의 흔들림을 줄일 수 있는지 배깅과 랜덤 포레스트의 작은 예제로 확인한다.

## 참고 자료

- [Google Machine Learning Glossary, Decision Forests](https://developers.google.com/machine-learning/glossary/df) — 결정트리의 조건·리프·루트와 지니 불순도의 정의.
- [Google Machine Learning, Exact splitter for binary classification](https://developers.google.com/machine-learning/decision-forests/binary-classification) — 수치 특성의 후보 기준과 분할 전후 불순도 비교 절차.
- [scikit-learn User Guide, Decision Trees](https://scikit-learn.org/stable/modules/tree.html) — 분류 트리의 수학적 정의와 과적합 조절 방법.
- [scikit-learn API, DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) — `criterion`, `max_depth`, `min_samples_leaf` 매개변수의 공식 설명.
