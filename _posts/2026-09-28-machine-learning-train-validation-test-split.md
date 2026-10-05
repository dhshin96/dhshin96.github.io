---
title: "머신러닝 입문 ②: 학습·검증·테스트 데이터 분리와 올바른 평가"
date: 2026-09-28 20:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [머신러닝, 데이터-분리, 학습-세트, 검증-세트, 테스트-세트, MAE, Python]
description: "학습·검증·테스트 데이터의 역할을 구별하고 테스트 세트를 마지막까지 보관해야 하는 이유를 배웁니다. 순수 Python으로 두 예측 규칙을 비교해 검증 MAE 4.0점인 규칙을 고르고 최종 테스트 MAE를 확인합니다."
reading_time: 11
series: "처음 시작하는 머신러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 같은 데이터를 학습·검증·테스트 세트로 나누고, 검증 세트로 예측 규칙을 고른 뒤 테스트 세트로 최종 오차를 한 번 확인한다.

10개의 합성 자료를 **학습 6개, 검증 2개, 테스트 2개**로 나눴다. 학습 자료만 사용해 만든 두 규칙의 검증 평균절대오차(MAE)는 각각 **8.0점과 4.0점**이다. 더 작은 4.0점의 규칙을 고른 뒤 처음으로 테스트 자료를 열었더니 최종 MAE도 **4.0점**이었다. 이 숫자는 모델을 복잡하게 만드는 실습이 아니라, 서로 다른 자료가 서로 다른 결정을 맡아야 함을 확인하는 결과다.

특성과 레이블, 회귀와 분류가 아직 낯설다면 [앞 글의 지도학습 문제 정의]({{ '/posts/machine-learning-supervised-features-labels/' | relative_url }})를 먼저 읽는 편이 좋다. 이번 글은 그때 남겨 둔 질문, “새 자료에서도 잘 맞는지 어떻게 확인할까?”에 답한다.

## 1. 직관: 공부 문제와 모의고사, 마지막 시험

학습 세트는 풀이법을 익히는 문제다. 모델은 이 자료에서 평균이나 가중치처럼 예측에 필요한 값을 배운다. 검증 세트는 여러 풀이법 중 하나를 고르는 모의고사다. 후보 모델, 설정값, 사용할 특성을 바꿀 때 그 선택을 돕는다. 테스트 세트는 선택을 모두 끝낸 뒤 여는 마지막 시험이다.

테스트 점수를 보고 규칙을 바꾼다면 그 자료는 더 이상 낯선 시험이 아니다. 정답을 직접 학습시키지 않았더라도 점수를 통해 선택에 영향을 주었기 때문이다. 테스트를 여러 번 보고 가장 잘 맞는 규칙을 택하면, 그 테스트의 우연한 특징에 맞춰질 수 있다. 따라서 **학습으로 만들고, 검증으로 고르고, 테스트로 마지막 확인을 한다**는 역할을 지켜야 한다.

<figure class="study-figure">
  <img src="{{ '/assets/images/machine-learning/train-validation-test-split.svg' | relative_url }}" alt="10개 합성 자료를 학습 6개, 검증 2개, 테스트 2개로 나눈 흐름. 학습 세트로 평균 규칙과 가장 가까운 값 규칙을 만들고 검증 MAE 8.0점과 4.0점을 비교해 두 번째 규칙을 선택한 뒤 테스트 세트에서 최종 MAE 4.0점을 한 번 확인">
  <figcaption>그림 1. 각 세트는 자료의 이름이 아니라 결정 과정에서 맡은 역할로 구별한다. 테스트 결과는 모델 선택으로 되돌려 보내지 않는다.</figcaption>
</figure>

## 2. 기호와 수식: 세 집합과 하나의 오차

전체 자료 $D$를 겹치지 않는 세 집합으로 나눈다.

$$
D = D_{\mathrm{train}} \cup D_{\mathrm{val}} \cup D_{\mathrm{test}},
\qquad
D_{\mathrm{train}} \cap D_{\mathrm{val}} = D_{\mathrm{train}} \cap D_{\mathrm{test}} = D_{\mathrm{val}} \cap D_{\mathrm{test}} = \varnothing.
$$

학습 세트 $D_{\mathrm{train}}$으로 후보 규칙 $f_1,f_2$를 만든다. 검증 세트에서 실제 점수 $y_i$와 예측 $hat y_i=f(x_i)$의 차이를 평균낸 MAE로 후보를 비교한다.

$$
\operatorname{MAE}(f;D)=\frac{1}{|D|}\sum_{(x_i,y_i)\in D}|y_i-f(x_i)|.
$$

MAE가 4점이면 해당 자료에서 예측이 실제값과 평균적으로 4점 떨어졌다는 뜻이다. 검증 MAE가 가장 작은 $f^*$를 고른 다음에만 $\operatorname{MAE}(f^*;D_{\mathrm{test}})$를 계산한다. 6·2·2는 흐름을 보기 위한 예시일 뿐, 모든 문제에 고정된 분할 비율은 아니다. 실제 비율은 자료의 크기와 대표성, 평가의 불확실성을 함께 고려해 정한다.

## 3. 작은 손계산: 검증 자료가 고르는 규칙

학습 자료는 공부 시간과 점수의 여섯 쌍 `(1,43), (2,47), (4,55), (6,63), (8,71), (10,79)`다. 검증 자료는 `(3,51), (7,67)`, 테스트 자료는 `(5,59), (9,75)`로 따로 둔다.

첫 후보는 학습 점수의 평균만 예측한다. 평균은 $(43+47+55+63+71+79)/6=59.67$점이다. 검증 실제값 51점과 67점에 대한 절대오차는 8.67점과 7.33점이므로 MAE는 8.0점이다.

둘째 후보는 새 공부 시간과 가장 가까운 학습 행의 점수를 가져온다. 3시간에는 2시간의 47점을, 7시간에는 6시간의 63점을 예측한다. 동률이면 먼저 나온 행을 고른다는 규칙까지 미리 정했다. 두 절대오차가 모두 4점이므로 검증 MAE는 4.0점이다. 여기서 둘째 후보를 선택한다. 아직 테스트의 59점과 75점은 규칙 선택에 쓰지 않았다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 패키지 없이 같은 계산을 재현한다. 두 후보는 오직 `train`에서 값을 배우거나 저장한다. `validation`은 후보 선택에, `test`는 선택이 끝난 뒤 한 번의 최종 평가에만 사용한다.

```python
train = [(1, 43), (2, 47), (4, 55), (6, 63), (8, 71), (10, 79)]
validation = [(3, 51), (7, 67)]
test = [(5, 59), (9, 75)]

def predict_mean(train_rows, x):
    return sum(y for _, y in train_rows) / len(train_rows)

def predict_nearest(train_rows, x):
    return min(train_rows, key=lambda row: abs(row[0] - x))[1]

def mae(rows, predictor):
    errors = [abs(y - predictor(train, x)) for x, y in rows]
    return sum(errors) / len(errors)

candidates = {
    "평균 규칙": predict_mean,
    "가장 가까운 값 규칙": predict_nearest,
}

print("학습/검증/테스트 크기:", len(train), len(validation), len(test))
for name, predictor in candidates.items():
    print(f"검증 MAE - {name}: {mae(validation, predictor):.1f}점")

selected_name, selected = min(
    candidates.items(), key=lambda item: mae(validation, item[1])
)
print("선택한 규칙:", selected_name)
print(f"최종 테스트 MAE: {mae(test, selected):.1f}점")
```

실행 출력은 다음과 같다.

```text
학습/검증/테스트 크기: 6 2 2
검증 MAE - 평균 규칙: 8.0점
검증 MAE - 가장 가까운 값 규칙: 4.0점
선택한 규칙: 가장 가까운 값 규칙
최종 테스트 MAE: 4.0점
```

## 5. 결과 해석: 테스트 세트는 선택 도구가 아니다

검증 결과는 둘째 규칙을 고르는 근거이고, 테스트 4.0점은 **선택된 규칙**을 새 자료에서 확인한 값이다. 같은 4.0점이라도 역할이 다르다. 테스트 MAE가 마음에 들지 않아 첫 규칙으로 돌아가거나 새 규칙을 추가하면, 테스트 결과가 선택에 개입한다. 그때는 별도의 새 테스트 자료가 필요하다.

분할은 전처리보다 먼저 생각해야 한다. 전체 자료의 평균으로 결측값을 채우거나 전체 자료를 보고 특성을 고르면 테스트 정보가 학습 과정에 흘러드는 **데이터 누수**가 생긴다. 전처리 기준도 학습 세트에서만 구하고, 같은 기준을 검증·테스트 세트에 적용해야 한다. scikit-learn을 쓸 때 고정된 정수를 `random_state`에 주면 무작위 분할을 다시 재현할 수 있지만, 재현 가능성이 올바른 분할 설계를 대신하지는 않는다.

행에 시간 순서가 있다면 무작위로 섞는 것부터 의심해야 한다. 미래로 과거를 학습시키지 않는 방법은 [시계열의 시간 순서 분할 실습]({{ '/posts/time-series-components-chronological-split/' | relative_url }})에서 별도로 확인할 수 있다. 같은 사람의 여러 기록이나 중복 표본이 있다면 서로 다른 세트로 흩어지지 않도록 묶어서 나누는 원칙도 필요하다.

## 6. 직접 바꿔 보는 연습

1. 테스트 자료를 보지 않고 검증 자료 `(3,51), (7,67)`의 두 MAE를 손으로 다시 계산한다.
2. `validation`의 첫 점수를 51에서 60으로 바꾼다. 어떤 규칙이 선택되는지, 선택 기준이 바뀌면 테스트를 다시 사용해도 되는지 설명한다.
3. 공부 시간 순서가 실제 날짜 순서라고 가정한다. 무작위 분할 대신 과거·현재·미래 순서로 세 세트를 만드는 방법을 적어 본다.

## 핵심 요약

- 학습 세트는 모델을 만들고, 검증 세트는 후보를 고르며, 테스트 세트는 선택이 끝난 뒤 최종 성능을 확인한다.
- 테스트 결과를 보고 모델이나 특성을 바꾸면 테스트가 선택 과정에 섞여 최종 평가의 의미가 약해진다.
- 분할은 전처리보다 먼저 설계하고, 시간·사람·중복처럼 행 사이의 관계도 함께 고려한다.

## 다음 글 예고

다음 글에서는 [선형회귀의 기울기·절편과 최소제곱법]({{ '/posts/machine-learning-linear-regression-least-squares/' | relative_url }})을 다룬다. 다섯 점에 가장 잘 맞는 직선을 손으로 구하고, 예측 오차를 줄이는 기준을 Python으로 확인한다.

## 참고 자료

- [Google for Developers, Datasets: Dividing the original dataset](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets) — 학습·검증·테스트 세트의 역할과 테스트 반복 사용의 문제.
- [scikit-learn User Guide, Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) — 테스트 정보가 전처리와 모델 선택에 섞이는 데이터 누수 방지 원칙.
- [scikit-learn API Reference, train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — 무작위 분할, `random_state`, `shuffle`, `stratify` 매개변수의 공식 설명.
