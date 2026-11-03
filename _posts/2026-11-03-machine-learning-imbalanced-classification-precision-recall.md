---
title: "머신러닝 입문 ⑦: 불균형 분류 혼동행렬과 정밀도·재현율 손계산"
date: 2026-11-03 19:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [머신러닝, 불균형-분류, 혼동행렬, 정밀도, 재현율, Python]
description: "불균형 분류에서 정확도만 보면 소수 클래스를 놓치는 이유를 혼동행렬로 확인합니다. 같은 점수의 기준값을 0.5에서 0.3으로 바꾸며 정밀도와 재현율의 변화를 재현합니다."
reading_time: 12
series: "처음 시작하는 머신러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 불균형 분류의 예측을 혼동행렬로 나누고, 정밀도·재현율을 계산해 분류 기준값을 바꿀 때 무엇이 달라지는지 설명한다.

양성 4개와 음성 16개가 있는 자료 20개를 분류했다. 모든 자료를 음성으로만 예측해도 정확도는 **80%**지만 양성을 한 개도 찾지 못해 재현율은 **0%**다. 같은 자료에서 점수 기준값을 0.5로 두면 정밀도와 재현율이 각각 **50.0%**, 기준값을 0.3으로 낮추면 **42.9%**와 **75.0%**가 된다.

앞 글에서 [배깅과 랜덤 포레스트가 여러 분류 결과를 모으는 원리]({{ '/posts/machine-learning-random-forest-bagging/' | relative_url }})를 배웠다. 이제 모델이 낸 하나의 분류 결과를 “맞음·틀림” 두 칸이 아니라 실제 양성 여부와 예측 양성 여부를 함께 보는 네 칸으로 펼쳐 보자.

## 1. 직관: 맞힌 개수만으로는 놓친 양성이 보이지 않는다

**불균형 분류**는 한 클래스가 다른 클래스보다 훨씬 적은 분류 문제다. 이 글에서는 적은 쪽인 양성을 우리가 찾아야 할 대상으로 둔다. 자료 20개 중 실제 양성이 4개뿐일 때 모두 음성이라고 답하면 음성 16개는 맞힌다. 정확도는 $16/20=0.8$이지만, 이 규칙은 실제 양성 4개를 전부 놓친다.

정확도 계산이 틀린 것은 아니다. 전체에서 맞힌 비율이라는 질문에는 정확히 답한다. 다만 “찾아야 할 양성을 얼마나 찾았는가?”와 “양성이라고 한 예측을 얼마나 믿을 수 있는가?”에는 답하지 않는다. 이 두 질문을 분리하려면 먼저 각 예측을 네 종류로 세어야 한다.

- **진짜 양성(TP)**: 실제 양성을 양성으로 예측
- **거짓 양성(FP)**: 실제 음성을 양성으로 잘못 예측
- **거짓 음성(FN)**: 실제 양성을 음성으로 잘못 예측
- **진짜 음성(TN)**: 실제 음성을 음성으로 예측

이 네 개의 수를 표로 놓은 것이 **혼동행렬(confusion matrix)**이다. 여기서 ‘양성’은 좋은 결과라는 뜻이 아니라, 문제를 정의할 때 관심 대상으로 정한 클래스다.

<figure class="study-figure">
  <img src="{{ '/assets/images/machine-learning/imbalanced-classification-precision-recall.svg' | relative_url }}" alt="실제 양성 4개와 음성 16개인 자료에서 기준값 0.5의 혼동행렬은 TP 2, FP 2, FN 2, TN 14이고 정밀도와 재현율은 모두 50.0퍼센트다. 기준값 0.3에서는 TP 3, FP 4, FN 1, TN 12가 되어 정밀도 42.9퍼센트, 재현율 75.0퍼센트가 된다">
  <figcaption>그림 1. 기준값을 낮추자 양성 예측이 늘어 재현율은 높아지고, 거짓 양성도 늘어 정밀도는 낮아졌다.</figcaption>
</figure>

## 2. 기호와 수식: 분모가 말해 주는 질문

정확도는 네 칸 전체에서 맞은 두 칸의 비율이다.

$$
\text{Accuracy}=\frac{TP+TN}{TP+FP+FN+TN}
$$

**정밀도(precision)**는 모델이 양성이라고 예측한 $TP+FP$개 가운데 실제 양성의 비율이다. “양성 예측을 얼마나 믿을 수 있는가?”에 가깝다.

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

**재현율(recall)**은 실제 양성 $TP+FN$개 가운데 모델이 찾아낸 비율이다. “실제 양성을 얼마나 놓치지 않았는가?”에 가깝다.

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

정밀도는 예측 양성이 분모이고, 재현율은 실제 양성이 분모다. 이름을 외우기보다 어떤 집합을 기준으로 비율을 묻는지 확인하면 두 식을 구분하기 쉽다. 양성 예측이 하나도 없으면 $TP+FP=0$이므로 정밀도는 수학적으로 정의되지 않는다는 점도 함께 기억하자.

## 3. 작은 손계산: 기준값 0.5와 0.3 비교

모델이 각 자료에 0부터 1 사이의 양성 점수를 냈다고 하자. [로지스틱 회귀의 시그모이드 출력]({{ '/posts/machine-learning-logistic-regression-sigmoid-log-loss/' | relative_url }})처럼 연속적인 점수를 분류로 바꾸려면 기준값 이상을 양성으로 정한다.

기준값 0.5에서는 $TP=2$, $FP=2$, $FN=2$, $TN=14$다. 따라서 정확도는 $(2+14)/20=0.8$, 정밀도는 $2/(2+2)=0.5$, 재현율은 $2/(2+2)=0.5$다. 모두 음성이라고 한 규칙과 정확도는 같지만, 이번 모델은 양성 두 개를 찾았다는 차이가 혼동행렬에 드러난다.

기준값을 0.3으로 낮추면 양성으로 예측하는 범위가 넓어진다. $TP=3$, $FP=4$, $FN=1$, $TN=12$가 되어 정밀도는 $3/7\approx0.429$, 재현율은 $3/4=0.75$다. 양성을 하나 더 찾은 대신 음성 두 개를 양성으로 더 잘못 표시했다. 기준값을 낮추면 이 예시처럼 재현율이 높아지고 정밀도가 낮아질 수 있지만, 정밀도가 모든 자료에서 반드시 단조롭게 변한다고 단정할 수는 없다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 패키지 없이 같은 20개 점수에 두 기준값을 적용한다. `score >= threshold`를 양성 예측으로 정하고 네 칸과 세 지표를 직접 센다.

```python
labels = [1, 1, 1, 1] + [0] * 16
scores = [0.91, 0.72, 0.43, 0.28,
          0.68, 0.55, 0.34, 0.31, 0.26, 0.22,
          0.18, 0.15, 0.12, 0.10, 0.08, 0.07,
          0.05, 0.04, 0.03, 0.02]

def evaluate(threshold):
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(y == 1 and pred == 1 for y, pred in zip(labels, predictions))
    fp = sum(y == 0 and pred == 1 for y, pred in zip(labels, predictions))
    fn = sum(y == 1 and pred == 0 for y, pred in zip(labels, predictions))
    tn = sum(y == 0 and pred == 0 for y, pred in zip(labels, predictions))
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(f"기준값 {threshold:.1f}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  정확도={accuracy:.3f}, 정밀도={precision:.3f}, 재현율={recall:.3f}")

for threshold in (0.5, 0.3):
    evaluate(threshold)
```

실행 출력은 다음과 같다.

```text
기준값 0.5: TP=2, FP=2, FN=2, TN=14
  정확도=0.800, 정밀도=0.500, 재현율=0.500
기준값 0.3: TP=3, FP=4, FN=1, TN=12
  정확도=0.750, 정밀도=0.429, 재현율=0.750
```

## 5. 결과 해석: 지표보다 먼저 실수의 비용을 정한다

0.3 기준은 0.5 기준보다 정확도가 0.05 낮지만 실제 양성 네 개 중 세 개를 찾는다. 대신 양성이라고 표시한 일곱 개 중 네 개가 거짓 양성이다. 어느 기준이 더 낫다고 숫자만으로 결정할 수는 없다. 거짓 음성 한 건과 거짓 양성 한 건의 결과가 문제마다 다르기 때문이다.

놓친 양성인 FN의 비용이 크면 재현율을 우선할 수 있다. 잘못된 양성 알림인 FP의 비용이 크면 정밀도를 우선할 수 있다. 그렇다고 한 지표만 보고 다른 지표를 숨겨서는 안 된다. 선택한 기준값의 혼동행렬과 정밀도·재현율을 함께 제시해야 어떤 실수를 줄이고 어떤 실수를 받아들였는지 확인할 수 있다.

기준값도 모델 학습이 끝난 뒤 테스트 결과를 보며 고르면 테스트 자료가 선택 과정에 새어 들어간다. [학습·검증·테스트 자료를 나눈 원칙]({{ '/posts/machine-learning-train-validation-test-split/' | relative_url }})에 따라 기준값과 우선 지표는 검증 자료에서 정하고, 테스트 자료는 마지막 평가에 한 번 사용해야 한다. 불균형 문제라고 해서 자료 분리 원칙이 달라지는 것은 아니다.

이번 예시는 지표 계산을 분명히 보이려고 양성 비율을 20%로 정했다. 어느 비율부터 불균형이라고 부르는 절대 경계는 없으며, 양성의 희소성뿐 아니라 표본 수와 오류 비용도 함께 봐야 한다. 정밀도·재현율은 불균형을 자동으로 해결하는 학습 방법이 아니라, 모델의 실수를 다른 관점에서 측정하는 평가 지표다.

## 6. 직접 바꿔 보는 연습

1. 기준값 목록을 `(0.7, 0.5, 0.3)`으로 바꾸고 각 혼동행렬과 지표를 표로 정리한다.
2. 네 번째 양성 점수 `0.28`을 `0.32`로 바꾸고 기준값 0.3의 재현율이 어떻게 달라지는지 계산한다.
3. 음성 점수 `0.68`을 `0.48`로 바꾸고 기준값 0.5의 정밀도와 정확도를 다시 구한다.

## 핵심 요약

- 불균형 분류에서는 다수 클래스만 예측해도 정확도가 높을 수 있으므로 혼동행렬의 TP·FP·FN·TN을 함께 본다.
- 정밀도는 양성 예측 중 실제 양성의 비율이고, 재현율은 실제 양성 중 찾아낸 비율이다.
- 기준값을 바꾸면 네 칸과 지표가 함께 바뀐다. 오류 비용에 맞춰 검증 자료에서 기준값을 정하고 테스트 자료로 마지막 평가를 한다.

## 다음 글 예고

다음 글에서는 **특성 선택과 모델 해석**을 다룬다. 불필요한 특성을 줄이는 이유와 순열 중요도가 예측 성능의 변화를 이용해 특성의 기여를 측정하는 원리를 작은 예제로 확인한다.

## 참고 자료

- Google for Developers, [Thresholds and the confusion matrix](https://developers.google.com/machine-learning/crash-course/classification/thresholding) — 분류 기준값과 TP·FP·FN·TN, 불균형 자료의 관계를 설명하는 공개 교재.
- Google for Developers, [Accuracy, recall, precision, and related metrics](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) — 정확도·정밀도·재현율의 정의와 기준값에 따른 절충을 설명하는 공개 교재.
- scikit-learn, [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) — 실제값과 예측값에서 혼동행렬을 계산하는 공식 API 문서.
- scikit-learn, [precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html) — 정밀도·재현율과 평균 방식을 정의한 공식 API 문서.
