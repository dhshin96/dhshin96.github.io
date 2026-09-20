---
title: "생존분석 입문 ⑩: 랜드마크별 동적 AUC와 위험 순위 Python 실습"
date: 2026-09-20 20:00:00 +0900
categories: [기반 지식, 학습과 모델링]
tags: [생존분석, 동적-AUC, 시간의존-ROC, 랜드마크-분석, 구분능력, Python]
description: "예측 시점마다 누적 사례와 동적 대조군을 다시 정의하고, 사례–대조군 쌍의 위험 순서로 시간에 따른 구분 능력을 계산합니다."
reading_time: 12
series: "처음 시작하는 생존분석"
level: "입문"
math: true
---

> **이번 글의 목표:** 랜드마크마다 사례와 대조군을 다시 정하고, 동적 AUC가 미래 사건 위험의 순서를 얼마나 잘 구분하는지 손계산과 Python으로 확인한다.

8명의 합성 자료에서 4시간 안의 사건을 구분한 AUC는 랜드마크 시간 2에서 0.833, 시간 4에서 0.917, 시간 6에서 0.889였다. 같은 모형을 평가해도 시간에 따라 값이 달라진다. 평가 대상과 끝점이 함께 움직이기 때문이다. **동적 AUC는 생존확률이 정답에 얼마나 가까운지가 아니라, 사건을 먼저 겪을 사람에게 더 높은 위험 점수를 주었는지 평가한다.**

## 1. 직관: 오늘의 사례와 대조군을 다시 나눈다

직전 글에서 [시간별 Brier 점수와 보정을 따로 보는 이유]({{ '/posts/survival-dynamic-brier-calibration/' | relative_url }})를 배웠다. Brier 점수는 예측확률의 오차를 재고, 보정은 예측값과 관찰 비율의 일치를 본다. 이번 질문은 다르다. “앞으로 4시간 안에 사건을 겪을 사람을 더 위험하다고 순서 매겼는가?”를 묻는다.

랜드마크를 $s$, 예측창을 $w$, 끝점을 $h=s+w$라 하자. 시간 $s$까지 사건 없이 남은 사람 중 $h$까지 사건을 겪은 사람은 **누적 사례**, $h$를 지나 사건 없이 남은 사람은 **동적 대조군**이다. $s$가 바뀌면 이미 사건을 겪은 사람은 빠지고, $h$도 이동한다. 따라서 AUC 옆에는 반드시 $(s,h)$를 함께 적어야 한다.

[랜드마크 동적 예측에서 위험집합과 예측창을 갱신하는 규칙]({{ '/posts/survival-dynamic-landmark-prediction/' | relative_url }})이 낯설다면 먼저 그 글을 복습하자. 이번 글은 그 시점별 예측에 구분 능력이라는 평가 축을 붙인다.

## 2. 기호와 수식: 맞게 정렬한 쌍의 비율

대상 $i$의 사건 시간을 $T_i$, 위험 점수를 $R_i(s,h)$라 하자. 점수가 클수록 끝점 전 사건 위험이 높다고 정한다. 랜드마크 $s$에서 사례 집합과 대조군 집합은 다음과 같다.

$$
\mathcal C(s,h)=\{i:s<T_i\le h\},\qquad
\mathcal D(s,h)=\{j:T_j>h\}
$$

검열이 없는 작은 자료에서 동적 AUC는 모든 사례–대조군 쌍 가운데 사례 점수가 더 높은 비율이다. 점수가 같으면 절반만 맞은 것으로 센다.

$$
\widehat{AUC}(s,h)=
\frac{1}{|\mathcal C||\mathcal D|}
\sum_{i\in\mathcal C}\sum_{j\in\mathcal D}
\left[I(R_i>R_j)+\frac12I(R_i=R_j)\right]
$$

따라서 0.5는 무작위 순서와 비슷하고, 1은 모든 사례가 모든 대조군보다 높은 점수를 받은 상태다. 0.8이라는 값은 사건확률이 80%라는 뜻이 아니다. 임의의 사례–대조군 한 쌍에서 사례의 위험 점수가 더 높을 확률을 표본에서 추정한 값이다.

ROC 곡선은 위험 점수의 임계값을 높은 쪽부터 낮추면서 민감도와 1-특이도가 함께 어떻게 변하는지 그린다. AUC는 그 곡선 아래 면적이므로 특정 임계값 하나의 정답률이 아니다. 같은 AUC를 가진 두 모형도 관심 있는 민감도 구간에서는 다른 결과를 낼 수 있다. 따라서 실제 분류 기준이 필요한 분석에서는 AUC와 함께 선택한 임계값의 민감도·특이도도 별도로 제시한다.

<figure class="study-figure">
  <img src="{{ '/assets/images/survival-analysis/landmark-dynamic-auc.svg' | relative_url }}" alt="랜드마크 시간 2, 4, 6에서 4시간 예측창에 따라 사례와 대조군을 다시 나누고 동적 AUC 0.833, 0.917, 0.889를 비교한 그림">
  <figcaption>그림 1. 예측창 길이는 같아도 랜드마크가 이동하면 비교 가능한 사례–대조군 쌍과 AUC가 달라진다.</figcaption>
</figure>

## 3. 시간 4에서 작은 손계산 해보기

시간 4에 사건 없이 남은 B부터 H까지 7명을 시간 8까지 평가한다. B, D, E는 시간 5, 7, 8에 사건을 겪어 사례가 되고, C, F, G, H는 시간 8을 지나므로 대조군이 된다. 위험 점수는 B 0.70, D 0.80, E 0.50, C 0.40, F 0.20, G 0.60, H 0.30이다.

B는 대조군 네 명보다 모두 높아 4쌍을 맞힌다. D도 4쌍을 맞힌다. E는 C, F, H보다 높지만 G보다 낮아 3쌍만 맞힌다. 전체 12쌍 중 11쌍의 순서가 맞았다.

$$
\widehat{AUC}(4,8)=\frac{4+4+3}{3\times4}
=\frac{11}{12}=0.917
$$

여기서 모든 점수에 10을 곱해도 순서는 같으므로 AUC는 변하지 않는다. 반대로 확률값이 실제 비율과 잘 맞더라도 순서가 자주 뒤집히면 AUC는 낮을 수 있다. 이 때문에 AUC만으로 확률 예측의 정확성이나 보정을 판단할 수 없다.

## 4. Python으로 세 랜드마크의 AUC 계산하기

아래 코드는 검열이 없는 설명용 자료에서 쌍별 비교를 그대로 구현한다. 별도 라이브러리가 필요하지 않으며, 동점은 0.5로 처리한다.

```python
event_time = {
    "A": 3, "B": 5, "C": 12, "D": 7,
    "E": 8, "F": 12, "G": 10, "H": 12,
}

risk_scores = {
    2: {"A": .90, "B": .55, "C": .20, "D": .70,
        "E": .60, "F": .10, "G": .40, "H": .30},
    4: {"B": .70, "C": .40, "D": .80, "E": .50,
        "F": .20, "G": .60, "H": .30},
    6: {"C": .50, "D": .80, "E": .70,
        "F": .20, "G": .40, "H": .30},
}


def pairwise_dynamic_auc(landmark, scores, window=4):
    horizon = landmark + window
    cases = [person for person in scores
             if landmark < event_time[person] <= horizon]
    controls = [person for person in scores
                if event_time[person] > horizon]

    credit = 0.0
    for case in cases:
        for control in controls:
            credit += scores[case] > scores[control]
            credit += 0.5 * (scores[case] == scores[control])

    pairs = len(cases) * len(controls)
    return horizon, cases, controls, credit, credit / pairs


for s, scores in risk_scores.items():
    h, cases, controls, correct, auc = pairwise_dynamic_auc(s, scores)
    print(f"s={s}, h={h}, cases={cases}, controls={controls}")
    print(f"ordered={correct:.0f}/{len(cases) * len(controls)}, AUC={auc:.3f}")
```

핵심 출력은 다음과 같다.

```text
s=2, h=6, cases=['A', 'B'], controls=['C', 'D', 'E', 'F', 'G', 'H']
ordered=10/12, AUC=0.833
s=4, h=8, cases=['B', 'D', 'E'], controls=['C', 'F', 'G', 'H']
ordered=11/12, AUC=0.917
s=6, h=10, cases=['D', 'E', 'G'], controls=['C', 'F', 'H']
ordered=8/9, AUC=0.889
```

시간 4의 값이 가장 높지만 세 값의 표본 수와 사례 구성이 다르다. 0.917과 0.889의 차이를 곧바로 성능 하락으로 해석하면 안 된다. 작은 표본에서는 한 쌍의 순서만 바뀌어도 값이 크게 움직이므로 실제 분석에서는 독립 검증 자료와 신뢰구간이 필요하다.

## 5. 검열이 있으면 IPCW가 필요한 이유

끝점 전에 검열된 사람은 사례인지 대조군인지 알 수 없다. 이들을 모두 빼면 오래 관찰될 가능성이 큰 사람만 평가에 남아 구분 능력이 치우칠 수 있다. 실제 누적/동적 AUC는 검열 생존함수의 역수인 IPCW로 관찰된 사례의 대표성을 보충한다. 검열분포는 보통 훈련 자료에서 Kaplan–Meier 방식으로 추정하며, 평가 시간은 검열 생존확률을 안정적으로 추정할 수 있는 추적 범위 안에 두어야 한다.

[Kaplan–Meier 곡선의 위험집합과 검열 처리]({{ '/posts/km-curve-logrank/' | relative_url }})가 이 가중치의 출발점이다. 다만 위의 순수 Python 코드는 원리를 보이기 위해 검열을 넣지 않았다. 검열 자료에 그대로 적용해 완전 사례만 비교하지 말고, 검증된 IPCW 구현을 사용해야 한다.

또한 라이브러리마다 입력 방향을 확인한다. `scikit-survival`의 `cumulative_dynamic_auc`는 값이 클수록 사건 위험이 높은 **위험 점수**를 받는다. 생존확률을 그대로 넣으면 순서가 반대로 되어 결과를 잘못 읽을 수 있다. 시간별 예측을 평가할 때는 대상별·시간별 점수 행렬과 평가 시점의 열도 일치해야 한다.

## 6. Brier 점수·보정과 함께 읽기

동적 AUC가 높다는 것은 위험 순위의 구분이 좋다는 뜻이다. 하지만 위험 점수 0.9가 실제 사건확률 90%인지 말해 주지 않는다. 반대로 모든 사람의 확률을 비슷하게 압축하면 보정은 괜찮아 보여도 사례와 대조군을 나누는 순위는 약할 수 있다.

따라서 같은 $(s,h)$와 같은 검증 표본에서 세 질문을 나눈다. 동적 AUC로 순위를, Brier 점수로 전체 확률 오차를, 보정 그림으로 예측값과 관찰값의 일치를 본다. 지표마다 평가 대상과 검열 처리도 함께 기록해야 숫자를 서로 연결할 수 있다.

<div class="related-reading">
  <strong>이어 읽기</strong>
  <p>랜드마크 동적 예측 글로 평가 대상을 정하고, 시간별 Brier 점수 글로 확률 오차와 보정을 비교한 뒤, Kaplan–Meier 글로 IPCW의 검열분포 추정을 연결해 보세요.</p>
</div>

## 직접 바꿔 보는 연습

1. 시간 4에서 E의 위험 점수를 0.50에서 0.65로 바꾸고 AUC를 다시 계산한다.
2. 시간 6에서 G와 C의 점수를 같게 만들어 동점 0.5 처리가 결과에 미치는 영향을 확인한다.
3. 예측창을 3으로 줄였을 때 각 랜드마크의 사례·대조군과 비교 가능한 쌍 수를 먼저 손으로 적는다.

## 핵심 요약

1. 누적/동적 AUC는 끝점까지 사건을 겪은 누적 사례와 끝점을 지난 동적 대조군의 위험 순서를 비교한다.
2. 랜드마크와 끝점이 바뀌면 평가 대상과 사례 정의도 바뀌므로 AUC는 반드시 $(s,h)$와 함께 보고한다.
3. 끝점 전 검열이 있으면 단순 제외 대신 검열분포를 이용한 IPCW가 필요하다.
4. 동적 AUC는 구분 능력 지표이므로 Brier 점수와 보정 평가를 대신하지 않는다.

## 다음 글

다음 글에서는 **부트스트랩으로 동적 AUC의 신뢰구간을 만들고 낙관 편향을 확인하는 법**을 배운다.

## 참고 자료

- Heagerty, Lumley, Pepe (2000), [*Time-Dependent ROC Curves for Censored Survival Data and a Diagnostic Marker*](https://doi.org/10.1111/j.0006-341X.2000.00337.x), 검열 자료의 시간의존 ROC와 AUC 추정 틀을 제시한다.
- [scikit-survival `cumulative_dynamic_auc` 공식 문서](https://scikit-survival.readthedocs.io/en/latest/api/generated/sksurv.metrics.cumulative_dynamic_auc.html), 누적 사례·동적 대조군, IPCW 정의와 입력 조건을 설명한다.
- [scikit-survival 생존모형 평가 가이드](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html), C-index·동적 AUC·Brier 점수가 답하는 질문을 비교한다.
