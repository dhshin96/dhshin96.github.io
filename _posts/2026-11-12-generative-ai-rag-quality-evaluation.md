---
title: "생성형 AI 품질 평가 입문: 검색 성공과 답변 충실성 나눠 재기"
date: 2026-11-12 20:00:00 +0900
categories: [학습과 모델링, 지능형 시스템]
tags: [생성형-AI, RAG, 품질-평가, 충실성, Python]
description: "RAG의 생성 품질을 검색 적중률과 답변 충실성으로 나눠 평가합니다. 순수 Python 실습으로 두 지표는 각각 75%, 전체 통과율은 50%가 되는 과정을 확인합니다."
reading_time: 10
series: "처음 시작하는 생성형 AI"
level: "입문"
math: true
---

> **이번 글의 목표:** RAG 결과를 검색 단계와 생성 단계로 나누고, 검색 적중률과 답변 충실성을 작은 평가표에서 직접 계산한다.

먼저 결과부터 보자. 네 질문을 평가했더니 검색 적중률은 `75%`, 근거가 뒷받침한 답변 주장 비율도 `75%`였다. 그러나 두 조건을 한꺼번에 통과한 질문은 둘뿐이라 전체 통과율은 `50%`였다. 같은 75% 두 개만 보고 시스템 전체도 75%라고 결론 내리면 실패 원인을 놓친다.

앞 글의 [RAG 검색과 근거 결합 과정]({{ '/posts/generative-ai-rag-retrieval-grounding/' | relative_url }})을 먼저 보면 질문, 검색 문서, 답변이 어떻게 이어지는지 복습할 수 있다. 이번에는 그 흐름의 각 연결을 평가표로 바꾼다.

## 1. 직관: 답이 틀렸다면 어느 단계부터 틀렸는가

RAG의 답이 좋으려면 두 일이 차례로 성공해야 한다. 검색기는 질문에 필요한 문서를 찾아야 하고, 생성기는 찾은 문서가 말하는 범위 안에서 답해야 한다. 잘못된 문서를 찾았는데 문장만 자연스러울 수도 있고, 맞는 문서를 찾고도 답에 근거 없는 내용을 덧붙일 수도 있다.

RAGAS 논문도 RAG 평가를 하나의 점수로만 보지 않는다. 관련 있고 집중된 문맥을 검색하는 능력, 그 문맥을 충실하게 사용하는 능력, 생성 자체의 품질을 서로 다른 차원으로 설명한다. 이번 입문 실습은 그중 두 연결만 사람이 미리 붙인 참·거짓 표로 단순화한다.

<figure class="study-figure">
  <img src="{{ '/assets/images/generative-ai/rag-quality-evaluation-two-stages.svg' | relative_url }}" alt="질문 네 개에서 정답 문서와 검색 문서가 같은지 먼저 검사해 검색 적중률 75%를 구하고, 답변 주장 여덟 개 중 근거가 뒷받침하는 여섯 개를 세어 충실성 75%를 구한 뒤 두 조건을 모두 만족한 전체 통과율 50%를 계산하는 흐름.">
  <figcaption>그림 1. 검색과 생성을 따로 재면 같은 75% 안에서도 실패한 위치가 드러난다.</figcaption>
</figure>

## 2. 기호와 수식: 적중과 충실성을 각각 센다

질문이 $N$개이고, 질문 $i$의 정답 문서를 $g_i$, 검색기가 첫 번째로 고른 문서를 $r_i$라고 하자. 두 문서가 같으면 적중값 $h_i$는 1, 다르면 0이다.

$$
h_i=\begin{cases}1 & g_i=r_i\\0 & g_i\ne r_i\end{cases},\qquad
\text{검색 적중률}=\frac{\sum_{i=1}^{N}h_i}{N}
$$

Stanford 정보검색 교재는 평가에 문서 모음, 질문 모음, 질문과 문서 쌍의 관련성 판단이 필요하다고 설명한다. 순위가 있는 검색에서는 상위 $k$개를 기준으로 평가할 수 있다. 여기서는 가장 단순한 $k=1$만 보고, 정답 문서 ID를 관련성 판단으로 사용한다.

답변은 검증 가능한 짧은 주장으로 나눈다. 전체 주장 수를 $M$, 검색된 근거가 뒷받침하는 주장에 붙인 값 $s_j$를 1 또는 0이라고 하자.

$$
\text{답변 충실성}=\frac{\sum_{j=1}^{M}s_j}{M}
$$

이 값은 답변이 검색된 근거를 따랐는지만 잰다. 검색된 근거 자체가 정답 문서인지, 질문에 충분히 답했는지까지 자동으로 보장하지 않는다.

## 3. 작은 손계산: 75%와 75%가 왜 50%가 되는가

네 질문 중 Q1, Q3, Q4는 정답 문서를 첫 번째로 찾았고 Q2만 놓쳤다. 검색 적중률은 $3/4=0.75$다. 각 답변을 주장 두 개씩, 모두 여덟 개로 나누자. 근거가 뒷받침한 주장은 Q1의 2개, Q2의 1개, Q3의 2개, Q4의 1개이므로 충실성은 $6/8=0.75$다.

전체 통과는 검색이 적중하고 두 주장도 모두 근거에 있을 때만 1로 둔다. Q1과 Q3만 통과하므로 $2/4=0.50$이다. Q2는 검색부터 실패했고, Q4는 맞는 문서를 찾았지만 주장 하나를 근거 밖에서 더했다. 같은 실패로 묶지 않아야 고칠 단계가 보인다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 API나 패키지를 쓰지 않는다. 실제 모델을 자동 채점하는 대신, 사람이 검토해 만든 작은 참·거짓 평가표를 집계한다.

```python
cases = [
    {"id": "Q1", "gold": "D1", "retrieved": "D1", "claims": [True, True]},
    {"id": "Q2", "gold": "D2", "retrieved": "D3", "claims": [False, True]},
    {"id": "Q3", "gold": "D3", "retrieved": "D3", "claims": [True, True]},
    {"id": "Q4", "gold": "D4", "retrieved": "D4", "claims": [True, False]},
]

retrieval_hits = 0
supported_claims = 0
total_claims = 0
end_to_end_passes = 0

for case in cases:
    retrieval_ok = case["gold"] == case["retrieved"]
    supported = sum(case["claims"])
    all_supported = supported == len(case["claims"])

    retrieval_hits += retrieval_ok
    supported_claims += supported
    total_claims += len(case["claims"])
    end_to_end_passes += retrieval_ok and all_supported

    print(
        f'{case["id"]}: retrieval={int(retrieval_ok)}, '
        f'supported={supported}/{len(case["claims"])}, '
        f'pass={int(retrieval_ok and all_supported)}'
    )

print(f"retrieval hit rate: {retrieval_hits / len(cases):.1%}")
print(f"claim faithfulness: {supported_claims / total_claims:.1%}")
print(f"end-to-end pass rate: {end_to_end_passes / len(cases):.1%}")
```

실행 출력은 다음과 같다.

```text
Q1: retrieval=1, supported=2/2, pass=1
Q2: retrieval=0, supported=1/2, pass=0
Q3: retrieval=1, supported=2/2, pass=1
Q4: retrieval=1, supported=1/2, pass=0
retrieval hit rate: 75.0%
claim faithfulness: 75.0%
end-to-end pass rate: 50.0%
```

## 5. 결과 해석: 지표는 원인표와 함께 읽는다

검색 적중률만 보면 Q4의 문제는 보이지 않는다. 충실성만 보면 Q2가 애초에 엉뚱한 문서를 찾았다는 사실이 흐려진다. 행별 결과를 함께 남기면 Q2는 검색 데이터·표현·순위 쪽을, Q4는 생성 지시나 근거 밖 주장 쪽을 먼저 살펴볼 수 있다.

이 실습의 참·거짓 값은 사람이 미리 판정했다. 긴 답변에서는 주장을 어떻게 나눌지, 여러 문서가 한 주장을 함께 뒷받침하는지에 따라 판정이 달라질 수 있다. RAGAS는 언어 모델을 이용해 답변을 더 작은 주장으로 나누고 문맥에서 추론 가능한지 평가하는 자동화 방식을 제안하지만, 자동 평가 결과도 정의와 표본을 확인해야 한다.

평가 결과를 기계가 읽는 일정한 모양으로 저장하려면 [JSON 스키마로 필드와 자료형을 검사한 글]({{ '/posts/generative-ai-structured-output-json-schema/' | relative_url }})이 도움이 된다. 질문 ID, 정답 문서, 검색 문서, 주장별 판정을 고정 필드로 만들면 집계 전에 누락을 잡을 수 있다.

## 6. 직접 바꿔 보는 연습

1. Q4의 두 번째 주장을 `True`로 바꾸고 충실성과 전체 통과율을 다시 계산한다.
2. 검색기가 문서 두 개를 반환한다고 가정하고, 정답 문서가 둘 중 하나에 있으면 적중하는 `hit@2`를 만든다.
3. 질문별 주장 수가 다를 때 전체 주장 기준 충실성과 질문별 충실성 평균이 어떻게 달라지는지 비교한다.

## 핵심 요약

- RAG 평가는 검색기가 필요한 문서를 찾았는지와 답변이 검색 근거를 따랐는지를 나눠 봐야 한다.
- 네 질문의 검색 적중률과 여덟 주장의 충실성은 각각 `75%`였지만, 두 조건을 모두 만족한 전체 통과율은 `50%`였다.
- 지표 하나만으로 실패 원인을 찾을 수 없다. 질문별 검색 판정과 주장별 근거 판정을 함께 남겨야 한다.
- 이번 코드는 사람이 만든 결정적 평가표를 집계한 최소 예제이며, 실제 모델 판정을 흉내 낸 출력이 아니다.

## 다음 글 예고

다음 글에서는 **파인튜닝의 선택 기준**을 다룬다. 프롬프트, RAG, 파인튜닝이 각각 무엇을 바꾸는지 비교하고 작은 의사결정 표로 선택 순서를 정한다.

## 참고 자료

1. Shahul Es et al., [RAGAs: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/) — 검색 문맥, 답변 충실성, 생성 품질을 나눠 평가하는 RAG 평가 틀(2026-11-12 확인).
2. Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze, [Introduction to Information Retrieval, Chapter 8](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf) — 테스트 모음, 관련성 판정, precision·recall과 상위 $k$ 검색 평가의 기초(2026-11-12 확인).
