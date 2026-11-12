---
title: "생성형 AI RAG 검색 입문: 문서를 찾아 답변 근거로 붙이는 과정"
date: 2026-11-08 18:00:00 +0900
categories: [학습과 모델링, 지능형 시스템]
tags: [생성형-AI, RAG, 검색-증강-생성, 코사인-유사도, Python]
description: "RAG가 질문과 가까운 문서를 검색해 생성 입력에 붙이는 원리를 배웁니다. 순수 Python 실습으로 문서 세 개의 유사도를 계산하고 0.707점인 근거를 선택합니다."
reading_time: 10
series: "처음 시작하는 생성형 AI"
level: "입문"
math: true
---

> **이번 글의 목표:** 질문과 문서를 같은 단어 벡터로 바꾸고, 가장 가까운 문서 하나를 골라 생성 입력의 근거로 붙인다.

먼저 결과부터 보자. 질문 `RAG 질문 문서 검색`과 문서 세 개의 코사인 유사도는 `0.000`, `0.447`, `0.707`이었다. 가장 높은 `D3`를 고르자 답변 입력에 “RAG 질문 관련 문서 검색 생성 입력 추가”라는 근거가 들어갔다. 이번 실습에서 RAG는 모델을 다시 학습시키는 기술이 아니라 **검색 결과를 생성 전에 건네는 흐름**이다.

앞 글에서는 [생성 결과의 모양을 JSON 스키마로 검사하는 방법]({{ '/posts/generative-ai-structured-output-json-schema/' | relative_url }})을 배웠다. 구조화 출력이 답의 형식을 통제한다면, RAG는 답을 만들 때 참고할 바깥 근거를 고른다.

## 1. 직관: 먼저 찾고, 찾은 내용을 보고 답한다

책상 위에 짧은 메모가 세 장 있다고 생각해 보자. 질문을 받자마자 답을 쓰지 않고, 먼저 질문과 가장 관련 있는 메모를 찾는다. 그 메모와 질문을 한 입력에 묶은 뒤 답을 만든다. 이때 앞부분이 **검색(retrieval)**, 뒷부분이 **생성(generation)**이다.

Lewis 등이 2020년에 발표한 원 논문은 RAG를 사전 학습된 생성 모델의 매개변수 기억과 외부의 비매개변수 기억을 결합하는 방식으로 설명한다. 원 논문은 위키백과의 밀집 벡터 색인과 신경망 검색기를 사용하지만, 핵심 연결은 단순하다.

$$
q \rightarrow \text{관련 문서 검색} \rightarrow (q, d^*) \rightarrow \text{답변 생성}
$$

여기서 $q$는 질문, $d^*$는 검색 점수가 가장 높은 문서다. 이번에는 외부 API와 언어 모델 없이 단어 빈도와 모의 생성 함수로 이 연결만 재현한다.

<figure class="study-figure">
  <img src="{{ '/assets/images/generative-ai/rag-retrieval-grounding-flow.svg' | relative_url }}" alt="질문을 단어 벡터로 바꾸고 문서 세 개와 코사인 유사도를 계산한 뒤, 최고 점수 0.707인 D3를 선택해 질문과 함께 생성 입력에 붙이는 흐름.">
  <figcaption>그림 1. RAG의 최소 흐름. 검색 점수가 가장 높은 문서만 근거 칸에 넣고, 질문과 함께 생성 단계로 보낸다.</figcaption>
</figure>

## 2. 기호와 수식: 질문과 문서의 방향을 비교한다

질문 벡터를 $\mathbf{q}$, 문서 벡터를 $\mathbf{d}_i$라고 하자. 각 칸에는 해당 단어가 나타난 횟수를 넣는다. 두 벡터의 코사인 유사도는 다음과 같다.

$$
s_i=\cos(\mathbf{q},\mathbf{d}_i)
=\frac{\mathbf{q}\cdot\mathbf{d}_i}
{\lVert\mathbf{q}\rVert\lVert\mathbf{d}_i\rVert}
$$

검색기는 $s_i$가 가장 큰 문서를 선택한다.

$$
d^*=\underset{d_i}{\arg\max}\;s_i
$$

코사인 유사도는 길이 자체보다 벡터의 방향을 비교한다. Stanford의 정보 검색 교재도 문서 길이 차이의 영향을 줄이기 위해 벡터를 길이로 정규화한 코사인 유사도를 설명한다. 다만 이번 단어 빈도 벡터는 어순을 버리고, 표현이 다른 동의어도 자동으로 가깝게 보지 못한다.

## 3. 작은 손계산: D3의 점수는 왜 0.707인가

질문에는 `RAG`, `질문`, `문서`, `검색` 네 단어가 한 번씩 있다. 따라서 $\lVert\mathbf{q}\rVert=\sqrt{4}=2$다. D3에는 여덟 단어가 한 번씩 있으므로 $\lVert\mathbf{d}_3\rVert=\sqrt{8}$이다. 두 문장이 공유하는 네 단어의 내적은 4다.

$$
\cos(\mathbf{q},\mathbf{d}_3)
=\frac{4}{2\sqrt{8}}
\approx 0.707
$$

D2는 `문서`, `검색` 두 단어만 공유한다. 다섯 단어로 이루어진 D2의 점수는 $2/(2\sqrt{5})\approx0.447$이다. D1은 겹치는 단어가 없어 0이다. 그러므로 검색 순위는 D3, D2, D1이다.

## 4. 실행 가능한 Python 실습

아래 코드는 표준 라이브러리만 사용한다. 공백으로 나눈 단어의 빈도 벡터를 만들고, 최고 점수 문서를 근거로 붙인 뒤 모의 생성 함수가 한 문장 답을 반환한다.

```python
from collections import Counter
from math import sqrt

documents = {
    "D1": "토큰화 문장 토큰 분리",
    "D2": "임베딩 문서 벡터 유사도 검색",
    "D3": "RAG 질문 관련 문서 검색 생성 입력 추가",
}
query = "RAG 질문 문서 검색"


def cosine(left, right):
    left_counts = Counter(left.split())
    right_counts = Counter(right.split())
    dot = sum(left_counts[word] * right_counts[word] for word in left_counts)
    left_norm = sqrt(sum(value ** 2 for value in left_counts.values()))
    right_norm = sqrt(sum(value ** 2 for value in right_counts.values()))
    return dot / (left_norm * right_norm)


scores = {name: cosine(query, text) for name, text in documents.items()}
for name in sorted(scores):
    print(f"{name}: {scores[name]:.3f}")

best_id = max(scores, key=scores.get)
context = documents[best_id]
prompt = f"[근거] {context}\n[질문] RAG는 무엇을 검색하나요?"


def mock_generate(grounded_prompt):
    assert "RAG 질문 관련 문서 검색 생성 입력 추가" in grounded_prompt
    return "RAG는 질문과 관련된 문서를 검색해 생성 입력에 추가합니다."


print(f"선택: {best_id}")
print(prompt)
print(f"모의 생성: {mock_generate(prompt)}")
```

실행 출력은 다음과 같다.

```text
D1: 0.000
D2: 0.447
D3: 0.707
선택: D3
[근거] RAG 질문 관련 문서 검색 생성 입력 추가
[질문] RAG는 무엇을 검색하나요?
모의 생성: RAG는 질문과 관련된 문서를 검색해 생성 입력에 추가합니다.
```

## 5. 결과 해석: 검색 성공과 답변 정확성은 다르다

`D3`가 선택된 것은 우리가 만든 세 문서와 네 단어 질문 안에서 가장 가깝다는 뜻이다. 실제 세계에서 가장 정확한 문서라는 보장은 아니다. 문서 모음에 정답 근거가 없거나 질문 표현이 크게 달라지면 검색부터 실패할 수 있다. 좋은 근거를 찾았더라도 생성기가 근거를 잘못 읽거나 근거 밖의 내용을 덧붙일 수도 있다.

이전 글의 [임베딩과 코사인 유사도 실습]({{ '/posts/generative-ai-embedding-cosine-similarity/' | relative_url }})을 다시 보면 단어 빈도 대신 의미 벡터를 쓰는 이유를 연결해 이해할 수 있다. 실제 RAG는 문서를 적당한 조각으로 나누고, 질문과 문서 조각을 벡터로 표현해 가까운 조각 여러 개를 찾는 구성이 흔하다. 검색된 근거의 관련성과 최종 답변의 충실성은 서로 나눠 평가해야 한다.

## 6. 직접 바꿔 보는 연습

1. 질문에서 `RAG`를 빼고 세 점수와 순위가 어떻게 달라지는지 확인한다.
2. D2에 `질문`과 `RAG`를 추가해 D2와 D3의 점수를 다시 계산한다.
3. 최고 점수가 `0.5`보다 낮으면 “근거 부족”을 반환하도록 검색 단계에 문턱값을 추가한다.

## 핵심 요약

- RAG는 질문과 관련된 외부 문서를 먼저 찾고, 그 문서를 질문과 함께 생성 입력에 붙인다.
- 단어 빈도 코사인 유사도 실습에서 D3가 `0.707`으로 선택됐고, D2와 D1은 각각 `0.447`, `0.000`이었다.
- 검색 점수가 높다는 사실은 답변의 정확성을 보장하지 않는다. 검색과 생성을 따로 확인해야 한다.
- 이번 코드는 RAG의 연결 원리를 보여 주는 최소 모형이며, 실제 신경망 검색기나 언어 모델의 출력이 아니다.

## 다음 글 예고

다음 글에서는 [생성 품질을 검색 성공과 답변 충실성으로 나눠 평가한다]({{ '/posts/generative-ai-rag-quality-evaluation/' | relative_url }}). 작은 평가표와 결정적 Python 예제로 두 실패 지점을 따로 측정한다.

## 참고 자료

1. Patrick Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) — 매개변수 기억과 외부 비매개변수 기억을 결합한 RAG 원 논문(2026-11-08 확인).
2. Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze, [Introduction to Information Retrieval, Chapter 6](https://nlp.stanford.edu/IR-book/pdf/06vect.pdf) — 벡터 공간 모델과 코사인 유사도의 길이 정규화 설명(2026-11-08 확인).
