---
title: "AI 에이전트 검색 결합 입문: 근거 조각을 고르는 Python 실습"
date: 2026-11-15 20:00:00 +0900
categories: [지능형 시스템, 개발과 제품화]
tags: [AI-에이전트, 검색-결합, RAG, 근거, Python]
description: "AI 에이전트가 질문과 관련된 문서 조각을 검색해 답변 입력에 결합하는 구조를 배웁니다. 순수 Python으로 후보 3개를 점수화하고 근거 1개만 선택합니다."
reading_time: 11
series: "처음 시작하는 AI 에이전트 개발"
level: "입문"
math: true
---

> **이번 글의 목표:** 질문과 관련된 **근거 조각을 검색**하고, 기준을 통과한 결과만 답변 입력에 **결합**하는 최소 구조를 구현한다.

먼저 실행 결과를 보자. 질문 `환불 기간`과 문서 조각 세 개를 비교하면 `refund-policy`가 0.707점으로 1위다. 임계값 0.5를 넘은 이 조각 한 개만 컨텍스트가 되고, 모의 답변은 근거 ID를 함께 표시한다.

```text
query=환불 기간
rank1 id=refund-policy score=0.707
rank2 id=refund-method score=0.408
rank3 id=shipping score=0.000
selected=1
context=[refund-policy] 환불 신청 기간은 결제일로부터 7일입니다.
mock_answer=근거에 따르면 환불 신청 기간은 결제일로부터 7일입니다. [refund-policy]
```

앞 글에서 [작업 상태와 장기 기억을 나누고 필요한 기억만 회수하는 방법]({{ '/posts/ai-agent-memory-working-long-term/' | relative_url }})을 배웠다. 장기 기억은 에이전트가 실행을 넘어 보존한 정보다. 이번 글의 검색 대상은 사용자의 기억이 아니라 정책 문서처럼 별도로 관리되는 외부 지식이다.

## 1. 직관: 답을 외우지 말고 근거를 찾아 붙인다

검색 결합은 질문을 받자마자 답을 쓰는 대신 중간에 두 단계를 둔다. 먼저 검색기가 문서 저장소에서 관련된 조각을 고른다. 다음으로 에이전트가 질문과 선택된 조각을 하나의 입력으로 묶어 답변 생성기에 전달한다. 보통 이를 검색 증강 생성, 즉 RAG(Retrieval-Augmented Generation)라고 부른다.

예를 들어 정책 문서가 바뀌어 환불 기간이 수정되었다고 하자. 모델의 내부 지식만 기대하면 어느 시점의 규칙인지 확인하기 어렵다. 검색 결과에 문서 ID와 실제 문장을 함께 남기면, 답변이 어떤 근거를 사용했는지 다시 검사할 수 있다. 다만 검색된 문장이 자동으로 참이 되는 것은 아니다. 문서의 최신성, 접근 권한, 출처도 별도로 검증해야 한다.

<figure class="study-figure">
  <img src="{{ '/assets/images/ai-agents/retrieval-augmentation-flow.svg' | relative_url }}" alt="질문을 문서 조각 세 개와 비교해 점수 임계값을 넘은 환불 정책 조각 하나만 선택하고, 질문과 근거를 답변 입력으로 결합하는 흐름">
  <figcaption>그림 1. 검색 결합은 검색 결과 전체가 아니라 기준을 통과한 근거만 답변 입력에 넣는다.</figcaption>
</figure>

[도구 호출에서 함수 이름과 인수를 계약으로 연결했던 구조]({{ '/posts/ai-agent-tool-calling-json-schema/' | relative_url }})를 떠올리면 검색도 하나의 도구로 볼 수 있다. 에이전트는 `search(query)`를 호출하고, 반환된 조각과 점수를 관찰한 뒤 답변을 계속한다. 차이는 검색 도구의 결과가 최종 답이 아니라 답을 만들 근거라는 점이다.

## 2. 기호와 수식: 질문과 문서의 겹침을 잰다

질문의 핵심어 집합을 $Q$, 문서 조각 $d$의 핵심어 집합을 $K_d$라고 하자. 이번 실습은 두 집합을 0과 1로 표현한 뒤 코사인 형태의 점수를 사용한다.

$$
S(Q,d)=\frac{|Q\cap K_d|}{\sqrt{|Q|\,|K_d|}}
$$

겹치는 핵심어가 많을수록 분자가 커진다. 문서에 핵심어가 지나치게 많으면 분모도 커져, 질문과 직접 맞는 짧은 조각이 유리해진다. $S\ge 0.5$인 조각만 선택하고 점수가 같으면 문서 ID의 오름차순으로 순서를 고정한다.

이 점수는 원리를 손으로 검산하기 위한 단순 모형이다. 실제 의미 검색은 문장을 임베딩 벡터로 바꿔 표현이 달라도 의미가 가까운 조각을 찾을 수 있다. 따라서 이 예제의 숫자를 실제 검색 품질의 기준으로 일반화하면 안 된다.

## 3. 작은 손계산: 후보 세 개에서 하나를 고른다

질문의 핵심어는 $Q=\{환불, 기간\}$이다. `refund-policy`의 핵심어는 $\{환불, 기간, 결제일, 7일\}$이므로 두 개가 겹친다.

$$
S(Q,d_1)=\frac{2}{\sqrt{2\times4}}\approx0.707
$$

`refund-method`는 `환불` 하나만 겹쳐 $1/\sqrt{2\times3}\approx0.408$이고, `shipping`은 겹침이 없어 0이다.

| 문서 조각 | 겹친 핵심어 | 점수 | 0.5 이상 |
| --- | ---: | ---: | --- |
| refund-policy | 2 | 0.707 | 선택 |
| refund-method | 1 | 0.408 | 제외 |
| shipping | 0 | 0.000 | 제외 |

상위 한 개를 무조건 쓰는 대신 임계값도 적용한 이유는 관련 없는 질문에 억지 근거를 붙이지 않기 위해서다. 모든 점수가 0.5 미만이면 `selected=0`으로 두고, 근거를 찾지 못했다고 처리해야 한다.

## 4. 실행 가능한 Python 실습

아래 코드는 표준 라이브러리만 사용한다. 실제 언어 모델은 호출하지 않는다. 검색, 임계값 선택, 컨텍스트 조립이라는 경계를 결정적으로 재현한 뒤 선택된 문장을 이용해 모의 답변을 만든다.

```python
from math import sqrt

QUERY = ("환불", "기간")
DOCUMENTS = [
    {
        "id": "refund-policy",
        "keywords": {"환불", "기간", "결제일", "7일"},
        "text": "환불 신청 기간은 결제일로부터 7일입니다.",
    },
    {
        "id": "refund-method",
        "keywords": {"환불", "수단", "결제"},
        "text": "환불 금액은 원래 결제 수단으로 지급됩니다.",
    },
    {
        "id": "shipping",
        "keywords": {"배송", "조회", "운송장"},
        "text": "배송 상태는 운송장 번호로 조회합니다.",
    },
]


def score(query, keywords):
    query_terms = set(query)
    return len(query_terms & keywords) / sqrt(len(query_terms) * len(keywords))


def retrieve(query, documents, threshold=0.5):
    ranked = [
        (score(query, item["keywords"]), item) for item in documents
    ]
    ranked.sort(key=lambda row: (-row[0], row[1]["id"]))
    selected = [row for row in ranked if row[0] >= threshold]
    return ranked, selected


ranked, selected = retrieve(QUERY, DOCUMENTS)
print("query=" + " ".join(QUERY))
for index, (value, item) in enumerate(ranked, start=1):
    print(f"rank{index} id={item['id']} score={value:.3f}")
print(f"selected={len(selected)}")

context = "\n".join(
    f"[{item['id']}] {item['text']}" for _, item in selected
)
print(f"context={context}")
if selected:
    source = selected[0][1]
    print(f"mock_answer=근거에 따르면 {source['text']} [{source['id']}]")
else:
    print("mock_answer=확인할 근거를 찾지 못했습니다.")
```

## 5. 결과 해석: 검색과 답변을 따로 검사한다

출력에서 `rank1`은 검색 단계의 결과이고 `context`는 결합 단계의 결과다. 두 단계를 분리하면 실패 원인도 나눠 볼 수 있다. 올바른 문서가 순위에 없다면 검색어나 색인, 점수식을 점검한다. 올바른 문서가 검색됐는데 답변이 다른 내용을 말한다면 답변 입력과 생성 규칙을 점검한다.

근거 ID를 답변에 남긴 것도 중요하다. 사용자는 원문을 확인할 수 있고, 개발자는 어떤 조각이 선택됐는지 테스트할 수 있다. 실제 서비스에서는 ID뿐 아니라 제목, 원문 위치, 갱신일을 함께 보관해야 한다. 접근 권한을 확인한 뒤 검색해야 하며, 검색 결과 안의 지시문을 시스템 명령처럼 실행해서도 안 된다.

2026년 11월 15일 현재 OpenAI 공식 Retrieval 문서는 `vector_stores`에 파일을 추가하면 조각화·임베딩·색인이 이뤄지고, `search`가 관련 조각과 유사도 점수, 원본 파일을 반환한다고 설명한다. OpenAI Agents SDK에는 Vector Store를 검색하는 `FileSearchTool`이 있으며 `vector_store_ids`, `max_num_results`, `filters`, `ranking_options` 등을 받는다. 이 이름과 지원 범위는 바뀔 수 있으므로 구현 시점의 공식 문서를 다시 확인해야 한다.

## 6. 직접 바꿔 보는 연습

먼저 `threshold`를 0.4로 낮춰 보자. `refund-method`까지 포함되어 `selected=2`가 된다. 컨텍스트가 늘면 보완 근거를 얻을 수 있지만, 질문과 덜 관련된 문장도 함께 들어간다. 반대로 0.8로 올리면 선택 결과가 0건이 된다.

다음에는 질문을 `{"배송", "조회"}`로 바꿔 `shipping`이 1위가 되는지 확인하자. 마지막으로 문서에 `updated_at`과 `allowed_users`를 추가하고, 점수를 계산하기 전에 최신성과 권한 조건을 통과한 문서만 남겨 보자. 검색 품질과 접근 통제는 서로 다른 검사라는 점을 코드 구조에도 드러내는 것이 핵심이다.

## 핵심 요약

- 검색 결합은 질문과 관련된 외부 근거를 찾고, 선택된 조각을 답변 입력에 붙이는 구조다.
- 검색 순위와 임계값을 함께 사용하면 관련 없는 최고 점수를 억지 근거로 쓰지 않을 수 있다.
- 문서 ID와 위치를 보존해야 답변의 근거를 확인하고 검색 실패와 생성 실패를 나눠 진단할 수 있다.
- 실제 구현은 의미 유사도뿐 아니라 최신성, 접근 권한, 출처를 함께 검사해야 한다.

## 다음 글 예고

다음 글에서는 하나의 목표를 작은 단계로 나누고, 각 단계의 실행 결과에 따라 다음 행동을 고치는 **계획과 실행**의 최소 구조를 구현한다.

## 참고 자료

1. [Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) — 매개변수 지식과 검색 가능한 외부 기억을 결합한 RAG의 원 논문.
2. [OpenAI API, Retrieval](https://developers.openai.com/api/docs/guides/retrieval) — 의미 검색, Vector Store, 조각화·색인과 검색 결과의 공식 설명.
3. [OpenAI Agents SDK, Tools](https://openai.github.io/openai-agents-python/tools/) — `FileSearchTool`과 현재 검색 옵션의 공식 문서.
