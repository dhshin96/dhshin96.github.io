---
title: "AI 에이전트 구조화 출력 입문: JSON 스키마 검증 Python 실습"
date: 2026-10-23 21:44:00 +0900
categories: [지능형 시스템, 개발과 제품화]
tags: [AI-에이전트, 구조화-출력, JSON-스키마, 검증, Python]
description: "AI 에이전트의 자유로운 최종 답을 정해진 필드와 자료형을 가진 구조화 출력으로 바꾸는 원리를 배웁니다. 순수 Python 검증기로 정상 출력 1건과 잘못된 출력 3건을 구분합니다."
reading_time: 11
series: "처음 시작하는 AI 에이전트 개발"
level: "입문"
math: true
---

> **이번 글의 목표:** 에이전트의 최종 답을 **스키마가 있는 JSON 객체**로 받고, 필드 누락·잘못된 자료형·허용하지 않은 필드를 실행 전에 거절한다.

먼저 검증 결과부터 보자. 같은 요청에 대한 후보 출력 네 개를 순수 Python 검증기에 넣으면 정상 객체 한 개만 통과한다. `limit`가 문자열인 출력, `query`가 빠진 출력, `debug`가 몰래 추가된 출력은 각각 다른 이유로 거절된다.

```text
case=valid status=accepted action=search_docs limit=3
case=wrong_type status=rejected error=limit must be int
case=missing_field status=rejected error=keys mismatch: missing=['query'] extra=[]
case=extra_field status=rejected error=keys mismatch: missing=[] extra=['debug']
summary accepted=1 rejected=3
```

앞 글에서 만든 [에이전트 루프는 최종 답이 나오면 반복을 끝냈다]({{ '/posts/ai-agent-loop-state-python/' | relative_url }}). 이제 그 “최종 답”이 사람이 읽을 문장인지, 다음 코드가 바로 사용할 객체인지 구분할 차례다.

## 1. 직관: 문장을 계약이 있는 객체로 바꾼다

자유로운 문장 `문서에서 구조화 출력을 3개 찾아볼게요`는 사람에게 자연스럽다. 하지만 프로그램은 동작 이름, 검색어, 개수를 다시 추측해야 한다. 띄어쓰기나 표현이 바뀌면 문자열 파싱도 쉽게 깨진다.

구조화 출력은 답의 모양을 먼저 정한다. 이번 예제의 계약은 `action`, `query`, `limit` 세 필드가 모두 있어야 하고, 각각 문자열·문자열·정수여야 한다. `action`은 `search_docs` 또는 `answer`만 허용하고 `limit`는 1에서 5 사이다. 계약을 통과한 객체만 다음 단계로 보낸다.

<figure class="study-figure">
  <img src="{{ '/assets/images/ai-agents/structured-output-validation.svg' | relative_url }}" alt="자유로운 모델 출력이 JSON 스키마 검증기를 지나 필드와 자료형이 맞으면 실행 가능한 객체가 되고 누락, 잘못된 자료형, 추가 필드는 오류로 거절되는 흐름">
  <figcaption>그림 1. 구조화 출력은 모델의 답과 프로그램 실행 사이에 명시적인 검증 경계를 만든다.</figcaption>
</figure>

[도구 호출 글에서 입력 인수를 JSON 스키마로 검사한 이유]({{ '/posts/ai-agent-tool-calling-json-schema/' | relative_url }})와 방향만 다르다. 도구 호출에서는 모델이 제안한 **입력**을 검증했고, 이번에는 에이전트가 반환한 **최종 출력**을 검증한다.

## 2. 기호와 수식: 통과 또는 거절

후보 출력 객체를 $x$, 스키마를 $S$라고 하자. 검증 함수 $V_S(x)$는 계약을 모두 만족할 때만 1이다.

$$
V_S(x)=I(K_x=K_S)\cdot I(T_x=T_S)\cdot I(C_S(x))
$$

$K$는 필드 집합, $T$는 각 필드의 자료형, $C$는 허용값과 범위 같은 추가 제약이다. 세 조건 중 하나라도 실패하면 곱은 0이 되고 객체를 사용하지 않는다. 여기서 핵심은 JSON 문법이 맞는 것과 스키마가 맞는 것이 다르다는 점이다. `{"limit": "3"}`은 올바른 JSON이지만 `limit`가 정수라는 계약에는 맞지 않는다.

## 3. 작은 손검증: 네 후보 중 하나만 통과한다

필수 필드 집합을 $K_S=\{action, query, limit\}$라고 두자. 정상 후보는 세 필드가 정확히 일치하고 자료형과 범위도 맞으므로 통과한다.

| 후보 | 필드 집합 | 자료형·제약 | 결과 |
| --- | --- | --- | --- |
| 정상 | 일치 | `str`, `str`, `int`; `limit=3` | 통과 |
| 잘못된 자료형 | 일치 | `limit="3"` | 거절 |
| 필드 누락 | `query` 없음 | 검사 중단 | 거절 |
| 추가 필드 | `debug` 있음 | 허용하지 않은 키 | 거절 |

따라서 통과율은 $1/4=0.25$, 거절률은 $3/4=0.75$다. 이 비율은 모델 성능 평가가 아니라 검증기가 네 개의 고정 입력을 예상대로 분류하는지 확인한 단위 테스트 결과다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 패키지와 API를 쓰지 않는다. 실제 모델 호출 대신 네 개의 고정 후보를 넣어, 스키마 검증의 책임과 실패 경로만 결정적으로 재현한다. Python에서 `bool`은 `int`의 하위형처럼 동작하므로 `isinstance` 대신 `type(value) is int`로 엄격히 확인한다.

```python
ALLOWED_ACTIONS = {"search_docs", "answer"}
REQUIRED_KEYS = {"action", "query", "limit"}


def validate_output(data):
    keys = set(data)
    if keys != REQUIRED_KEYS:
        missing = sorted(REQUIRED_KEYS - keys)
        extra = sorted(keys - REQUIRED_KEYS)
        raise ValueError(f"keys mismatch: missing={missing} extra={extra}")
    if type(data["action"]) is not str:
        raise TypeError("action must be str")
    if type(data["query"]) is not str:
        raise TypeError("query must be str")
    if type(data["limit"]) is not int:
        raise TypeError("limit must be int")
    if data["action"] not in ALLOWED_ACTIONS:
        raise ValueError("action is not allowed")
    if not 1 <= data["limit"] <= 5:
        raise ValueError("limit must be between 1 and 5")
    return data


cases = {
    "valid": {"action": "search_docs", "query": "구조화 출력", "limit": 3},
    "wrong_type": {"action": "search_docs", "query": "구조화 출력", "limit": "3"},
    "missing_field": {"action": "search_docs", "limit": 3},
    "extra_field": {
        "action": "search_docs", "query": "구조화 출력", "limit": 3, "debug": True
    },
}

accepted = 0
for name, candidate in cases.items():
    try:
        result = validate_output(candidate)
        accepted += 1
        print(
            f"case={name} status=accepted "
            f"action={result['action']} limit={result['limit']}"
        )
    except (TypeError, ValueError) as error:
        print(f"case={name} status=rejected error={error}")

print(f"summary accepted={accepted} rejected={len(cases) - accepted}")
```

실행 출력은 글 첫머리와 같다. 성공 경로는 검증된 객체를 반환하고, 실패 경로는 오류 이유를 남긴다. 잘못된 값을 임의로 고쳐 실행하지 않는 것이 중요하다. 예를 들어 문자열 `"3"`을 자동으로 정수로 바꾸면 편리해 보이지만, 계약 위반을 숨기고 예상하지 못한 입력까지 통과시킬 수 있다.

## 5. 결과 해석: 형식 보장과 내용 보장은 다르다

구조화 출력은 후속 코드가 기대하는 필드와 자료형을 안정적으로 받게 한다. JSON Schema에서는 `required`로 필수 필드를 지정하고 `additionalProperties: false`로 선언하지 않은 필드를 막을 수 있다. OpenAI Agents SDK의 현재 Python 문서에서는 `Agent`에 `output_type`을 전달하면 일반 문자열 대신 구조화 출력을 사용하며, Pydantic 모델·dataclass·`TypedDict` 같은 형식을 지정할 수 있다.

다만 스키마를 통과했다고 내용이 사실이라는 뜻은 아니다. `query`가 문자열이라는 사실은 그 검색어가 좋은지 보장하지 않고, `action="search_docs"`가 허용값이라는 사실은 실제로 검색할 권한이 있다는 뜻도 아니다. 구조 검증 뒤에도 도구 권한, 값의 의미, 최신성, 안전 규칙을 별도로 확인해야 한다.

실패를 다루는 기본 순서는 단순하다. 오류가 난 객체는 실행하지 않고, 필드 위치와 오류 종류를 기록하고, 필요하면 제한된 횟수만 다시 생성을 요청한다. 그래도 실패하면 명시적으로 중단한다. Pydantic은 검증 중 발견한 오류를 `ValidationError` 하나에 모아 제공하므로 실제 애플리케이션에서는 어떤 필드가 왜 실패했는지 구조적으로 처리할 수 있다.

오류 기록에는 원문 전체보다 `case`, 실패한 필드, 오류 종류처럼 재현에 필요한 최소 정보만 남기는 편이 안전하다. 모델에게 재시도를 요청할 때도 “다시 써 줘”가 아니라 `limit은 정수여야 한다`처럼 실패한 계약을 정확히 전달한다. 그러면 형식 오류와 내용 오류를 섞지 않고 고칠 수 있다.

스키마가 바뀔 때의 경계도 필요하다. 다음 버전에서 `language` 필드를 추가하면 예전 소비자는 그 필드를 모를 수 있다. 출력과 소비 코드를 함께 갱신하거나 스키마 버전을 명시해야 한다. `additionalProperties`를 무조건 허용해 호환 문제를 덮으면 오타 필드도 함께 통과하므로, 입문 단계에서는 닫힌 필드 집합으로 시작하는 편이 검증 결과를 이해하기 쉽다.

## 6. 직접 바꿔 보는 연습

먼저 `limit`를 `0`과 `6`으로 바꿔 범위 오류가 나는지 확인해 보자. 다음에는 `action="delete_docs"`를 넣어 허용 목록 검사가 동작하는지 본다. 두 검사는 자료형이 맞아도 값의 제약 때문에 거절되는 사례다.

마지막으로 후보를 한 번만 고쳐 재검증하는 `repair_output` 함수를 만들어 보자. 단, 누락된 `query`를 프로그램이 추측하지는 말고 오류를 반환한다. 자동 복구의 범위를 명확히 정하는 것도 출력 계약의 일부다.

## 핵심 요약

- 구조화 출력은 자유로운 최종 문장을 필드·자료형·제약이 명시된 객체로 바꾼다.
- JSON 문법이 올바른 것과 스키마 검증을 통과하는 것은 서로 다른 조건이다.
- 누락, 잘못된 자료형, 허용하지 않은 필드는 실행 전에 거절하고 오류로 다룬다.
- 스키마는 형식을 보장할 뿐 사실성, 권한, 안전성까지 보장하지 않는다.

## 다음 글 예고

다음 글에서는 한 번의 실행을 넘어 정보를 보존하는 **에이전트 메모리**를 다룬다. 현재 상태와 장기 메모리를 구분하고, 저장할 정보와 다시 꺼낼 정보를 고르는 최소 예제를 구현한다.

## 참고 자료

1. [OpenAI Agents SDK, Agents](https://openai.github.io/openai-agents-python/agents/) — `output_type`과 구조화 출력 형식.
2. [Pydantic, Models](https://docs.pydantic.dev/latest/concepts/models/) — 모델 검증과 `ValidationError` 처리.
3. [JSON Schema, Object](https://json-schema.org/understanding-json-schema/reference/object) — `properties`, `required`, `additionalProperties`의 동작.
