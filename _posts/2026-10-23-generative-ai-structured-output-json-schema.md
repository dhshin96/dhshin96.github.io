---
title: "생성형 AI 구조화 출력 입문: JSON 스키마로 결과 검증하기"
date: 2026-10-23 20:00:00 +0900
categories: [학습과 모델링, 지능형 시스템]
tags: [생성형-AI, 구조화-출력, JSON-스키마, 출력-검증, Python]
description: "JSON과 JSON 스키마의 차이를 이해하고, required·자료형·추가 필드 규칙으로 생성 결과를 검사합니다. 순수 Python 검증기로 정상 결과 1개와 오류 결과 3개를 구분합니다."
reading_time: 10
series: "처음 시작하는 생성형 AI"
level: "입문"
math: true
---

> **이번 글의 목표:** 생성 결과의 모양을 JSON 스키마로 먼저 정하고, 필수 필드·자료형·추가 필드 규칙을 순수 Python으로 검사한다.

먼저 결과부터 보자. 같은 JSON 객체 네 개를 검사했더니 정상 결과만 `통과`했다. 나머지는 각각 `difficulty` 누락, 정수여야 할 값의 문자열 입력, 허용하지 않은 `note` 추가 때문에 `실패`했다. JSON으로 읽을 수 있다는 사실과 필요한 구조를 만족한다는 사실은 서로 다르다.

앞 글에서 [지시·맥락·입력·출력 형식을 나눠 요청 조건을 고정하는 방법]({{ '/posts/generative-ai-prompt-structure/' | relative_url }})을 배웠다. 이번에는 그중 출력 형식을 사람이 읽는 문장에서 프로그램이 검사할 수 있는 계약으로 바꾼다.

## 1. 직관: JSON은 상자이고, 스키마는 상자 검사표다

`{"concept": "구조화 출력"}`은 올바른 JSON이다. 중괄호 안에 키와 값을 넣는 문법을 지켰기 때문이다. 하지만 프로그램이 `concept`, `difficulty`, `keywords` 세 필드를 모두 기대한다면 이 결과는 쓸 수 없다. 문법 검사는 JSON을 **읽을 수 있는지** 답하고, 스키마 검사는 읽은 값이 **약속한 모양인지** 답한다.

이번에 사용할 계약은 단순하다.

- `concept`는 문자열이다.
- `difficulty`는 1, 2, 3 중 하나인 정수다.
- `keywords`는 문자열을 두 개 이상 담은 배열이다.
- 세 필드는 모두 필수이며, 다른 필드는 허용하지 않는다.

<figure class="study-figure">
  <img src="{{ '/assets/images/generative-ai/structured-output-schema-validation.svg' | relative_url }}" alt="네 JSON 후보를 파싱한 뒤 필수 필드, 자료형, 추가 필드 규칙으로 검사하는 흐름. 정상 객체 하나만 통과하고 필드 누락, 자료형 오류, 추가 필드가 있는 객체 세 개는 실패한다.">
  <figcaption>그림 1. 구조화 출력은 JSON 파싱에서 끝나지 않는다. 스키마 규칙을 모두 통과해야 프로그램이 기대한 데이터가 된다.</figcaption>
</figure>

## 2. 기호와 수식: 오류 집합이 비어야 통과한다

출력 객체를 $x$, 필수 필드 오류를 $V_{required}$, 자료형 오류를 $V_{type}$, 추가 필드 오류를 $V_{extra}$라고 하자. 전체 오류 집합은 다음과 같다.

$$
V(x)=V_{required}(x)\cup V_{type}(x)\cup V_{extra}(x)
$$

통과 여부 $P(x)$는 오류 개수만 보면 된다.

$$
P(x)=
\begin{cases}
1 & \text{if } |V(x)|=0 \\
0 & \text{if } |V(x)|>0
\end{cases}
$$

정상 예제는 오류가 0개라 통과한다. 필드 누락, 자료형 오류, 추가 필드 예제는 각각 오류가 1개라 실패한다. 실제 검증기는 모든 오류를 모아 보여 줄 수도 있지만, 입문 실습에서는 첫 오류에서 멈춰 원인과 규칙을 한 쌍씩 확인한다.

## 3. 작은 손검사: 키 집합부터 비교하기

필수 키 집합을 $R=\{concept, difficulty, keywords\}$라고 하자. 후보가 다음과 같다면 실제 키 집합 $K$에는 `difficulty`가 없다.

```json
{"concept": "구조화 출력", "keywords": ["JSON", "스키마"]}
```

따라서 $R-K=\{difficulty\}$이고 첫 검사에서 실패한다. 반대로 `note`가 더 들어오면 $K-R=\{note\}$가 된다. `additionalProperties: false`는 이 차집합이 비어 있어야 한다는 뜻이다. 필수 키와 추가 키 검사를 통과한 다음 각 값의 자료형과 허용 범위를 확인하면 된다.

JSON Schema 공식 안내에 따르면 `properties`는 각 필드의 규칙을 정하지만, 그것만으로 필드가 필수가 되지는 않는다. 필수 키는 `required`에 따로 적어야 한다. 또한 일반 JSON Schema에서는 추가 필드가 기본적으로 허용되므로, 닫힌 구조가 필요하면 `additionalProperties`를 `false`로 둔다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 패키지나 API를 쓰지 않는다. 이번 스키마에서 필요한 규칙만 함수로 옮겨 네 결과를 결정적으로 검사한다.

```python
schema = {
    "properties": {
        "concept": {"type": "string"},
        "difficulty": {"type": "integer", "enum": [1, 2, 3]},
        "keywords": {"type": "array", "items": "string", "minItems": 2},
    },
    "required": ["concept", "difficulty", "keywords"],
    "additionalProperties": False,
}


def validate(data, rule):
    missing = [key for key in rule["required"] if key not in data]
    if missing:
        return False, f"필수 필드 누락: {', '.join(missing)}"

    allowed = set(rule["properties"])
    extra = sorted(set(data) - allowed)
    if not rule["additionalProperties"] and extra:
        return False, f"허용되지 않은 필드: {', '.join(extra)}"

    if not isinstance(data["concept"], str):
        return False, "concept: string 필요"
    if type(data["difficulty"]) is not int:
        return False, "difficulty: integer 필요"
    if data["difficulty"] not in rule["properties"]["difficulty"]["enum"]:
        return False, "difficulty: 1, 2, 3 중 하나 필요"
    if not isinstance(data["keywords"], list):
        return False, "keywords: array 필요"
    if len(data["keywords"]) < 2 or not all(
        isinstance(item, str) for item in data["keywords"]
    ):
        return False, "keywords: 문자열 2개 이상 필요"
    return True, "스키마 일치"


cases = {
    "정상": {"concept": "구조화 출력", "difficulty": 1,
             "keywords": ["JSON", "스키마"]},
    "필드 누락": {"concept": "구조화 출력",
                  "keywords": ["JSON", "스키마"]},
    "자료형 오류": {"concept": "구조화 출력", "difficulty": "1",
                    "keywords": ["JSON", "스키마"]},
    "추가 필드": {"concept": "구조화 출력", "difficulty": 1,
                  "keywords": ["JSON", "스키마"], "note": "초보"},
}

for name, data in cases.items():
    passed, reason = validate(data, schema)
    print(f"{name}: {'통과' if passed else '실패'} - {reason}")
```

실행 출력은 다음과 같다.

```text
정상: 통과 - 스키마 일치
필드 누락: 실패 - 필수 필드 누락: difficulty
자료형 오류: 실패 - difficulty: integer 필요
추가 필드: 실패 - 허용되지 않은 필드: note
```

## 5. 결과 해석: 형식 보장은 내용 보장이 아니다

정상 결과의 `concept`에 틀린 설명을 넣어도 자료형이 문자열이면 이 검사는 통과한다. 스키마는 필드의 존재와 모양을 검사할 뿐, 사실성이나 설명의 품질까지 판정하지 않는다. 내용 평가는 정답 예시, 채점 기준, 테스트 데이터로 별도 수행해야 한다.

2026년 10월 23일 확인한 OpenAI 공식 문서에서 Responses API의 구조화 출력은 `text.format`에 `type: "json_schema"`와 스키마를 지정하는 방식이다. 단순 JSON 모드는 문법상 올바른 JSON을 목표로 하지만, 구조화 출력은 지원되는 JSON Schema 범위 안에서 스키마 준수를 목표로 한다. 현재 엄격한 구조화 출력에서는 객체의 모든 필드를 `required`에 넣고 `additionalProperties: false`를 설정해야 한다. 지원 범위와 API 이름은 바뀔 수 있으므로 실제 호출 전 공식 문서를 다시 확인해야 한다.

외부 API를 사용하더라도 거절, 중단, 불완전 응답 같은 예외 처리는 남는다. 또한 모델 출력으로 도구를 실행한다면 형식 검사 뒤에 권한과 사용자 승인도 따로 확인해야 한다. 그 다음 단계가 궁금하다면 [JSON 스키마를 도구 호출 인수와 연결하는 글]({{ '/posts/ai-agent-tool-calling-json-schema/' | relative_url }})에서 검증과 실행의 경계를 이어서 볼 수 있다.

## 6. 직접 바꿔 보는 연습

1. 정상 예제의 `difficulty`를 `4`로 바꾸고 허용 범위 오류를 확인한다.
2. `keywords`를 문자열 하나만 가진 배열로 바꾸고 최소 개수 검사를 통과하는지 확인한다.
3. `summary` 문자열 필드를 스키마와 네 사례에 추가한다. 필수 키 목록을 함께 바꾸지 않았을 때 어떤 차이가 생기는지 비교한다.

## 핵심 요약

- JSON 문법이 맞는 것과 스키마가 요구한 구조를 만족하는 것은 다르다.
- `properties`는 필드 규칙, `required`는 필수 키, `additionalProperties: false`는 예상 밖의 키를 막는 규칙이다.
- 순수 Python 검사에서 정상 결과 1개만 통과했고, 누락·자료형·추가 필드 오류 3개는 각각 실패했다.
- 구조 검증 뒤에도 사실성 평가와 예외 처리, 실행 권한 검사는 별도로 필요하다.

## 다음 글 예고

다음 글에서는 **RAG**를 다룬다. 작은 문서 조각에서 질문과 가까운 근거를 검색하고, 검색된 근거만 생성 입력에 붙이는 흐름을 순수 Python으로 재현한다.

## 참고 자료

1. OpenAI, [Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses) — Responses API의 `text.format`, 구조화 출력과 JSON 모드의 차이, 지원되는 스키마 제약(2026-10-23 확인).
2. OpenAI, [Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create) — `json_schema` 응답 형식과 `strict` 필드의 현재 API 정의(2026-10-23 확인).
3. JSON Schema, [Object](https://json-schema.org/understanding-json-schema/reference/object) — `properties`, `required`, `additionalProperties`의 동작.
