---
title: "AI 에이전트 도구 호출 입문: JSON 스키마로 함수 연결하기"
date: 2026-10-03 20:00:00 +0900
categories: [지능형 시스템, 개발과 제품화]
tags: [AI-에이전트, 도구-호출, 함수-호출, JSON-스키마, Python]
description: "AI 에이전트가 도구 이름과 인수를 구조화해 제안하는 원리를 배웁니다. 순수 Python 검증기로 정상 호출과 잘못된 추가 인수를 구분하고 53,700원 계산을 재현합니다."
reading_time: 11
series: "처음 시작하는 AI 에이전트 개발"
level: "입문"
math: true
---

> **이번 글의 목표:** 모델이 도구를 직접 실행하는 것이 아니라 **도구 이름과 인수를 제안**한다는 점을 이해하고, 애플리케이션이 요청을 검증한 뒤 함수를 실행하는 최소 구조를 Python으로 재현한다.

먼저 결과부터 보자. `17,900원 상품 3개의 총액을 계산해 줘`라는 요청을 `multiply(a=17900, b=3)` 호출로 바꾸면 결과는 **53,700원**이다. 하지만 정의에 없는 `coupon` 인수가 섞이면 계산 전에 `허용되지 않은 인수: coupon`으로 거절된다. 에이전트의 출발점은 모델이 모든 일을 맡는 것이 아니라, **모델의 제안과 실제 코드 실행 사이에 계약과 검증 경계를 두는 것**이다.

이 글은 AI 에이전트 개발 시리즈의 첫 단원이다. 주제가 놓인 큰 지도를 먼저 보고 싶다면 [지능형 시스템과 개발·제품화 영역을 소개한 글]({{ '/posts/welcome/' | relative_url }})이 도움이 된다.

## 1. 직관: 모델은 호출서를 쓰고, 프로그램은 실행한다

일반적인 텍스트 생성은 모델이 문장을 반환하면 끝난다. 도구 호출에는 중간 산출물이 하나 더 있다. 모델은 `multiply`라는 도구를 고르고 `a=17900`, `b=3`을 담은 호출서를 반환한다. 애플리케이션은 허용된 도구인지, 인수 이름과 자료형이 맞는지 확인한 뒤 실제 Python 함수를 실행한다.

이 구분은 모델 출력이 곧 실행 권한은 아니라는 뜻이다. 모델은 없는 도구 이름을 내거나 빠진 인수, 불필요한 인수를 만들 수 있다. 파일 삭제처럼 상태를 바꾸는 도구라면 형식이 맞아도 사용자 권한과 승인 범위를 별도로 검사해야 한다. JSON 스키마는 입력 모양을 좁히는 계약이지 권한을 판단하는 장치가 아니다.

<figure class="study-figure">
  <img src="{{ '/assets/images/ai-agents/tool-calling-contract.svg' | relative_url }}" alt="사용자 요청이 모델의 구조화된 도구 호출 제안으로 바뀌고 애플리케이션의 스키마 검증을 통과한 뒤 함수가 실행되어 결과가 다시 모델로 돌아가는 흐름">
  <figcaption>그림 1. 모델은 호출을 제안하고, 애플리케이션은 검증·실행·결과 반환을 책임진다.</figcaption>
</figure>

## 2. 기호와 수식: 호출과 실행 결과를 분리하기

도구 호출을 $c=(n,a)$라고 쓰자. $n$은 도구 이름, $a$는 인수 객체다. 등록된 도구 집합을 $\mathcal{T}$, 스키마 검증 함수를 $V$라고 하면 실행 조건은 다음과 같다.

$$
V(c)=\text{true} \quad\text{and}\quad n\in\mathcal{T}
$$

조건을 통과할 때만 디스패처가 이름에 맞는 함수 $T_n$을 찾아 결과를 계산한다.

$$
y=T_n(a)
$$

이번 예제에서 $n=\texttt{multiply}$, $a=\{\texttt{a}:17900,\texttt{b}:3\}$이므로 $y=17900\times3=53700$이다. 모델이 먼저 반환하는 것은 $y$가 아니라 호출 $c$이고, 결과 $y$는 애플리케이션의 함수가 만든다.

## 3. 작은 손계산: 스키마가 무엇을 막는가

`multiply` 도구의 계약을 표로 적으면 다음과 같다.

| 항목 | 계약 |
| --- | --- |
| 도구 이름 | `multiply` |
| 필수 인수 | `a`, `b` |
| 인수 자료형 | 정수 |
| 추가 인수 | 허용하지 않음 |

정상 호출은 필수 인수를 모두 가지며 추가 인수가 없다. $17{,}900\times3=53{,}700$이므로 함수 출력도 손으로 확인할 수 있다. 반면 `{"a": 17900, "b": 3, "coupon": 1000}`은 숫자만 담겼어도 계약에 없는 `coupon` 때문에 유효하지 않다. JSON Schema에서 `additionalProperties`를 `false`로 두면 선언하지 않은 속성을 거절한다.

형식 검증과 의미 검증도 구별해야 한다. `a=-17900`은 정수라는 형식에는 맞지만 상품 가격으로 허용할지는 별도 규칙이다. 실제 프로그램에서는 최솟값, 허용 목록, 인증, 실행 횟수 같은 조건도 함께 검사한다.

strict 모드는 모델이 계약에 맞는 호출을 만들 가능성을 높이지만, 실행 전 검증을 없애지는 않는다. 모델 출력은 여전히 외부 입력으로 취급하고, 서버가 현재 등록한 도구와 사용자의 권한을 기준으로 다시 확인해야 한다. 스키마는 호출의 모양을 고정하고, 애플리케이션 검증은 실제 실행 여부를 결정한다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 API나 패키지 없이 도구 정의, 호출 검증, 함수 실행을 재현한다. `model_calls`는 실제 모델 응답이 아니라 학습용으로 고정한 모의 출력이므로 비밀키가 필요 없고 실행할 때마다 결과가 같다.

```python
import json

tool_schema = {
    "name": "multiply",
    "required": {"a", "b"},
    "additional_properties": False,
}


def multiply(a, b):
    return a * b


tools = {"multiply": multiply}


def execute_tool(call):
    name = call.get("name")
    arguments = call.get("arguments", {})

    if name not in tools:
        return {"ok": False, "error": f"알 수 없는 도구: {name}"}

    given = set(arguments)
    missing = sorted(tool_schema["required"] - given)
    extra = sorted(given - tool_schema["required"])
    if missing:
        return {"ok": False, "error": f"빠진 인수: {', '.join(missing)}"}
    if extra and not tool_schema["additional_properties"]:
        return {"ok": False, "error": f"허용되지 않은 인수: {', '.join(extra)}"}
    if any(type(arguments[key]) is not int for key in tool_schema["required"]):
        return {"ok": False, "error": "a와 b는 정수여야 합니다"}

    return {"ok": True, "value": tools[name](**arguments)}


model_calls = [
    {"name": "multiply", "arguments": {"a": 17900, "b": 3}},
    {"name": "multiply", "arguments": {"a": 17900, "b": 3, "coupon": 1000}},
]

for call in model_calls:
    result = execute_tool(call)
    print("호출:", json.dumps(call, ensure_ascii=False, separators=(",", ":")))
    print("결과:", json.dumps(result, ensure_ascii=False, separators=(",", ":")))

valid_result = execute_tool(model_calls[0])
print(f"최종 답변: 17,900원 상품 3개의 총액은 {valid_result['value']:,}원입니다.")
```

실행 출력은 다음과 같다.

```text
호출: {"name":"multiply","arguments":{"a":17900,"b":3}}
결과: {"ok":true,"value":53700}
호출: {"name":"multiply","arguments":{"a":17900,"b":3,"coupon":1000}}
결과: {"ok":false,"error":"허용되지 않은 인수: coupon"}
최종 답변: 17,900원 상품 3개의 총액은 53,700원입니다.
```

## 5. 결과 해석: 구조화된 호출은 실행 결과가 아니다

첫 호출은 이름, 필수 인수, 자료형을 모두 만족해 `multiply`가 실행되었다. 둘째 호출은 함수에 도달하기 전에 멈췄다. 디스패처는 이렇게 모델 출력과 실제 기능 사이에서 허용된 함수만 연결한다. 도구가 늘어도 이름을 함수에 대응시키고 각 계약에 따라 인수를 검사한다는 구조는 같다.

공식 문서의 필드 이름은 제공자마다 다르다. OpenAI Responses API는 함수 호출을 애플리케이션이 실행한 뒤 `function_call_output`을 돌려준다. 현재 strict 모드는 각 객체에 `additionalProperties: false`를 두고 모든 속성을 `required`에 넣도록 요구한다. Anthropic의 클라이언트 도구는 `input_schema`, `tool_use`, `tool_result`라는 이름을 쓴다. 이름은 달라도 **도구 정의 → 구조화된 호출 → 애플리케이션 실행 → 결과 반환**이라는 왕복은 같다.

도구 설명도 계약의 일부다. “계산 도구”보다 “두 정수의 곱을 반환한다”가 선택 조건과 인수 의미를 더 분명히 전한다. 그래도 설명만 믿어서는 안 된다. 성공 호출뿐 아니라 잘못된 이름, 누락 인수, 추가 인수를 테스트해야 한다.

모델에 들어가는 문장이 먼저 어떤 숫자 표현으로 바뀌는지 궁금하다면 [생성형 AI 토큰화 입문 글]({{ '/posts/generative-ai-tokenization-bpe/' | relative_url }})을 함께 읽어 보자. 토큰화는 모델 입력을 만드는 단계이고, 도구 스키마는 모델 출력과 애플리케이션 코드를 잇는 단계다.

## 6. 직접 바꿔 보는 연습

정상 호출에서 `b`를 지워 보자. 검증기는 함수를 실행하지 않고 `빠진 인수: b`를 반환해야 한다. 이어서 `b`를 문자열 `"3"`으로 바꾸면 자료형 검사에서 멈춰야 한다. 숫자로 자동 변환하지 않으면 입력 오류를 조용히 숨기지 않는다.

다음에는 `divide(a, b)` 도구를 추가해 보자. $b=0$은 스키마를 통과하더라도 실행 규칙에서 거절해야 한다. 이 차이를 확인하면 **형식이 유효하다**와 **실행해도 안전하다**가 같은 말이 아님을 알 수 있다.

## 핵심 요약

- 모델은 도구를 직접 실행하기보다 도구 이름과 인수를 담은 구조화된 호출을 제안한다.
- JSON 스키마는 필수 인수, 자료형, 추가 속성 허용 여부를 정하는 인터페이스 계약이다.
- 애플리케이션은 호출을 검증하고 허용된 함수만 실행한 뒤 결과를 모델에 돌려준다.
- 스키마가 맞아도 권한, 범위, 부작용 검사는 별도로 필요하다.

## 다음 글 예고

다음 글에서는 한 번의 도구 호출을 **에이전트 루프와 상태**로 확장한다. 모델 출력이 최종 답인지 도구 호출인지 판별하고, 도구 결과를 상태에 추가해 다음 단계로 이어 가는 반복 구조를 구현한다.

## 참고 자료

1. [OpenAI API, Function calling](https://developers.openai.com/api/docs/guides/function-calling) — 함수 도구 정의, strict 모드, `function_call_output` 반환 흐름.
2. [Anthropic, Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) — `input_schema`, `tool_use`, `tool_result`로 이어지는 클라이언트 도구 구조.
3. [JSON Schema, Object reference](https://json-schema.org/understanding-json-schema/reference/object) — `properties`, `required`, `additionalProperties`의 동작.
