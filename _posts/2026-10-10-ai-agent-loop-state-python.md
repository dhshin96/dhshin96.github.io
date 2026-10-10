---
title: "AI 에이전트 루프와 상태 입문: 도구 결과를 잇는 Python 실습"
date: 2026-10-10 21:20:00 +0900
categories: [지능형 시스템, 개발과 제품화]
tags: [AI-에이전트, 에이전트-루프, 상태, 도구-호출, Python]
description: "AI 에이전트가 도구 호출과 관찰 결과를 상태에 쌓으며 반복하는 원리를 배웁니다. 순수 Python으로 두 번의 도구 호출 뒤 10을 답하고 최대 단계에서 멈추는 루프를 재현합니다."
reading_time: 11
series: "처음 시작하는 AI 에이전트 개발"
level: "입문"
math: true
---

> **이번 글의 목표:** 모델의 한 번짜리 도구 호출을 **생각–행동–관찰 루프**로 확장하고, 매 단계의 메시지와 종료 조건을 명시적인 상태로 관리한다.

먼저 결과부터 보자. `2와 3을 더한 뒤 그 결과를 두 배로 만들어 줘`라는 요청은 `add(2, 3) → 5`, `multiply(5, 2) → 10`을 거쳐 세 번째 모델 판단에서 끝난다. 실행 상태에는 사용자 요청, 두 호출, 두 도구 결과가 순서대로 남고 최종 출력은 **10**이다. 같은 모델이 계속 도구만 요청하더라도 `max_steps=4`를 넘으면 중단한다.

한 번의 호출이 아직 낯설다면 [JSON 스키마로 도구와 함수를 연결한 앞 글]({{ '/posts/ai-agent-tool-calling-json-schema/' | relative_url }})을 먼저 읽자. 이번 글은 그 호출 결과를 다음 모델 입력에 다시 넣는 순간부터 시작한다.

## 1. 직관: 결과를 관찰해야 다음 행동을 고를 수 있다

단일 도구 호출은 “무엇을 실행할까?”에 답한다. 에이전트 루프는 실행 결과를 관찰한 뒤 “이제 무엇을 할까?”를 다시 묻는다. 첫 `add` 결과인 `5`를 상태에 넣지 않으면 다음 `multiply`는 어떤 값을 두 배로 만들지 알 수 없다.

따라서 루프의 주인공은 모델만이 아니다. 애플리케이션이 현재 상태를 모델에 전달하고, 출력이 도구 호출인지 최종 답인지 분류하고, 허용된 도구를 실행하고, 관찰 결과를 상태에 추가한다. 모델은 다음 행동을 제안하고 프로그램은 반복과 종료를 통제한다.

<figure class="study-figure">
  <img src="{{ '/assets/images/ai-agents/agent-loop-state.svg' | relative_url }}" alt="사용자 요청과 도구 결과가 상태에 쌓이고 모델 판단이 도구 호출이면 실행 결과를 상태에 추가해 반복하며 최종 답이면 종료하는 에이전트 루프">
  <figcaption>그림 1. 상태는 다음 판단에 필요한 기록이고, 종료 조건은 반복의 경계를 정한다.</figcaption>
</figure>

## 2. 기호와 수식: 상태 전이로 보는 한 단계

$t$번째 상태를 $S_t$, 모델의 판단을 $pi(S_t)$라고 하자. 판단은 도구 호출 $c_t$ 또는 최종 답 $f_t$ 중 하나다.

$$
\pi(S_t)=
\begin{cases}
c_t & \text{도구가 더 필요할 때}\\
f_t & \text{답을 마칠 때}
\end{cases}
$$

도구 호출이면 애플리케이션이 결과 $o_t=T(c_t)$를 얻어 상태에 호출과 관찰을 덧붙인다.

$$
S_{t+1}=S_t\mathbin{\Vert}[c_t,o_t]
$$

$\Vert$는 목록을 이어 붙인다는 뜻이다. 최종 답이면 상태를 더 바꾸지 않고 종료한다. 실제 프로그램에는 $t < M$이라는 최대 단계 조건도 둔다. 답이 나오지 않아도 $t=M$이면 멈춰야 비용과 실행 시간을 제한할 수 있다.

## 3. 작은 손계산: 두 관찰이 다음 입력이 된다

초기 상태 $S_0$에는 사용자 요청 하나만 있다. 첫 판단은 `add(a=2, b=3)`이고 도구 결과는 $2+3=5$다. 호출과 결과를 붙인 $S_1$에서 모델은 `5`를 읽고 `multiply(a=5, b=2)`를 고른다. 두 번째 관찰은 $5\times2=10$이다.

| 단계 | 모델 판단 | 관찰 | 누적 메시지 수 |
| --- | --- | --- | ---: |
| 1 | `add(2, 3)` | `5` | 3 |
| 2 | `multiply(5, 2)` | `10` | 5 |
| 3 | 최종 답 | `10` | 5 |

메시지 수가 1에서 바로 3이 되는 이유는 도구 호출과 도구 결과를 각각 기록하기 때문이다. 최종 답은 루프의 반환값으로 분리했다. 구현에 따라 답도 대화 기록에 넣을 수 있지만, 어떤 선택이든 상태의 항목과 순서를 명확히 정해야 한다.

## 4. 실행 가능한 Python 실습

아래 코드는 외부 API와 패키지를 쓰지 않는다. `mock_model`은 현재 상태를 읽어 다음 행동을 고르는 결정적 모의 모델이다. 실제 서비스에서는 이 함수 자리에 모델 호출이 들어가지만, 루프와 상태 전이의 책임은 그대로 애플리케이션에 남는다.

```python
def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


TOOLS = {"add": add, "multiply": multiply}


def mock_model(messages):
    observations = [m["value"] for m in messages if m["type"] == "observation"]
    if not observations:
        return {"type": "tool_call", "name": "add", "arguments": {"a": 2, "b": 3}}
    if len(observations) == 1:
        return {
            "type": "tool_call",
            "name": "multiply",
            "arguments": {"a": observations[-1], "b": 2},
        }
    return {"type": "final", "answer": f"계산 결과는 {observations[-1]}입니다."}


def run_agent(user_request, max_steps=4):
    state = {"messages": [{"type": "user", "content": user_request}], "status": "running"}

    for step in range(1, max_steps + 1):
        action = mock_model(state["messages"])
        if action["type"] == "final":
            state.update(status="completed", turns=step, answer=action["answer"])
            print(f"step={step} final={action['answer']}")
            return state

        name, arguments = action["name"], action["arguments"]
        state["messages"].append(action)
        result = TOOLS[name](**arguments)
        state["messages"].append({"type": "observation", "name": name, "value": result})
        print(f"step={step} action={name} args={arguments}")
        print(f"observation={result} state_items={len(state['messages'])}")

    state.update(status="stopped", turns=max_steps, answer=None)
    return state


result = run_agent("2와 3을 더한 뒤 그 결과를 두 배로 만들어 줘")
tool_calls = sum(m["type"] == "tool_call" for m in result["messages"])
print(
    f"status={result['status']} turns={result['turns']} "
    f"tool_calls={tool_calls} answer={result['answer']}"
)
```

실행 출력은 다음과 같다.

```text
step=1 action=add args={'a': 2, 'b': 3}
observation=5 state_items=3
step=2 action=multiply args={'a': 5, 'b': 2}
observation=10 state_items=5
step=3 final=계산 결과는 10입니다.
status=completed turns=3 tool_calls=2 answer=계산 결과는 10입니다.
```

## 5. 결과 해석: 상태와 메모리는 같은 말이 아니다

출력의 `turns=3`은 도구 호출 횟수가 아니다. 모델 판단은 세 번, 도구 실행은 두 번이다. `state_items=5`는 사용자 메시지 1개와 호출·관찰 두 쌍이 쌓였다는 뜻이다. 두 번째 판단이 첫 관찰 `5`를 읽었기 때문에 인수 `a=5`를 만들 수 있었다.

여기서 상태는 **현재 실행을 이어 가는 데 필요한 값**이다. 대화 기록, 중간 결과, 단계 수, 승인 대기 여부 등이 들어갈 수 있다. 장기 메모리는 여러 실행 사이에 정보를 보존하고 다시 꺼내는 별도 문제다. 모든 기록을 무조건 상태에 넣으면 입력이 길어지므로, 이번에는 현재 계산에 필요한 항목만 남겼다.

공식 OpenAI Agents SDK 문서에서 `Runner.run()`, `Runner.run_sync()`, `Runner.run_streamed()`는 최종 출력이 나올 때까지 루프를 관리한다. 도구 호출이 나오면 실행 결과를 추가해 다시 돌고, `max_turns`를 넘으면 `MaxTurnsExceeded`로 멈춘다. 대화 상태는 `result.to_input_list()`, 세션, `previous_response_id`나 `conversation_id` 중 한 방식을 선택할 수 있으며, 클라이언트 관리 방식과 서버 관리 방식을 무심코 섞으면 기록이 중복될 수 있다.

모델이 한 단계에서 다음 토큰을 만드는 내부 과정은 [트랜스포머 추론 입문 글]({{ '/posts/generative-ai-transformer-inference/' | relative_url }})에서 확인할 수 있다. 한 번의 모델 추론과 그 바깥에서 도구 결과를 누적하는 에이전트 루프는 서로 다른 층의 반복이다.

## 6. 직접 바꿔 보는 연습

`max_steps=2`로 바꿔 실행해 보자. 두 번째 도구 결과 `10`까지 상태에 들어가지만 세 번째 최종 판단 기회가 없어 `status=stopped`, `answer=None`이 된다. 최대 단계가 “도구 호출 수”가 아니라 “모델 판단 기회”를 제한한다는 점을 확인할 수 있다.

다음에는 `TOOLS`에 `subtract`를 추가하고 `10에서 4를 뺀 뒤 세 배` 요청을 고정된 모의 행동으로 재현해 보자. 각 관찰이 바로 다음 호출의 인수가 되는지, 알 수 없는 도구 이름을 실행 전에 거절하는지도 검사한다.

## 핵심 요약

- 에이전트 루프는 모델 판단, 도구 실행, 결과 관찰을 최종 답까지 반복한다.
- 상태는 다음 판단에 필요한 요청·호출·관찰과 실행 제어 값을 순서대로 담는다.
- 도구 결과를 상태에 추가해야 다음 모델 호출이 이전 행동의 결과를 사용할 수 있다.
- 최대 단계와 최종 답 판별은 무한 반복을 막는 애플리케이션의 책임이다.

## 다음 글 예고

다음 글에서는 자유로운 최종 문장을 **구조화 출력**으로 바꾼다. 답의 필드와 자료형을 스키마로 고정하고, 검증 실패를 안전하게 처리하는 최소 예제를 구현한다.

## 참고 자료

1. [OpenAI Agents SDK, Running agents](https://openai.github.io/openai-agents-python/running_agents/) — `Runner`의 에이전트 루프, `max_turns`, 상태 지속 방식.
2. [OpenAI Agents SDK, Quickstart](https://openai.github.io/openai-agents-python/quickstart/) — `Agent`, `Runner`, 두 번째 턴의 상태 전달 선택지.
3. [Anthropic, Trustworthy agents in practice](https://www.anthropic.com/research/trustworthy-agents) — 계획·행동·관찰·조정으로 이어지는 루프와 인간 통제 경계.
