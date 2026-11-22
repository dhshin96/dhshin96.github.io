---
title: "AI 에이전트 계획과 실행 입문: 실패 후 경로를 고치는 Python 실습"
date: 2026-11-22 22:50:00 +0900
categories: [지능형 시스템, 개발과 제품화]
tags: [AI-에이전트, 계획, 실행, 재계획, Python]
description: "AI 에이전트가 목표를 행동 순서로 나누고 실행 관찰에 따라 남은 계획을 고치는 구조를 배웁니다. 막힌 경로를 우회하는 순수 Python 예제로 재계획 1회를 검증합니다."
reading_time: 11
series: "처음 시작하는 AI 에이전트 개발"
level: "입문"
math: true
---

> **이번 글의 목표:** 목표까지의 **계획**을 만들고, 한 단계씩 **실행**한 관찰을 이용해 실패한 계획을 고치는 최소 구조를 구현한다.

먼저 실행 결과를 보자. 처음 계획은 `A -> B -> D`지만 `B -> D`가 막혀 있다. 에이전트는 현재 위치 `B`를 유지하고 막힌 간선을 기록한 뒤, 남은 경로를 `B -> C -> D`로 다시 계산한다. 재계획은 한 번, 이동 시도는 네 번이다.

```text
plan_v1=A -> B -> D
execute=A -> B observation=ok state=B
execute=B -> D observation=blocked state=B
plan_v2=B -> C -> D
execute=B -> C observation=ok state=C
execute=C -> D observation=ok state=D
final=D replans=1 attempts=4
```

앞 글에서 [질문과 관련된 근거를 검색해 답변 입력에 결합하는 방법]({{ '/posts/ai-agent-retrieval-augmentation/' | relative_url }})을 배웠다. 검색 결과도 실행 뒤에 얻는 관찰이다. 이번에는 관찰을 답변 근거로만 쓰지 않고, 아직 실행하지 않은 행동 순서를 바꾸는 데 사용한다.

## 1. 직관: 계획은 약속이 아니라 수정 가능한 후보다

계획은 목표를 바로 실행할 수 있는 작은 행동의 순서로 바꾼 것이다. 출발점 `A`에서 목표 `D`로 간다면 `A→B`, `B→D`처럼 한 번에 검사할 수 있는 이동으로 나눈다. 실행기는 첫 행동만 수행하고 `성공`, `차단`, `시간 초과` 같은 관찰을 돌려준다.

중요한 점은 처음 계획 전체를 무조건 밀어붙이지 않는 것이다. `B→D`가 막혔는데도 다음 단계가 성공했다고 가정하면 내부 상태와 실제 환경이 달라진다. 실패 관찰을 상태에 반영하고 현재 위치에서 남은 계획을 다시 만들어야 한다. 이 글에서는 이를 **재계획**이라고 부른다.

<figure class="study-figure">
  <img src="{{ '/assets/images/ai-agents/plan-execute-replan-loop.svg' | relative_url }}" alt="출발점 A에서 목표 D까지 계획한 뒤 A에서 B로 이동하고, B에서 D로 가는 길이 막혔다는 관찰을 받아 B에서 C를 거쳐 D로 가는 계획으로 수정하는 흐름">
  <figcaption>그림 1. 실행 관찰이 계획기로 돌아가면 남은 행동만 현재 상태에서 다시 계산할 수 있다.</figcaption>
</figure>

[에이전트 루프가 행동과 도구 결과를 상태에 차례로 쌓는 구조]({{ '/posts/ai-agent-loop-state-python/' | relative_url }})와 비교하면 역할이 선명하다. 루프는 `행동→관찰→다음 판단`을 반복하는 바깥 구조이고, 계획기는 그 판단 시점에 여러 단계의 후보를 만든다. 계획과 실행은 별도의 제품이 아니라 같은 루프 안에서 책임을 나눈 두 부분이다.

## 2. 기호와 수식: 관찰 뒤 남은 계획을 갱신한다

목표를 $g$, 시점 $t$의 상태를 $s_t$, 계획을 행동 목록 $P_t=(a_{t,1},a_{t,2},\ldots)$라고 하자. 실행기 $E$는 첫 행동을 수행해 새 상태와 관찰을 반환한다.

$$
(s_{t+1}, o_t)=E(s_t,a_{t,1})
$$

관찰이 성공이면 실행한 행동을 계획에서 제거한다. 실패이면 실패 정보를 지식 $K_t$에 추가하고, 계획기 $R$이 현재 상태부터 목표까지의 경로를 다시 만든다.

$$
P_{t+1}=R(g,s_{t+1},K_t\cup\{o_t\})
$$

이 식은 언어 모델의 내부 사고 과정을 공개하라는 뜻이 아니다. 프로그램이 검사할 수 있는 `현재 상태`, `다음 행동`, `관찰`, `남은 계획`을 명시적으로 관리하자는 뜻이다. 계획에는 실행 가능한 도구 이름과 인수만 담고, 관찰은 도구의 실제 반환값으로 채워야 한다.

## 3. 작은 손계산: 막힌 간선 하나를 반영한다

그래프의 간선이 `A→B`, `B→C`, `B→D`, `C→D`라고 하자. 처음에는 모든 간선을 사용할 수 있다고 알고 있으므로, 간선 두 개인 최단 경로 `A→B→D`를 고른다.

첫 이동 `A→B`는 성공한다. 두 번째 `B→D`는 실제 환경에서 막혀 있으므로 상태는 `B`에 그대로 남는다. 알려진 차단 집합에 `(B,D)`를 추가하면 사용할 수 있는 최단 경로는 `B→C→D` 하나다. 이미 성공한 `A→B`까지 다시 계획하지 않는 이유는 현재 상태가 출발점이 아니라 `B`이기 때문이다.

| 시점 | 현재 상태 | 다음 행동 | 관찰 | 남은 계획 |
| --- | --- | --- | --- | --- |
| 0 | A | A→B | 성공 | B→D |
| 1 | B | B→D | 차단 | 재계획 필요 |
| 2 | B | B→C | 성공 | C→D |
| 3 | C | C→D | 성공 | 완료 |

## 4. 실행 가능한 Python 실습

아래 코드는 표준 라이브러리만 사용한다. `make_plan`은 너비 우선 탐색으로 현재 알려진 그래프의 최단 경로를 만든다. 실행 중 실제 차단을 만나면 그 간선을 `known_blocked`에 넣고 현재 상태에서 다시 호출한다.

```python
from collections import deque

GRAPH = {
    "A": ("B",),
    "B": ("C", "D"),
    "C": ("D",),
    "D": (),
}
ACTUALLY_BLOCKED = {("B", "D")}


def make_plan(start, goal, known_blocked):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for next_node in GRAPH[node]:
            edge = (node, next_node)
            if edge not in known_blocked and next_node not in visited:
                visited.add(next_node)
                queue.append((next_node, path + [next_node]))
    raise ValueError("경로를 찾지 못했습니다.")


state = "A"
goal = "D"
known_blocked = set()
plan = make_plan(state, goal, known_blocked)
version = 1
replans = 0
attempts = 0
print(f"plan_v{version}=" + " -> ".join(plan))

while state != goal:
    next_state = plan[1]
    edge = (state, next_state)
    attempts += 1
    if edge in ACTUALLY_BLOCKED:
        print(f"execute={state} -> {next_state} observation=blocked state={state}")
        known_blocked.add(edge)
        plan = make_plan(state, goal, known_blocked)
        version += 1
        replans += 1
        print(f"plan_v{version}=" + " -> ".join(plan))
        continue
    state = next_state
    plan = plan[1:]
    print(f"execute={edge[0]} -> {edge[1]} observation=ok state={state}")

print(f"final={state} replans={replans} attempts={attempts}")
```

## 5. 결과 해석: 계획과 실행의 실패를 나눠 본다

`plan_v1`과 `plan_v2`를 남기면 어떤 관찰 때문에 계획이 바뀌었는지 추적할 수 있다. `attempts=4`에는 실패한 `B→D`도 포함된다. 성공한 이동만 세면 환경과 상호작용한 비용을 실제보다 작게 보게 된다.

실행기는 계획을 믿고 상태를 미리 바꾸면 안 된다. 코드가 `blocked`일 때 `state=B`를 유지하는 이유다. 반대로 계획기는 실제 이동을 직접 수행하지 않는다. 현재 상태와 알려진 차단 정보를 받아 다음 후보를 계산할 뿐이다. [구조화 출력에서 필드와 자료형을 먼저 검증했던 방식]({{ '/posts/ai-agent-structured-output-validation/' | relative_url }})처럼 실제 시스템에서는 계획 단계도 `tool`, `arguments`, `expected_result` 같은 구조로 제한하면 실행 전에 검사하기 쉽다.

계획에 가능한 예외를 모두 미리 나열할 필요는 없다. 대신 실행기는 `ok`, `blocked`, `invalid_action`처럼 계획기가 처리할 수 있는 관찰 형식을 일정하게 반환해야 한다. 계획기가 알 수 없는 오류라면 임의로 성공 처리하지 말고 중단 상태로 남긴다. 이 계약이 있어야 같은 실패를 반복하지 않고 재계획 한도도 정확히 셀 수 있다.

2026년 11월 22일 현재 OpenAI Agents SDK 공식 문서는 오케스트레이션을 LLM이 다음 단계를 고르는 방식과 코드가 흐름을 정하는 방식으로 나눈다. 현재 실행 API는 `Runner.run()`, `Runner.run_sync()`, `Runner.run_streamed()`이며 도구 호출 결과를 입력에 붙여 루프를 계속한다. 이 글은 특정 SDK 없이 코드 오케스트레이션의 핵심만 재현했다. API 이름과 지원 범위는 바뀔 수 있으므로 실제 구현 시 공식 문서를 다시 확인해야 한다.

## 6. 직접 바꿔 보는 연습

먼저 `ACTUALLY_BLOCKED`에 `("B", "C")`도 추가해 보자. `B`에서 목표로 가는 모든 길이 막혀 `ValueError`가 발생한다. 이 예외를 잡아 `final=unreachable`로 종료하면 무한 재계획을 피할 수 있다.

다음에는 간선마다 비용을 붙이고 너비 우선 탐색을 다익스트라 알고리즘으로 바꿔 보자. 가장 적은 단계와 가장 싼 경로가 다를 수 있다. 마지막으로 `max_replans=2`를 두고 한도를 넘으면 사람에게 확인을 요청하도록 만들면 종료 조건과 승인 경계를 함께 연습할 수 있다.

## 핵심 요약

- 계획은 목표를 실행 가능한 행동 순서로 나누는 현재의 후보이며, 고정된 약속이 아니다.
- 실행기는 한 단계의 실제 결과를 관찰하고, 성공한 뒤에만 상태를 갱신한다.
- 실패 관찰을 기록한 뒤 현재 상태에서 남은 계획만 다시 만들면 이미 끝난 행동을 반복하지 않는다.
- 계획 버전, 실패 시도, 재계획 횟수와 종료 한도를 남겨야 실행을 검사할 수 있다.

## 다음 글 예고

다음 글에서는 같은 입력에 대한 여러 실행 결과를 기준표로 비교하는 **에이전트 평가**의 최소 구조를 구현한다.

## 참고 자료

1. [Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 추론, 행동, 환경 관찰을 교대로 연결하고 예외에 따라 계획을 갱신하는 원 논문.
2. [OpenAI Agents SDK, Agent orchestration](https://openai.github.io/openai-agents-python/multi_agent/) — LLM 중심과 코드 중심 오케스트레이션의 현재 공식 구분.
3. [OpenAI Agents SDK, Running agents](https://openai.github.io/openai-agents-python/running_agents/) — `Runner.run*` 실행 API와 도구 호출을 잇는 에이전트 루프의 공식 설명.
4. [Microsoft, AI Agents for Beginners: 계획 설계](https://microsoft.github.io/ai-agents-for-beginners/translations/ko/07-planning-design/) — 전체 목표를 관리 가능한 하위 작업으로 나누는 한국어 입문 자료.
