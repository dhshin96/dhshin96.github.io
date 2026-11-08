---
title: "AI 에이전트 메모리 입문: 작업 상태와 장기 기억 Python 실습"
date: 2026-11-08 18:05:00 +0900
categories: [지능형 시스템, 개발과 제품화]
tags: [AI-에이전트, 메모리, 작업-상태, 장기-기억, Python]
description: "AI 에이전트의 현재 작업 상태와 실행을 넘어 남는 장기 기억을 구분합니다. 순수 Python으로 선호 2건을 저장하고 다음 실행에서 관련 기억 1건을 회수합니다."
reading_time: 11
series: "처음 시작하는 AI 에이전트 개발"
level: "입문"
math: true
---

> **이번 글의 목표:** 에이전트의 **작업 상태**와 **장기 기억**을 수명과 범위로 구분하고, 무엇을 저장하고 다음 실행에서 무엇을 꺼낼지 결정하는 최소 메모리를 Python으로 구현한다.

먼저 결과부터 보자. 첫 실행은 현재 목표와 두 단계를 작업 상태에 두고, 선호 두 건을 JSON 파일에 저장한다. 작업 상태를 지운 뒤 시작한 두 번째 실행은 질문 태그와 완전히 맞는 `Python 예제를 선호함` 한 건만 회수한다.

```text
run1 working_steps=2 saved=2
run2 recalled=1 top=Python 예제를 선호함
run2 top_score=1.0 working_goal=선호를 반영해 설명
after_run memory_file_exists=True
```

앞 글에서 [구조화 출력의 필드와 자료형을 검증하는 방법]({{ '/posts/ai-agent-structured-output-validation/' | relative_url }})을 배웠다. 이번에는 검증된 정보 중 일부를 실행 뒤에도 남기고, 새 요청에 필요한 것만 다시 작업 상태로 가져온다.

## 1. 직관: 책상과 보관함은 역할이 다르다

작업 상태는 지금 문제를 푸는 책상과 같다. 현재 목표, 방금 받은 도구 결과, 다음 단계처럼 이번 실행에 바로 필요한 정보를 펼쳐 둔다. [에이전트 루프가 단계별 관찰을 상태에 쌓았던 과정]({{ '/posts/ai-agent-loop-state-python/' | relative_url }})이 작업 상태의 예다. 실행이 끝나면 이 상태를 버리거나 체크포인트로 보존할 수 있지만, 모든 항목을 영구 기억으로 승격할 이유는 없다.

장기 기억은 실행을 넘어 다시 읽을 수 있는 보관함이다. 사용자 선호, 확인된 사실, 지난 작업의 결과처럼 다음 세션에도 쓸 정보가 대상이다. 중요한 차이는 저장 위치보다 **수명과 범위**다. 같은 Python 딕셔너리라도 현재 실행에서만 쓰면 작업 상태이고, 사용자 범위의 저장소에 남겨 다음 실행에서 읽으면 장기 기억이다.

<figure class="study-figure">
  <img src="{{ '/assets/images/ai-agents/agent-memory-working-long-term.svg' | relative_url }}" alt="첫 실행의 작업 상태에서 저장 기준을 통과한 정보만 장기 기억에 기록하고, 다음 실행에서 질문과 관련된 기억만 회수해 새 작업 상태에 넣는 흐름">
  <figcaption>그림 1. 메모리는 전부 저장하는 기능이 아니라 저장과 회수의 두 경계를 설계하는 일이다.</figcaption>
</figure>

## 2. 기호와 수식: 관련된 기억만 회수한다

새 요청의 태그 집합을 $Q$, 기억 $m$의 태그 집합을 $T_m$이라고 하자. 이번 예제는 질문 태그 중 기억과 겹치는 비율을 관련도 점수로 쓴다.

$$
R(Q,m)=\frac{|Q\cap T_m|}{|Q|}
$$

$R\ge\tau$인 기억만 회수하며 임계값은 $\tau=0.5$로 둔다. 실제 시스템은 임베딩 유사도, 최신성, 중요도 등을 함께 쓸 수 있지만, 여기서는 선택 기준과 결과를 눈으로 검산할 수 있는 집합 연산만 사용한다. 점수 계산은 기억을 **찾는** 규칙이고, 저장 여부는 별도 규칙이어야 한다. 일시적인 질문이나 비밀값을 관련 있다는 이유만으로 남겨서는 안 된다.

## 3. 작은 손계산: 두 기억 중 하나를 고른다

질문 태그를 $Q=\{python, 예제\}$로 둔다. 첫 번째 기억의 태그는 $T_1=\{python, 예제\}$이므로 교집합 크기는 2이고 $R(Q,m_1)=2/2=1.0$이다. 두 번째 기억은 $T_2=\{설명, 길이\}$라서 교집합이 없고 점수는 0이다.

| 기억 | 겹친 태그 수 | 관련도 | 회수 여부 |
| --- | ---: | ---: | --- |
| Python 예제를 선호함 | 2 | 1.0 | 회수 |
| 짧은 설명을 선호함 | 0 | 0.0 | 제외 |

따라서 임계값 0.5를 넘는 기억은 한 건이다. 이것은 의미를 이해하는 범용 검색 점수가 아니라, 고정된 두 입력에서 회수 경계를 확인하기 위한 결정적 예제다.

## 4. 실행 가능한 Python 실습

아래 코드는 표준 라이브러리만 사용한다. 첫 실행이 JSON 파일에 장기 기억을 쓴 뒤 작업 상태를 삭제하고, 두 번째 실행이 같은 파일을 새로 읽는다. 마지막에는 반복 실행 결과를 같게 만들기 위해 실습 파일을 지운다.

```python
import json
from pathlib import Path

MEMORY_PATH = Path("agent_memory_demo.json")


def save_memory(path, memories):
    path.write_text(
        json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def recall(path, query_tags, threshold=0.5):
    memories = json.loads(path.read_text(encoding="utf-8"))
    query = set(query_tags)
    ranked = []
    for item in memories:
        score = len(query & set(item["tags"])) / len(query)
        if score >= threshold:
            ranked.append((score, item["text"]))
    return sorted(ranked, key=lambda row: (-row[0], row[1]))


if MEMORY_PATH.exists():
    MEMORY_PATH.unlink()

working_state = {"goal": "사용자 선호 저장", "steps": ["관찰", "선별"]}
long_term = [
    {"text": "Python 예제를 선호함", "tags": ["python", "예제"]},
    {"text": "짧은 설명을 선호함", "tags": ["설명", "길이"]},
]
save_memory(MEMORY_PATH, long_term)
print(f"run1 working_steps={len(working_state['steps'])} saved={len(long_term)}")

del working_state
recalled = recall(MEMORY_PATH, ["python", "예제"])
new_working_state = {"goal": "선호를 반영해 설명", "memories": recalled}
print(f"run2 recalled={len(recalled)} top={recalled[0][1]}")
print(
    f"run2 top_score={recalled[0][0]:.1f} "
    f"working_goal={new_working_state['goal']}"
)
print(f"after_run memory_file_exists={MEMORY_PATH.exists()}")
MEMORY_PATH.unlink()
```

## 5. 결과 해석: 저장과 회수에 각각 문이 필요하다

`del working_state` 뒤에도 JSON 파일은 남는다. 그래서 두 번째 실행은 첫 실행의 Python 변수 없이도 선호를 회수한다. 반대로 파일에 있는 두 기억을 모두 작업 상태에 넣지 않고, 현재 질문과 관련된 한 건만 가져온다. 장기 저장과 현재 맥락 주입은 같은 동작이 아니다.

2026년 11월 8일 현재 OpenAI Agents SDK 공식 문서의 Sessions는 한 세션의 대화 이력을 실행 전 읽고 실행 후 새 항목을 저장한다. 로컬 개발용 `SQLiteSession`과 가져올 이력 수를 제한하는 `SessionSettings(limit=N)`도 제공한다. 이는 대화 이력을 유지하는 구현 선택지다. 사용자별 사실을 여러 세션에서 공유하는 장기 기억은 범위, 저장 규칙, 회수 규칙을 애플리케이션에서 따로 설계해야 한다.

저장 문은 “나중에도 유용한가, 사용자가 저장을 기대하거나 동의했는가, 출처와 갱신 시점을 기록할 수 있는가”를 검사해야 한다. 비밀번호·인증 토큰 같은 비밀값, 한 번뿐인 요청, 검증되지 않은 추측은 기본적으로 저장하지 않는다. 회수 문은 사용자와 세션의 소유권을 먼저 확인하고 관련도·최신성·개수 제한을 적용한다. 메모리가 있다고 해서 모델 입력에 자동으로 넣어도 되는 것은 아니다.

기억은 틀리거나 낡을 수 있다. 따라서 실제 저장 구조에는 값뿐 아니라 `source`, `created_at`, `updated_at`, `user_id` 같은 메타데이터와 수정·삭제 경로가 필요하다. 이번 예제는 점수 원리만 분리해 보기 위해 그 운영 요소를 생략했다. 파일 전체를 프롬프트에 붙이는 방식은 기억이 늘수록 비용과 잡음이 함께 커지므로, 필요한 일부만 선택하는 경계를 처음부터 두는 편이 낫다.

## 6. 직접 바꿔 보는 연습

먼저 질문 태그를 `['설명', '예제']`로 바꿔 보자. 두 기억이 각각 한 태그씩 겹치므로 점수 0.5로 모두 회수된다. 그다음 임계값을 0.6으로 올리면 둘 다 제외된다. 작은 변화가 회수 개수에 어떻게 반영되는지 출력으로 확인할 수 있다.

다음에는 기억에 `user_id`를 추가하고, 다른 사용자의 기억은 점수 계산 전에 제외해 보자. 마지막으로 저장 후보에 `kind='temporary'`를 넣고 선호(`preference`)만 파일에 쓰는 `should_store` 함수를 만들어 보자. 관련도 계산보다 먼저 소유권과 저장 허용 범위를 검사하는 순서를 유지한다.

## 핵심 요약

- 작업 상태는 현재 실행의 목표와 관찰을 담고, 장기 기억은 실행을 넘어 다시 읽을 정보를 담는다.
- 저장할 정보와 회수할 정보는 서로 다른 기준으로 선별한다.
- 회수 점수는 현재 질문에 필요한 일부만 작업 상태로 가져오기 위한 경계다.
- 장기 기억에는 소유권, 동의, 출처, 갱신·삭제 규칙이 함께 필요하다.

## 다음 글 예고

다음 글에서는 장기 기억과 외부 문서 검색의 차이를 구분하고, 질문과 관련된 근거 조각을 찾아 답변 입력에 결합하는 **검색 결합**의 최소 구조를 구현한다.

## 참고 자료

1. [OpenAI Agents SDK, Sessions](https://openai.github.io/openai-agents-python/sessions/) — 세션 이력의 저장·회수와 `SQLiteSession`, `SessionSettings`.
2. [LangGraph, Memory overview](https://docs.langchain.com/oss/python/concepts/memory) — thread 범위의 단기 메모리와 세션을 넘는 장기 메모리 구분.
3. [Sumers et al., Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427) — 작업 기억, 장기 기억, 회수와 학습 행동을 나눈 CoALA 구조.
