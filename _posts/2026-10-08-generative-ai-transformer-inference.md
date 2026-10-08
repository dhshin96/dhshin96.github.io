---
title: "생성형 AI 트랜스포머 추론 입문: 어텐션에서 다음 토큰까지"
date: 2026-10-08 20:00:00 +0900
categories: [학습과 모델링, 지능형 시스템]
tags: [생성형-AI, 트랜스포머, 셀프-어텐션, 소프트맥스, Python]
description: "트랜스포머가 현재 토큰의 질문으로 앞선 토큰을 모으고 다음 토큰 확률을 만드는 흐름을 배웁니다. 작은 벡터의 어텐션 가중치와 소프트맥스 확률을 손계산과 순수 Python으로 재현합니다."
reading_time: 11
series: "처음 시작하는 생성형 AI"
level: "입문"
math: true
---

> **이번 글의 목표:** 한 개의 어텐션 헤드가 현재 위치에 필요한 정보를 모으는 계산과, 그 결과가 다음 토큰 확률로 바뀌는 과정을 연결한다.

먼저 결과부터 보자. 세 위치의 점수가 `[0.707, 0.707, 1.414]`이면 소프트맥스가 만든 어텐션 가중치는 `[0.248, 0.248, 0.503]`이다. 이 비율로 값 벡터를 섞으면 문맥 벡터는 `[0.752, 0.752]`가 된다. 마지막 선형변환과 소프트맥스를 거친 다음 토큰 확률은 `를: 0.243`, `는: 0.243`, `다: 0.515`다. 탐욕 선택을 쓰면 `다`가 한 토큰 생성된다.

입력이 아직 숫자 벡터가 되는 과정이 낯설다면 [토큰 ID를 임베딩 벡터로 바꾸는 앞 글]({{ '/posts/generative-ai-embedding-cosine-similarity/' | relative_url }})을 먼저 읽자. 이번 글은 그 벡터에 위치 정보가 더해져 Transformer 층으로 들어간 다음부터 시작한다.

## 1. 직관: 현재 토큰이 과거에서 필요한 정보를 모은다

텍스트를 생성하는 디코더형 Transformer는 지금까지 주어진 토큰을 바탕으로 다음 토큰 하나의 분포를 만든다. 현재 위치의 벡터는 **질의(query)**가 되고, 읽을 수 있는 각 위치는 **키(key)**와 **값(value)**을 제공한다. 질의와 키가 잘 맞을수록 그 위치의 값이 더 큰 비중으로 합쳐진다.

질의는 “무엇을 찾을까”, 키는 “어떤 정보와 잘 맞을까”, 값은 “무엇을 전달할까”에 해당한다. 실제 모델은 같은 입력에 서로 다른 가중치 행렬을 곱해 세 벡터를 만든다. 여러 헤드와 층, 잔차 연결, 정규화, 피드포워드 층도 사용하지만 이번에는 **한 헤드의 어텐션과 다음 토큰 선택**만 남겨 계산한다.

<figure class="study-figure">
  <img src="{{ '/assets/images/generative-ai/transformer-inference-next-token.svg' | relative_url }}" alt="토큰 벡터가 마스킹된 셀프 어텐션에서 가중합되어 문맥 벡터가 되고 선형변환과 소프트맥스를 거쳐 다음 토큰 다를 선택하는 흐름">
  <figcaption>그림 1. Transformer 추론은 읽을 수 있는 이전 위치를 가중합하고, 마지막 위치의 표현을 어휘 전체의 확률로 바꿔 한 토큰씩 이어 붙인다.</figcaption>
</figure>

생성 중인 위치는 미래 토큰을 미리 보면 안 된다. 그래서 **인과 마스크(causal mask)**로 오른쪽의 미래 위치를 가린다. 아직 생성하지 않은 다음 위치는 계산에 사용할 수 없다.

## 2. 기호와 수식: 점수, 가중치, 문맥 벡터

한 질의 벡터를 $\mathbf q$, 각 위치의 키와 값을 $\mathbf k_i$, $\mathbf v_i$, 키의 차원을 $d_k$라고 하자. 스케일드 닷프로덕트 어텐션은 다음 순서로 계산한다.

$$
s_i=\frac{\mathbf q\cdot\mathbf k_i}{\sqrt{d_k}},
\qquad
\alpha_i=\frac{e^{s_i}}{\sum_j e^{s_j}},
\qquad
\mathbf c=\sum_i\alpha_i\mathbf v_i
$$

$s_i$는 관련성 점수, $\alpha_i$는 합이 1인 어텐션 가중치, $\mathbf c$는 값 벡터의 가중합인 문맥 벡터다. $\sqrt{d_k}$로 나누는 스케일은 차원이 커질 때 내적의 크기가 지나치게 커지는 것을 완화한다. 행렬로 여러 위치를 한꺼번에 계산하면 익숙한 $\operatorname{softmax}(QK^T/\sqrt{d_k})V$가 된다.

문맥을 포함한 마지막 위치의 은닉 벡터를 $\mathbf h$라 하면 출력층은 어휘 크기만큼의 로짓을 만든다. 로짓 $z_t$를 다시 소프트맥스에 넣으면 다음 토큰 $t$의 확률이 된다.

$$
\mathbf z=W_{\text{out}}\mathbf h+\mathbf b,
\qquad
P(t\mid x_{1:n})=\frac{e^{z_t}}{\sum_u e^{z_u}}
$$

## 3. 작은 손계산: 세 위치를 모아 `다`를 고르기

$\mathbf q=(1,1)$이고 키를 $(1,0)$, $(0,1)$, $(1,1)$로 두자. $d_k=2$이므로 세 점수는 각각 $1/\sqrt2\approx0.707$, $0.707$, $2/\sqrt2\approx1.414$다. 소프트맥스를 적용하면 가중치는 약 $(0.248,0.248,0.503)$이다. 세 번째 위치가 가장 잘 맞지만 앞의 두 위치도 버리지 않는다.

값도 $(1,0)$, $(0,1)$, $(1,1)$로 두면 문맥 벡터는 다음과 같다.

$$
\mathbf c=0.248(1,0)+0.248(0,1)+0.503(1,1)
\approx(0.752,0.752)
$$

설명용 출력 가중치가 `를`에는 $(1,0)$, `는`에는 $(0,1)$, `다`에는 $(1,1)$을 대응시킨다고 하자. 로짓은 $(0.752,0.752,1.503)$이고 소프트맥스 확률은 약 $(0.243,0.243,0.515)$다. 이 숫자는 원리를 확인하려고 정한 모형의 결과이지 실제 한국어 모델의 확률이 아니다.

## 4. 실행 가능한 Python 실습

다음 코드는 외부 패키지 없이 두 번의 소프트맥스를 재현한다. 큰 로짓에서도 지수 계산이 넘치지 않도록 최댓값을 먼저 뺀다.

```python
from math import exp, sqrt


def softmax(values):
    largest = max(values)
    shifted = [exp(value - largest) for value in values]
    total = sum(shifted)
    return [value / total for value in shifted]


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


query = [1.0, 1.0]
keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
values = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

scores = [dot(query, key) / sqrt(len(query)) for key in keys]
attention = softmax(scores)
context = [
    sum(weight * value[d] for weight, value in zip(attention, values))
    for d in range(2)
]

vocabulary = ["를", "는", "다"]
output_weights = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
logits = [dot(context, row) for row in output_weights]
probabilities = softmax(logits)
next_token = vocabulary[max(range(len(logits)), key=logits.__getitem__)]

print("점수:", [round(x, 3) for x in scores])
print("어텐션:", [round(x, 3) for x in attention])
print("문맥:", [round(x, 3) for x in context])
print("다음 토큰 확률:", {
    token: round(probability, 3)
    for token, probability in zip(vocabulary, probabilities)
})
print("탐욕 선택:", next_token)
```

실행 출력은 다음과 같다.

```text
점수: [0.707, 0.707, 1.414]
어텐션: [0.248, 0.248, 0.503]
문맥: [0.752, 0.752]
다음 토큰 확률: {'를': 0.243, '는': 0.243, '다': 0.515}
탐욕 선택: 다
```

## 5. 결과 해석: 확률 계산과 선택 규칙은 다르다

어텐션의 첫 소프트맥스는 **입력 위치들 사이의 비중**을 만든다. 출력층의 두 번째 소프트맥스는 **어휘 항목들 사이의 다음 토큰 확률**을 만든다. 둘 다 합이 1이지만 답하는 질문이 다르다. 어텐션 가중치 하나를 그대로 다음 토큰 확률로 읽으면 안 된다.

코드는 가장 큰 로짓의 토큰을 고르는 탐욕 선택을 사용했다. 현재 Hugging Face Transformers 공식 문서도 탐욕 탐색을 매 단계 가장 가능성 높은 토큰을 고르는 방식으로 설명하고, 샘플링은 어휘 확률분포에 따라 무작위로 고르는 방식으로 구분한다. 어떤 선택 규칙을 쓰더라도 모델이 먼저 로짓을 계산하는 단계는 필요하다.

선택한 `다`를 입력 뒤에 붙이면 같은 계산이 새 마지막 위치에서 반복된다. 이처럼 이미 생성한 토큰을 다음 단계의 조건으로 다시 쓰는 방식을 **자기회귀 생성**이라고 한다. 앞서 [BPE 토큰화로 문장을 ID 열로 만든 과정]({{ '/posts/generative-ai-tokenization-bpe/' | relative_url }})과 연결하면, 생성은 “ID 열 입력 → 다음 ID 선택 → ID 열에 추가”의 반복으로 볼 수 있다.

## 6. 직접 바꿔 보는 연습

먼저 질의를 `[1.0, 0.0]`으로 바꿔 첫 번째 키의 가중치가 어떻게 변하는지 확인해 보자. 다음으로 세 번째 값 벡터만 `[2.0, 0.0]`으로 바꿔 보자. 어텐션 가중치는 그대로지만 문맥 벡터와 다음 토큰 확률은 달라진다. **키는 선택 비중에, 값은 전달 내용에 관여한다**는 차이를 볼 수 있다.

마지막으로 `next_token` 선택을 확률 샘플링으로 바꾸기 전에 난수 시드를 고정해 보자. 같은 확률분포에서도 선택 규칙에 따라 출력이 달라질 수 있으며, 확률이 가장 높은 토큰이 언제나 뽑히는 것은 아님을 확인할 수 있다.

## 핵심 요약

- 셀프 어텐션은 질의와 키의 점수로 값 벡터를 가중합해 현재 위치의 문맥을 만든다.
- 인과 마스크는 생성되지 않은 미래 토큰을 현재 계산에서 보지 못하게 한다.
- 출력층은 마지막 위치의 표현을 어휘별 로짓으로 바꾸고, 소프트맥스는 이를 다음 토큰 확률로 정규화한다.
- 자기회귀 생성은 선택한 토큰을 입력에 붙이고 같은 추론을 반복한다.

## 다음 글 예고

다음 글에서는 같은 질문도 지시, 입력, 제약, 출력 형식을 어떻게 나누느냐에 따라 달라지는 **프롬프트 구조**를 살펴본다. 모호한 요청을 네 부분으로 분해하고 결과를 비교한다.

## 참고 자료

1. Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) — Transformer와 스케일드 닷프로덕트 어텐션을 제안한 원 논문.
2. PyTorch, [`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html) — $QK^T/\sqrt{d_k}$, 인과 마스크, 현재 공식 함수 시그니처를 설명한 문서(2026-10-08 확인).
3. Hugging Face Transformers, [Generation strategies](https://huggingface.co/docs/transformers/generation_strategies) — 탐욕 탐색과 확률 샘플링을 구분한 현재 공식 문서(2026-10-08 확인).
