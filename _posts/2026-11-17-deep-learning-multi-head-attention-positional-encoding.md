---
title: "딥러닝 입문 ⑨: 멀티헤드 어텐션과 위치 인코딩 손계산·Python 실습"
date: 2026-11-17 20:00:00 +0900
categories: [학습과 모델링]
tags: [딥러닝, 멀티헤드 어텐션, 위치 인코딩, Transformer, Python]
description: "여러 어텐션 헤드가 서로 다른 표현을 모으는 과정과 사인·코사인 위치 인코딩이 순서를 더하는 원리를 배웁니다. 두 헤드의 출력과 4차원 위치 벡터를 손계산하고 순수 Python으로 검산합니다."
reading_time: 13
series: "처음 시작하는 딥러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 두 어텐션 헤드가 서로 다른 가중합을 만드는 과정을 계산하고, 위치 인코딩이 같은 토큰에도 서로 다른 순서 정보를 더하는 이유를 이해한다.

두 헤드를 계산했더니 첫째 출력은 **15.6405**, 둘째 출력은 **2.0**이 되었다. 둘을 이어 붙이면 `[15.6405, 2.0]`이다. 여기에 위치 0, 1, 2의 4차원 위치 벡터를 계산하면 각각 `[0, 1, 0, 1]`, `[0.8415, 0.5403, 0.01, 1.0]`, `[0.9093, -0.4161, 0.02, 0.9998]`이 된다.

앞 글에서는 [한 어텐션 헤드가 쿼리·키·값으로 가중합을 만드는 과정]({{ '/posts/deep-learning-attention-query-key-value/' | relative_url }})을 살펴봤다. 이번에는 그 계산을 여러 표현 공간에서 나란히 수행하는 멀티헤드 어텐션과, 어텐션만으로는 구분하기 어려운 순서를 입력에 더하는 위치 인코딩을 연결한다.

## 1. 직관: 여러 관점으로 보고, 각 자리에 좌표를 붙인다

한 헤드는 쿼리와 키를 한 가지 표현 공간에서 비교한다. 멀티헤드 어텐션은 입력을 헤드마다 다른 방식으로 투영한 뒤 어텐션을 병렬로 계산한다. 각 헤드는 독립된 쿼리·키·값 투영을 학습할 수 있으므로, 같은 입력에서도 서로 다른 가중치와 출력을 만들 여지가 생긴다.

이를 색과 모양을 함께 보는 상황에 비유할 수 있다. 한 관점은 색이 비슷한 항목을 모으고, 다른 관점은 모양이 비슷한 항목을 모은다. 두 결과를 이어 붙인 뒤 다시 선형변환하면 두 관점이 하나의 출력 표현으로 합쳐진다. 모든 헤드가 반드시 사람이 이름 붙일 수 있는 역할을 맡는다는 뜻은 아니다. 핵심은 서로 다른 학습 가능한 투영을 쓴다는 구조다.

하지만 셀프 어텐션 계산만 놓고 보면 입력의 배치 순서를 자동으로 보존하지 않는다. 같은 토큰 집합의 순서를 바꾸면 그 순서를 알려 주는 별도 신호가 필요하다. 원래 Transformer는 토큰 임베딩에 위치마다 다른 사인·코사인 벡터를 더했다. 토큰의 내용과 자리 좌표를 같은 차원의 벡터에 함께 담는 셈이다.

[토큰 ID가 의미 벡터로 바뀌는 임베딩 과정]({{ '/posts/generative-ai-embedding-cosine-similarity/' | relative_url }})을 먼저 확인하면, 왜 위치 벡터도 임베딩과 같은 차원이어야 더할 수 있는지 이해하기 쉽다.

## 2. 기호와 수식: 헤드를 합치고 위치를 더한다

헤드 $i$는 입력 $Q$, $K$, $V$에 서로 다른 학습 행렬을 곱한 뒤 어텐션을 계산한다.

$$
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$h$개 헤드의 출력은 이어 붙이고, 출력 행렬 $W^O$로 다시 변환한다.

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O
$$

이번 손계산에서는 흐름을 드러내기 위해 두 투영이 입력의 서로 다른 두 좌표를 고른다고 단순화한다. 마지막 $W^O$도 생략해 두 스칼라 출력을 그대로 이어 붙인다. 실제 모델에서는 투영 행렬과 $W^O$가 학습된다.

위치 $pos$, 모델 차원 $d$, 차원 쌍의 번호 $i$에 대한 원 논문의 고정 위치 인코딩은 다음과 같다.

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

짝수 차원에는 사인, 홀수 차원에는 코사인을 쓰고 차원이 높아질수록 더 느리게 변하는 주기를 사용한다. 위치 벡터 $PE$는 토큰 임베딩 $X$와 모양이 같아야 하며, 어텐션에 들어가기 전에 $X+PE$로 더한다.

## 3. 작은 손계산: 두 헤드와 세 위치

첫째 헤드는 앞 글과 같은 쿼리 `[1, 0]`으로 키 `[1, 0]`, `[0, 1]`, `[-1, 0]`을 비교한다. 가중치 `[0.5760, 0.2840, 0.1400]`으로 값 `[10, 20, 30]`을 섞으면 출력은 15.6405다.

둘째 헤드도 쿼리는 `[1, 0]`이지만 다른 투영을 거친 키 `[1, 0]`, `[0, 1]`, `[1, 0]`과 값 `[1, 2, 3]`을 사용한다고 하자. 가중치는 `[0.4011, 0.1978, 0.4011]`이고 출력은 다음과 같다.

$$
0.4011\times1+0.1978\times2+0.4011\times3=2.0
$$

두 헤드를 이어 붙인 결과는 `[15.6405, 2.0]`이다. 첫째 헤드는 첫 위치에 가장 큰 비중을 주고, 둘째 헤드는 첫 위치와 셋째 위치를 같은 비율로 본다. 이것이 서로 다른 투영이 서로 다른 정보 조합을 만들 수 있다는 작은 예다.

$d=4$일 때 위치 0의 벡터는 `[sin(0), cos(0), sin(0), cos(0)]`, 즉 `[0, 1, 0, 1]`이다. 위치 1은 약 `[0.8415, 0.5403, 0.0100, 1.0000]`이다. 같은 토큰 임베딩이라도 이 벡터를 더하면 위치 0과 위치 1에서 어텐션에 들어가는 표현이 달라진다.

<figure class="study-figure">
  <img src="{{ '/assets/images/deep-learning/multi-head-attention-positional-encoding.svg' | relative_url }}" alt="두 어텐션 헤드와 위치 인코딩 계산도. 첫째 헤드는 가중치 0.5760, 0.2840, 0.1400으로 출력 15.6405를 만들고 둘째 헤드는 가중치 0.4011, 0.1978, 0.4011로 출력 2.0을 만든다. 두 출력을 15.6405, 2.0으로 이어 붙이고, 아래에는 위치 0과 1의 4차원 사인 코사인 벡터가 토큰 임베딩에 더해지는 흐름이 표시된다.">
  <figcaption>그림 1. 헤드마다 다른 가중합을 만든 뒤 합치고, 입력에는 위치마다 다른 좌표를 더한다.</figcaption>
</figure>

## 4. 실행 가능한 Python 실습

다음 코드는 외부 패키지 없이 두 헤드와 4차원 위치 인코딩을 계산한다. 학습 과정 대신 이미 정한 투영 결과를 넣어 순전파의 두 핵심만 분리했다.

```python
import math


def softmax(scores):
    largest = max(scores)
    exps = [math.exp(score - largest) for score in scores]
    return [value / sum(exps) for value in exps]


def attention_head(query, keys, values):
    scale = math.sqrt(len(query))
    scores = [sum(q * k for q, k in zip(query, key)) / scale
              for key in keys]
    weights = softmax(scores)
    output = sum(weight * value for weight, value in zip(weights, values))
    return scores, weights, output


def positional_encoding(position, dimension=4):
    vector = []
    for index in range(0, dimension, 2):
        angle = position / (10000 ** (index / dimension))
        vector.extend([math.sin(angle), math.cos(angle)])
    return vector


heads = [
    ([1, 0], [[1, 0], [0, 1], [-1, 0]], [10, 20, 30]),
    ([1, 0], [[1, 0], [0, 1], [1, 0]], [1, 2, 3]),
]

outputs = []
for number, arguments in enumerate(heads, 1):
    scores, weights, output = attention_head(*arguments)
    outputs.append(output)
    print(f"head {number} weights:", [round(x, 4) for x in weights])
    print(f"head {number} output:", round(output, 4))

print("concatenated:", [round(x, 4) for x in outputs])
for position in range(3):
    vector = positional_encoding(position)
    print(f"position {position}:", [round(x, 4) for x in vector])
```

실행 출력은 다음과 같다.

```text
head 1 weights: [0.576, 0.284, 0.14]
head 1 output: 15.6405
head 2 weights: [0.4011, 0.1978, 0.4011]
head 2 output: 2.0
concatenated: [15.6405, 2.0]
position 0: [0.0, 1.0, 0.0, 1.0]
position 1: [0.8415, 0.5403, 0.01, 1.0]
position 2: [0.9093, -0.4161, 0.02, 0.9998]
```

## 5. 결과 해석: 헤드 수와 순서 정보는 다른 문제다

첫째 헤드와 둘째 헤드의 가중치가 다른 이유는 서로 다른 키 표현을 사용했기 때문이다. 헤드를 늘린다는 것은 똑같은 가중합을 복제하는 일이 아니라, 서로 다른 투영에서 여러 어텐션 결과를 만들고 합칠 통로를 늘리는 일이다. 다만 실제 학습 결과에서 모든 헤드가 완전히 다른 역할을 맡는다고 보장되지는 않는다.

위치 벡터는 헤드와 별개의 문제를 해결한다. 위치 0과 1은 첫 두 좌표에서 크게 다르고, 뒤의 좌표에서는 천천히 변한다. 여러 주기의 사인·코사인이 함께 있으므로 각 위치가 하나의 숫자가 아니라 여러 속도의 좌표로 표현된다. 이 벡터를 같은 토큰 임베딩에 더하면 토큰 내용은 같아도 자리 정보가 다른 입력이 된다.

멀티헤드 어텐션은 “어디에서 어떤 정보를 모을까”를 여러 표현 공간에서 계산하고, 위치 인코딩은 “그 정보가 어느 자리에 있었나”를 입력에 알려 준다. 두 장치를 결합해야 순환 연산 없이도 내용 관계와 순서를 함께 다룰 기반이 생긴다. [어텐션부터 다음 토큰 출력까지의 전체 추론 흐름]({{ '/posts/generative-ai-transformer-inference/' | relative_url }})은 이번 계산이 Transformer 블록 안에서 어디에 놓이는지 확인할 때 이어서 볼 수 있다.

## 6. 직접 바꿔 보는 연습

둘째 헤드의 세 번째 키를 `[-1, 0]`으로 바꾸고 가중치와 출력이 첫째 헤드에 가까워지는지 확인한다. 이어서 둘째 헤드의 값만 `[3, 2, 1]`로 뒤집어, 키가 정한 가중치와 값이 전달하는 내용의 역할을 다시 구분한다.

위치 인코딩에서는 `range(3)`을 `range(6)`으로 바꿔 각 차원이 얼마나 빠르게 변하는지 비교한다. 마지막으로 동일한 임베딩 `[0.5, 0.5, 0.5, 0.5]`에 위치 0과 위치 1 벡터를 각각 더해 결과가 어떻게 달라지는지 출력한다.

## 핵심 요약

- 멀티헤드 어텐션은 헤드마다 다른 학습 가능한 쿼리·키·값 투영을 사용한다.
- 각 헤드의 가중합을 이어 붙인 뒤 출력 선형변환으로 결합한다.
- 셀프 어텐션만으로는 순서를 직접 보존하지 않으므로 위치 정보가 필요하다.
- 사인·코사인 위치 인코딩은 위치마다 다른 벡터를 만들어 토큰 임베딩에 더한다.
- 헤드는 정보 조합의 관점을 늘리고, 위치 인코딩은 입력의 자리 좌표를 제공한다.

## 다음 글 예고

다음 글에서는 멀티헤드 셀프 어텐션을 잔차 연결, 층 정규화, 위치별 피드포워드 신경망과 묶어 Transformer 인코더 블록 하나를 완성한다.

## 참고 자료

1. [Vaswani 외, Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 멀티헤드 어텐션과 사인·코사인 위치 인코딩을 제안한 원 논문.
2. [Dive into Deep Learning, Multi-Head Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html) — 여러 투영 공간의 헤드와 출력 결합을 설명한 공개 교과서.
3. [Dive into Deep Learning, Self-Attention and Positional Encoding](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html) — 셀프 어텐션의 순서 한계와 위치 인코딩 식을 설명한 공개 교과서.
