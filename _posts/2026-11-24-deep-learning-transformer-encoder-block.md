---
title: "딥러닝 입문 ⑩: Transformer 인코더 블록 손계산·Python 실습"
date: 2026-11-24 20:00:00 +0900
categories: [학습과 모델링]
tags: [딥러닝, Transformer, 인코더 블록, 잔차 연결, 층 정규화, 피드포워드, Python]
description: "Transformer 인코더 블록에서 잔차 연결, 층 정규화, 위치별 피드포워드 신경망이 이어지는 순서를 배웁니다. 두 토큰의 벡터가 블록 하나를 통과하는 과정을 손계산하고 순수 Python으로 검산합니다."
reading_time: 14
series: "처음 시작하는 딥러닝"
level: "입문"
math: true
---

> **이번 글의 목표:** 멀티헤드 셀프 어텐션 뒤에 잔차 연결·층 정규화·위치별 피드포워드 신경망을 붙여, Transformer 인코더 블록 하나의 입력과 출력을 계산한다.

첫 번째 토큰 벡터 `[1, 2, 3]`은 어텐션 출력 `[0.5, -0.5, 1.0]`을 받은 뒤 `[1.5, 1.5, 4.0]`이 된다. 이를 정규화하고 피드포워드 신경망과 두 번째 잔차 연결까지 통과시키면 블록 출력은 **`[-0.4319, -0.9503, 1.3822]`**다. 이번 글에서는 이 숫자가 만들어지는 순서를 하나씩 확인한다.

앞 글에서 [멀티헤드 어텐션과 위치 인코딩이 여러 관점의 정보와 순서를 더하는 과정]({{ '/posts/deep-learning-multi-head-attention-positional-encoding/' | relative_url }})을 계산했다. 이제 그 결과가 인코더 블록 안에서 어떻게 보존되고 변환되는지 이어서 살펴보자.

## 1. 직관: 새 정보를 더하되, 들어온 길을 남긴다

원래 Transformer 인코더의 한 층에는 두 부분층이 있다. 첫째는 토큰끼리 정보를 주고받는 멀티헤드 셀프 어텐션이고, 둘째는 각 토큰을 따로 변환하는 위치별 피드포워드 신경망이다. 각 부분층 바깥에는 입력을 출력에 더하는 잔차 연결과 그 결과를 정돈하는 층 정규화가 놓인다.

잔차 연결은 부분층이 만든 값을 입력의 대체품이 아니라 **수정량**처럼 더한다. 입력이 $x$이고 부분층 값이 $F(x)$라면 합은 $x+F(x)$다. 두 벡터의 차원은 같아야 한다. 원래 표현과 기울기가 덧셈 경로로 이어지는 이유는 [연쇄법칙으로 기울기를 전달했던 역전파 글]({{ '/posts/deep-learning-backpropagation-chain-rule/' | relative_url }})과 연결해서 볼 수 있다.

층 정규화는 한 토큰 벡터의 값들로 평균과 분산을 구한다. 다른 토큰 통계에 기대지 않고 각 토큰의 특징 차원에서 계산한다. 이번 손계산에서는 학습 가능한 $\gamma=1$, $\beta=0$으로 둔다.

## 2. 기호와 수식: 인코더의 두 Add & Norm

원 논문의 인코더는 부분층 뒤에 정규화를 두는 Post-LN 구조다. 어텐션 출력을 $A(x)$라고 하면 첫 단계는 다음과 같다.

$$
y=\mathrm{LayerNorm}(x+A(x))
$$

벡터 $z=(z_1,\ldots,z_d)$의 평균과 분산, 정규화 값은 다음처럼 계산한다.

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}z_i,\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(z_i-\mu)^2
$$

$$
\mathrm{LayerNorm}(z_i)=\frac{z_i-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

$\epsilon$은 분모가 0에 가까워지는 일을 막는 작은 수다. 이어지는 위치별 피드포워드 신경망은 같은 두 선형변환을 모든 토큰 위치에 독립적으로 적용한다.

$$
\mathrm{FFN}(y)=\max(0,yW_1+b_1)W_2+b_2
$$

중간의 $\max(0,\cdot)$는 ReLU다. [퍼셉트론과 ReLU 글에서 확인한 음수를 0으로 자르는 계산]({{ '/posts/deep-learning-perceptron-relu/' | relative_url }})이 여기에서도 그대로 쓰인다. FFN의 마지막 출력 차원을 입력 차원과 같게 만들고, 두 번째 잔차 연결과 정규화를 적용하면 블록이 완성된다.

$$
\mathrm{output}=\mathrm{LayerNorm}(y+\mathrm{FFN}(y))
$$

## 3. 작은 손계산: 첫 번째 토큰의 두 지름길

첫 토큰의 입력을 $x=[1,2,3]$, 어텐션 출력을 $A(x)=[0.5,-0.5,1.0]$이라고 하자. 첫 잔차 합은 `[1.5, 1.5, 4.0]`이다. 평균은 $7/3$, 분산은 약 $1.3889$이므로 정규화 결과는 `[-0.7071, -0.7071, 1.4142]`다.

이 벡터를 고정한 가중치의 FFN에 넣는다. 첫 선형층은 3차원을 4차원으로 넓히고 ReLU를 적용해 `[0.7071, 0, 0, 0.7071]`을 만든다. 두 번째 선형층이 다시 3차원으로 줄인 수정량은 `[0.3536, -0.3536, 0.7071]`이다.

두 번째 잔차 합은 정규화된 입력과 FFN 수정량을 더한 `[-0.3536, -1.0607, 2.1213]`이다. 여기에 층 정규화를 한 번 더 적용하면 최종 출력 `[-0.4319, -0.9503, 1.3822]`를 얻는다. 두 잔차 연결 모두 입력과 수정량의 차원이 같고, 두 정규화 결과는 평균이 거의 0이다.

<figure class="study-figure">
  <img src="{{ '/assets/images/deep-learning/transformer-encoder-block.svg' | relative_url }}" alt="Transformer 인코더 블록 계산도. 입력 벡터에 셀프 어텐션 출력을 더하고 층 정규화해 마이너스 0.7071, 마이너스 0.7071, 1.4142를 만든다. 이어 위치별 피드포워드 신경망의 수정량 0.3536, 마이너스 0.3536, 0.7071을 더하고 다시 정규화해 최종 출력 마이너스 0.4319, 마이너스 0.9503, 1.3822를 만든다.">
  <figcaption>그림 1. 인코더 블록은 셀프 어텐션과 위치별 FFN을 각각 잔차 연결과 층 정규화로 감싼다.</figcaption>
</figure>

## 4. 실행 가능한 Python 실습

다음 코드는 외부 패키지 없이 두 토큰이 같은 인코더 블록을 통과하는 계산을 재현한다. 어텐션 출력과 FFN 가중치는 학습하지 않고 고정해, 블록의 순서와 모양만 분리해서 본다.

```python
import math


def layer_norm(vector, eps=1e-5):
    mean = sum(vector) / len(vector)
    variance = sum((x - mean) ** 2 for x in vector) / len(vector)
    return [(x - mean) / math.sqrt(variance + eps) for x in vector]


def linear(vector, weights):
    columns = zip(*weights)
    return [sum(x * w for x, w in zip(vector, column))
            for column in columns]


def ffn(vector):
    w1 = [[1, 0, -1, 0.5],
          [0, 1, 1, -0.5],
          [1, -1, 0, 0.5]]
    w2 = [[0.5, 0, 0.5],
          [0, 0.5, -0.5],
          [0.5, 0.5, 0],
          [0, -0.5, 0.5]]
    hidden = [max(0, x) for x in linear(vector, w1)]
    return hidden, linear(hidden, w2)


inputs = [[1, 2, 3], [2, 0, 1]]
attention_outputs = [[0.5, -0.5, 1.0], [-0.5, 1.0, 0.5]]

for index, (x, attention) in enumerate(
        zip(inputs, attention_outputs), 1):
    residual1 = [a + b for a, b in zip(x, attention)]
    y = layer_norm(residual1)
    hidden, update = ffn(y)
    output = layer_norm([a + b for a, b in zip(y, update)])

    print(f"token {index} after norm 1:",
          [round(v, 4) for v in y])
    print(f"token {index} ffn hidden:",
          [round(v, 4) for v in hidden])
    print(f"token {index} block output:",
          [round(v, 4) for v in output])
```

실행 출력은 다음과 같다.

```text
token 1 after norm 1: [-0.7071, -0.7071, 1.4142]
token 1 ffn hidden: [0.7071, 0, 0, 0.7071]
token 1 block output: [-0.4319, -0.9503, 1.3822]
token 2 after norm 1: [0.707, -1.4141, 0.707]
token 2 ffn hidden: [1.4141, 0, 0, 1.4141]
token 2 block output: [0.508, -1.397, 0.889]
```

## 5. 결과 해석: 토큰끼리 섞는 단계와 토큰별 변환을 구분한다

셀프 어텐션은 한 토큰의 출력을 만들 때 다른 토큰의 값을 가중합할 수 있다. 반면 위치별 FFN은 토큰 1과 토큰 2에 **같은 가중치**를 쓰지만, 각 벡터를 서로 섞지 않고 따로 변환한다. 두 토큰의 FFN 은닉값과 최종 출력이 다른 이유는 입력 표현이 다르기 때문이다.

층 정규화 뒤의 벡터는 평균이 거의 0이고 분산이 거의 1이지만 서로 같아지지는 않는다. 정규화는 상대적인 값의 관계를 남긴 채 중심과 척도를 조정한다.

이번 계산은 원 논문의 Post-LN 순서를 따른다. 현대 구현에는 정규화를 부분층 앞에 두는 Pre-LN 변형도 있으므로 모든 Transformer가 완전히 같은 순서를 쓴다고 일반화하면 안 된다. PyTorch의 `TransformerEncoderLayer`도 `norm_first` 옵션으로 두 순서를 구분하며, 기본값은 정규화를 부분층 뒤에 두는 방식이다.

## 6. 직접 바꿔 보는 연습

첫째, `attention_outputs`를 모두 0으로 바꿔 첫 잔차 연결이 입력을 그대로 보존하는지 확인한다. 둘째, `w1`의 마지막 열을 모두 음수로 바꿔 ReLU가 네 번째 은닉값을 0으로 만드는지 살펴본다. 셋째, `layer_norm`의 분산 계산을 출력해 정규화 전후의 평균과 분산을 비교한다.

마지막으로 세 번째 토큰과 같은 차원의 어텐션 출력을 추가해 보자. FFN은 토큰 수가 늘어나도 각 위치에 같은 가중치를 적용한다.

## 핵심 요약

- Transformer 인코더 한 층은 멀티헤드 셀프 어텐션과 위치별 FFN이라는 두 부분층으로 구성된다.
- 원래 구조에서는 각 부분층의 입력과 출력을 더한 뒤 층 정규화를 적용한다.
- 잔차 연결을 하려면 입력과 부분층 출력의 차원이 같아야 한다.
- 층 정규화는 각 토큰 벡터의 특징 차원에서 평균과 분산을 계산한다.
- FFN은 모든 위치에 같은 가중치를 쓰지만 토큰마다 독립적으로 계산한다.

## 다음 글 예고

다음 글에서는 Transformer 디코더가 미래 토큰을 보지 못하게 하는 인과 마스크를 손계산하고, 마스크드 셀프 어텐션의 가중치가 어떻게 달라지는지 확인한다.

## 참고 자료

1. [Vaswani 외, Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 인코더의 두 부분층, 잔차 연결 뒤 층 정규화, 위치별 FFN을 제시한 원 논문.
2. [Dive into Deep Learning, The Transformer Architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) — AddNorm, 위치별 FFN, 인코더 블록 구현을 연결해 설명한 공개 교과서.
3. [PyTorch, TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) — 셀프 어텐션·FFN 구성과 `norm_first` 옵션을 확인한 공식 문서.
4. [PyTorch, LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) — 마지막 특징 차원의 평균·분산과 학습 가능한 affine 매개변수를 설명한 공식 문서.
