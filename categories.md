---
layout: default
title: 카테고리
permalink: /categories/
---
<section class="page-hero container">
  <p class="eyebrow">CATEGORIES</p>
  <h1>주제별 기록</h1>
  <p>기초 이론부터 현장 적용과 커리어 회고까지, 서로 연결되는 지식을 주제별로 정리합니다.</p>
</section>

<section class="container category-sections">
{% assign category_list = "수학|개념 정리 · 선형대수 · 미적분 · 최적화,통계|확률 · 추론 · 실험 설계 · 시계열,머신러닝|지도/비지도 학습 · 피처 엔지니어링 · MLOps,딥러닝|신경망 · 비전 · 자연어 처리 · 모델 구현,AI|생성형 AI · LLM · RAG · 평가,AI Agent|에이전트 설계 · 도구 사용 · 워크플로 · 자동화,바이브코딩|프로토타이핑 · 개발 도구 · 제품화 · 회고,제조업|공정 최적화 · 예지보전 · 품질 · 비전 검사,커리어·프로젝트|개인 프로젝트 · 회사 경험 · 포트폴리오 · 이직 준비" | split: "," %}
{% for item in category_list %}
  {% assign c = item | split: "|" %}
  <article class="category-block" id="{{ c[0] | slugify }}">
    <div class="category-title"><span>{{ forloop.index | prepend: '0' }}</span><div><h2>{{ c[0] }}</h2><p>{{ c[1] }}</p></div></div>
    <div class="category-posts">
      {% assign matched = false %}
      {% for post in site.posts %}
        {% if post.categories contains c[0] %}
          {% assign matched = true %}
          <a href="{{ post.url | relative_url }}"><time>{{ post.date | date: '%Y.%m.%d' }}</time><strong>{{ post.title }}</strong><span>→</span></a>
        {% endif %}
      {% endfor %}
      {% unless matched %}<p class="coming-soon">기록을 준비하고 있습니다.</p>{% endunless %}
    </div>
  </article>
{% endfor %}
</section>

