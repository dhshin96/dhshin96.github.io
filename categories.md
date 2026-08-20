---
layout: default
title: 카테고리
permalink: /categories/
---
<section class="page-hero container">
  <p class="eyebrow">CATEGORIES</p>
  <h1>지식의 지도</h1>
  <p>기초를 이해하고, 모델을 만들고, 시스템으로 구현해 현장의 문제를 푸는 흐름으로 기록을 구성합니다.</p>
</section>

<section class="container category-sections">
{% for category in site.data.taxonomy %}
  <article class="category-block" id="{{ category.id }}">
    <div class="category-title">
      <span>{{ category.number }}</span>
      <div><h2>{{ category.name }}</h2><p>{{ category.description }}</p></div>
    </div>
    <div class="category-content">
      <div class="subtopic-grid">
        {% for child in category.children %}
        <div class="subtopic"><strong>{{ child.name }}</strong><p>{{ child.topics }}</p></div>
        {% endfor %}
      </div>
      <div class="category-posts">
        {% assign matched = false %}
        {% for post in site.posts %}
          {% if post.categories contains category.name %}
            {% assign matched = true %}
            <a href="{{ post.url | relative_url }}"><time>{{ post.date | date: '%Y.%m.%d' }}</time><strong>{{ post.title }}</strong><span>→</span></a>
          {% endif %}
        {% endfor %}
        {% unless matched %}<p class="coming-soon">이 분야의 기록을 준비하고 있습니다.</p>{% endunless %}
      </div>
    </div>
  </article>
{% endfor %}
</section>
