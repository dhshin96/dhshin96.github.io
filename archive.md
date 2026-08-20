---
layout: default
title: 아카이브
permalink: /archive/
---
<section class="page-hero compact container">
  <p class="eyebrow">ARCHIVE</p>
  <h1>모든 기록</h1>
  <p>시간순으로 모아보는 공부와 프로젝트의 흔적입니다.</p>
</section>
<section class="container archive-list">
{% for post in site.posts %}
  {% assign current_year = post.date | date: '%Y' %}
  {% if current_year != previous_year %}<h2>{{ current_year }}</h2>{% assign previous_year = current_year %}{% endif %}
  <a href="{{ post.url | relative_url }}"><time>{{ post.date | date: '%m.%d' }}</time><strong>{{ post.title }}</strong><span>{{ post.categories | join: ' · ' }}</span></a>
{% else %}
  <div class="empty-state"><strong>아직 공개된 글이 없습니다.</strong><p>초안 템플릿을 활용해 첫 글을 작성해 보세요.</p></div>
{% endfor %}
</section>

