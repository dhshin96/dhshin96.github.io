# DH Shin · Tech Notes

`https://dhshin96.github.io`에 배포되는 개인 기술 블로그입니다.

## 글 작성

1. `_drafts/post-template.md` 또는 `_drafts/project-template.md`를 복사합니다.
2. 파일명을 `_posts/YYYY-MM-DD-english-slug.md` 형식으로 저장합니다.
3. Front Matter의 제목, 날짜, 카테고리를 수정합니다.
4. 로컬 미리보기 후 커밋하고 푸시합니다.

```powershell
bundle install
bundle exec jekyll serve
```

브라우저에서 `http://localhost:4000`을 열어 확인할 수 있습니다.

## 분류 체계

상위 카테고리는 아래 6개이며, 세부 주제는 `_data/taxonomy.yml`에서 관리합니다.

- `기반 지식`: 수리적 사고, 통계적 추론
- `학습과 모델링`: 머신러닝, 딥러닝, 실험과 평가
- `지능형 시스템`: 생성형 AI, 지식과 검색, 에이전트
- `개발과 제품화`: AI 엔지니어링, 프로토타이핑, 운영과 개선
- `산업 AI`: 품질·설비·공정 지능화
- `프로젝트와 성장`: 프로젝트 로그, 업무 사례, 커리어 노트

글의 `categories`에는 상위 카테고리를, `tags`에는 세부 주제를 넣습니다. 카테고리 이름은 위 목록과 정확히 일치해야 카테고리 페이지에 표시됩니다.
