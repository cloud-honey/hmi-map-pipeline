# HMI Map Pipeline — Web UI 기획서

Date: 2026-05-16
Status: 기획 완료 → 구현 대기

---

## 1. 개요

마스터가 파이프라인을 **편하게 사용하고 피드백을 줄 수 있는 웹 대시보드**.
현재 boom-master-dashboard (React+Express, port 5173)에 통합하거나 독립 배포.

**핵심 목적:**
- DXF/DWG 파일 업로드 → 파이프라인 실행 → 결과 확인
- 실행 이력 관리 (타임라인, 로그)
- 피드백 제출 (좋은 결과 /失败的 결과 신고)
- 설정 변경 (AI 파라미터, QA阈值)

---

## 2. 주요 기능

### 2.1 업로드 + 실행

```
[파일 선택] → [Config 선택 or 생성] → [실행] → [결과 미리보기]
```

- **Drag & Drop** DXF/DWG/PNG/PDF 파일
- **Config 선택**: 저장된 설정 nebo 커스텀
- **실행 버튼**: 로컬 파이프라인 실행
- **실시간 로그**: 터미널 출력 스트리밍
- **결과 미리보기**: background.png 썸네일 + 메타데이터

### 2.2 실행 이력

- 타임라인 뷰 (최근 20개)
- 각 실행: 파일명, 실행 시간, 소요시간, QA 점수, 결과 Thumb
- 클릭 → 상세 보기 (로그 + 설정 + 출력)

### 2.3 피드백 시스템

```
결과 옆 [👍 좋은 결과] [👎 문제가 있었어요]
                ↓
        [어떤 문제가 있었나요?]
        - 벽이 잘못됨
        - 색상이 이상함
        - AI가 추가 객체를 생성함
        - 기타 (텍스트 입력)
                ↓
        DB에 저장 → 향후 학습 데이터로 활용
```

- 간단한_rating + 선택식 문제 유형
- 텍스트 코멘트 (선택)

### 2.4 설정 관리

- AI 리파인먼트 On/Off
- Denoise strength (0.0~0.5 slider)
- Tile size (512/768/1024)
- QA threshold values
- Config 저장/불러오기 (JSON)

### 2.5 결과 뷰어

- 확대/축소 가능한 ISO 결과 이미지
- Transparent overlay 토글
- Masks overlay 토글
- Anchors 좌표 표시

---

## 3. 페이지 구조

```
┌─────────────────────────────────────────────────────────┐
│ HMI Map Pipeline                          [설정] [로그] │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [파일 업로드 / Drag & Drop]          [실행 버튼]       │
│                                                         │
│  Config: [dropdown ▾]              [설정 편집]        │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  실행 결과 미리보기 (대형)                              │
│  background.png + 메타데이터 + QA 점수                  │
│                                                         │
│  [👍] [👎]  피드백 버튼                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘

하단 탭:
- [실행 기록] — 타임라인
- [피드백] — 제출된 피드백 목록
- [설정] — Config 편집
```

---

## 4. 기술 선택

### Stack (기존 boom-master-dashboard 충돌 방지)

| 옵션 | 장점 | 단점 |
|------|------|------|
| **독립 배포** (port 3001 등) | 충돌 없음, 독립 운영 | 별도 PM2 등록 필요 |
| **boom-dashboard 통합** (route 추가) | 관리简化 | 변경 위험, 마스터 컨펌 필요 |

**권장: 독립 배포** — `hmimap-web/` 프로젝트, port 3001

### Frontend
- **React** (기존 boom-dashboard와 동일)
- Vite로 빌드
- Tailwind CSS (boom-dashboard와 동일)

### Backend
- **Express** (booms-dashboard와 동일한 패턴)
- Python subprocess로 파이프라인 실행
- SQLite로 실행 이력 + 피드백 저장 (simple, 파일 하나로 관리)

### 데이터 저장
- **SQLite**: 실행 기록, 피드백, 설정
- 파일: `data/hmimap.db`
- Blob 등 큰 데이터: 파일 시스템

---

## 5. DB 스키마

```sql
-- 실행 기록
CREATE TABLE runs (
  id INTEGER PRIMARY KEY,
  input_file TEXT,
  config TEXT,           -- JSON
  started_at TEXT,
  finished_at TEXT,
  duration_sec REAL,
  status TEXT,          -- success / failed / running
  qa_score REAL,
  output_path TEXT,
  log TEXT
);

-- 피드백
CREATE TABLE feedbacks (
  id INTEGER PRIMARY KEY,
  run_id INTEGER,
  rating TEXT,          -- good / bad
  issue_type TEXT,      -- wall / color / artifact / other
  comment TEXT,
  created_at TEXT,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);
```

---

## 6. 구현 단계

### Phase 1: 기본 UI (1~2일)
- React + Vite 세팅
- 업로드 화면 (drag & drop)
- Config 선택기
- 실행 버튼 + 로딩 상태

### Phase 2: 백엔드 연동 (1~2일)
- Express 서버 (port 3001)
- Python subprocess 실행
- SQLite 연결
- 실행 이력 저장

### Phase 3: 결과 뷰어 (1일)
- Image preview + zoom
- transparent/masks overlay toggle
- Anchors 좌표 표시

### Phase 4: 피드백 시스템 (1일)
- 👍/👎 버튼
- 선택式 코멘트
- 피드백 목록

### Phase 5: 설정 관리 (1일)
- Config editor (JSON)
- 저장/불러오기
- AI 파라미터 sliders

---

## 7. 파일 구조 (예상)

```
hmimap-web/
├── src/
│   ├── App.jsx
│   ├── components/
│   │   ├── FileUploader.jsx
│   │   ├── ConfigSelector.jsx
│   │   ├── RunButton.jsx
│   │   ├── ResultViewer.jsx
│   │   ├── FeedbackButtons.jsx
│   │   ├── RunHistory.jsx
│   │   └── SettingsEditor.jsx
│   ├── pages/
│   │   ├── Home.jsx
│   │   ├── History.jsx
│   │   └── Settings.jsx
│   └── lib/
│       └── api.js
├── server/
│   ├── index.js        -- Express 서버
│   ├── runPipeline.js  -- Python subprocess 실행
│   ├── db.js            -- SQLite
│   └── routes/
│       ├── runs.js
│       ├── feedbacks.js
│       └── config.js
├── data/                -- SQLite DB (gitignore)
├── dist/                -- Vite 빌드 산출물
└── package.json
```

---

## 8. 우선순위

| 순위 | 기능 | 이유 |
|------|------|------|
| 1 | 파일 업로드 + 실행 | 핵심 기능 |
| 2 | 결과 미리보기 | 마스터가 결과 확인 필수 |
| 3 | 실행 이력 | 히스토리 관리 |
| 4 | 피드백 | 마스터 피드백 수집 |
| 5 | 설정 관리 | 고급 사용자용 |

---

## 9. 컨펌 필요 사항

마스터에게 확인 후 진행:

1. **독립 배포 (port 3001) vs boom-dashboard 통합?**
2. **Phase 1부터 바로 구현? 아니면 기획서 컨펌 후?**

---

## 10. 참고: 기존 boom-master-dashboard

- 경로: `/home/sykim/workspace/boom-master-dashboard`
- Port: 5173 (PM2: boom-dashboard)
- Framework: React + Express + Tailwind CSS
- Pattern: `server/index.js` + `src/components/*.jsx` + `dist/` (Vite)
- 배포: Cloudflare Pages (boom-master-dashboard repo)