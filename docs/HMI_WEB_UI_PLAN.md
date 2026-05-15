# HMI Map Pipeline — Web UI 기획서

Date: 2026-05-16
Status: 기획 완료 → 구현 대기

---

## 1. 개요

마스터가 파이프라인을 **편하게 사용하고 피드백을 줄 수 있는 웹 대시보드**.

**핵심 목적:**
- DXF/DWG 파일 업로드 → 파이프라인 실행 → 결과 확인
- **파이프라인 각 단계별 검수 +パラ미터 조정**
- 실행 이력 관리 (타임라인, 로그)
- 피드백 제출 (좋은 결과 / 실패한 결과 신고)
- 설정 변경 (AI 파라미터, QA threshold)

---

## 2. 파이프라인 단계별 검수 시스템 (핵심)

### 전체 흐름

```
[파일 업로드]
    ↓
1️⃣ 전처리 (Preprocess)
   - 출력: 정규화된 entities, layer mapping 결과
   - 검수: 레이어 목록, entity 수, scale (mm/unit)
   - [실행] → [결과 미리보기] → [통과✅ / 재실행↻ / 수정✏️]
    ↓
2️⃣ 파싱 (Parser)
   - 출력: wall_segments, rooms, columns, openings
   - 검수: 벽 segment 목록, 방 면적, 기둥 위치, 개구부
   - visualized: 벽 graph overlay on floor plan (2D vector view)
   - [실행] → [결과] → [통과✅ / 재실행↻]
    ↓
3️⃣ 렌더링 (ISO Renderer)
   - 출력: base_render.png, depth_map, normal_map
   - 검수: 벽 두께, 높이, 색상, 조명, depth quality
   - visualized: 3D ISO preview (interactive)
   - [실행] → [결과] → [통과✅ / 파라미터수정✏️]
    ↓
4️⃣ AI 리파인먼트 (AIRefiner) — 선택
   - 출력: ai_enhanced.png
   - 검수: 텍스처, 조명, 잔상물 (hallucination)
   - [실행] → [결과] → [통과✅ / Skip⏭️]
    ↓
5️⃣ QA (AutoQA)
   - 출력: qa_report.json, background.png
   - 검수: alignment score, color score, artifact score
   - [통과✅ / 재실행↻ / Deterministic 폴백🔄]
    ↓
6️⃣ 최종 검수 (Final Review)
   - Final preview: background.png + transparent overlay + masks overlay
   - [피드백 제출] → 완료
```

### 실행 모드 3가지

| 모드 | 설명 | 용도 |
|------|------|------|
| **한번에 실행** | 전체 End-to-End 자동 | 빠른 결과 필요 시 |
| **단계별 검수** | 각 단계 완료마다 검수 → 다음 | 품질 중요한 경우 |
| **특정 단계부터** | 이전 결과 재사용, 특정 단계만 다시 | 파라미터 조정 시 |

### 중간 결과 저장

- 각 단계 완료 시 → DB에 저장
- 다음에 접속 시 → 이어서 계속 가능
- 예: 렌더링 단계에서 파라미터 수정 → 렌더링만 다시 → QA继续

### 단계별 검수 포인트 상세

#### 1️⃣ 전처리 검수
- 레이어 매핑 표 (원본 → 정규화)
- Entity 수 카운트 (walls, doors, windows, columns)
- Scale 확인 (mm/inch 자동 감지)
- 문제: 누락된 레이어, 스케일 오류 → 수정 후 재실행

#### 2️⃣ 파싱 검수
- Wall segments: 수, 총길이, 평균 길이
- Rooms: 방 수, 면적 합계
- Columns: 수, 위치
- Openings: 문/창 수, 위치
- 2D floor plan visualization (SVG/Canvas overlay)

#### 3️⃣ 렌더링 검수
- ISO 스냅샷 (interactive zoom/pan)
- Depth map 미리보기 (pseudo-color)
- Normal map 미리보기
- 파라미터: wall_height, wall_thickness, bg_color → 실시간 반영

#### 4️⃣ AI 리파인먼트 검수
- Before/After 비교 (slider)
- ControlNet Tile strength 조절
- Denoise strength 조절 (0.0~0.5)
- 문제 감지: hallucination → 파라미터 조정 후 재실행

#### 5️⃣ QA 검수
- 구조 alignment: 0~100%
- Color rule: grayscale/non-grayscale %
- Artifact detection:有问题 영역 highlight
- QA threshold 설정: alignment < X% → failed

#### 6️⃣ 최종 검수
- background.png 고해상도 미리보기
- transparent_background.png
- masks.png overlay
- anchors.json 좌표 목록
- 최종 피드백 제출

---

## 3. 주요 기능

### 3.1 파일 업로드
- Drag & Drop DXF/DWG/PNG/PDF
- 파일 정보 표시 (크기, type, entities 수 추정)

### 3.2 설정 관리
- AI 리파인먼트 On/Off
- Denoise strength (0.0~0.5 slider)
- Tile size (512/768/1024)
- QA threshold values
- Config 저장/불러오기 (JSON)
- 파이프라인 모드: 한번에 / 단계별 / 특정 단계부터

### 3.3 실행 이력
- 타임라인 뷰 (최근 20개)
- 각 실행: 파일명, 실행 시간, 소요시간, QA 점수, 단계 수
- 클릭 → 상세 보기 (로그 + 설정 + 출력)

### 3.4 피드백 시스템
```
결과 옆 [👍 좋은 결과] [👎 문제가 있었어요]
                ↓
        [어떤 문제가 있었나요?]
        - 벽이 잘못됨
        - 색상이 이상함
        - AI가 추가 객체를 생성함
        - 렌더링 왜곡
        - 기타 (텍스트 입력)
                ↓
        DB에 저장 → 향후 학습 데이터로 활용
```

---

## 4. 페이지 구조

```
┌─────────────────────────────────────────────────────────┐
│ HMI Map Pipeline                    [모드: 단계별▾]   │
│                                      [설정] [이력]     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [파일 업로드 / Drag & Drop]                            │
│                                                         │
│  현재 단계: 3️⃣ 렌더링                      [실행▶]     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ 1 전처리│→│ 2 파싱  │→│ 3 렌더링│→│ 4 AI    │→...  │
│  │   ✅   │  │   ✅   │  │  🔄    │  │   ⏸   │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│                                                         │
│  ════════════════════ 결과 미리보기 ════════════════  │
│                                                         │
│  [ISO 렌더링 결과 썸네일]         [depth map]          │
│                                                         │
│  파라미터: height=3000 th=200 color=#A0A0A0            │
│                                    [통과✅] [수정✏️]   │
│                                                         │
└─────────────────────────────────────────────────────────┘

하단 탭:
- [홈] — 현재 작업
- [이력] — 실행 기록 타임라인
- [피드백] — 제출된 피드백 목록
- [설정] — Config 편집
```

---

## 5. 기술 선택

**독립 배포** — `hmimap-web/` 프로젝트, port 3001 (boom-dashboard 충돌 방지)

### Frontend
- **React** + Vite
- Tailwind CSS
- Canvas/SVG: 단계별 시각화 (floor plan overlay, ISO preview)

### Backend
- **Express** (port 3001)
- Python subprocess로 파이프라인 실행
- Streaming log (Server-Sent Events)

### 데이터 저장
- **SQLite**: 실행 기록, 피드백, 설정, 중간 결과
- 파일: `data/hmimap.db`

---

## 6. DB 스키마

```sql
-- 실행 기록
CREATE TABLE runs (
  id INTEGER PRIMARY KEY,
  input_file TEXT,
  mode TEXT,             -- 'full' / 'step' / 'resume'
  config TEXT,           -- JSON
  current_step INTEGER,
  started_at TEXT,
  finished_at TEXT,
  duration_sec REAL,
  status TEXT            -- running / completed / failed
);

-- 단계별 결과 (중간 결과 저장)
CREATE TABLE step_results (
  id INTEGER PRIMARY KEY,
  run_id INTEGER,
  step INTEGER,          -- 1=preprocess, 2=parser, 3=renderer, 4=ai, 5=qa
  step_name TEXT,
  output_path TEXT,       -- 파일 경로
  log TEXT,
  created_at TEXT,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);

-- 피드백
CREATE TABLE feedbacks (
  id INTEGER PRIMARY KEY,
  run_id INTEGER,
  step INTEGER,
  rating TEXT,            -- good / bad
  issue_type TEXT,        -- wall / color / artifact / other
  comment TEXT,
  created_at TEXT,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);

-- 설정 저장
CREATE TABLE configs (
  id INTEGER PRIMARY KEY,
  name TEXT,
  config_json TEXT,
  created_at TEXT
);
```

---

## 7. 구현 단계

### Phase 1: 기본 UI
- React + Vite + Tailwind 세팅
- 파일 업로드 (drag & drop)
- 단계별 진행률 UI (스텝 인디케이터)
- 실행 버튼 + 로딩 상태

### Phase 2: 단계별 실행 + 검수
- Express 서버 (port 3001)
- Python subprocess 실행 (streaming log via SSE)
- 각 단계별 결과 미리보기
- [통과/재실행/수정] 버튼
- SQLite: runs + step_results

### Phase 3: 시각화
- 2D floor plan overlay (wall graph on canvas)
- ISO renderer preview (interactive zoom)
- Depth/normal map preview (pseudo-color)
- Before/After slider (AI)

### Phase 4: 피드백 + 이력
- 👍/👎 버튼 + 코멘트
- 실행 이력 타임라인
- 상세 로그 뷰어

### Phase 5: 설정 관리
- Config editor (JSON)
- 저장/불러오기
- AI 파라미터 sliders
- 파이프라인 모드 선택

---

## 8. 컨펌 필요 사항

1. **독립 배포 (port 3001) vs boom-dashboard 통합?**
2. **Phase 1부터 바로 구현? 검수 프로세스同意?**
3. **단계별 시각화: Canvas (2D floor plan overlay) 필요?** (파싱 결과 wall graph를 원본 CAD 위에 오버레이해서 보여주는 기능)