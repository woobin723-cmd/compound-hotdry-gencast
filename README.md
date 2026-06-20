# 복합 극한기후(Hot-Dry) 탐지 + GenCast 예보 평가

> ERA5로 복합 극한기후(폭염 ∧ 가뭄, **Compound Hot-Dry**) **탐지 모델**을 직접 학습하고,
> 그 모델을 **고정(Frozen) 평가 잣대**로 삼아 사전학습 AI 기상모델 **GenCast의 15일 예측**이
> 복합극한을 며칠 리드까지 재현하는지 정량 평가한다.
> *(동아시아 · 1° · 여름 · Hot-Dry 전용 — 범용 예보 모델이 아님)*

이 저장소의 핵심 기여는 단일 가중치가 아니라 **2-파이프라인 평가 프레임 + 사건(grid-cell day) 중심 평가 방법론**이다.
탐지모델을 Frozen 잣대로 공유해 "AI 기상모델이 복합극한을 며칠까지 재현하나"를
**리드타임 Skill Decay · 상대 Skill Score · Spread-Error · Reliability**로 정량화한다.

📊 **결과 요약**: [`RESULTS.md`](RESULTS.md)  ·  🖼 **그림**: [`notebooks/figures/`](notebooks/figures/)
*(상세 보고서·발표자료는 개인정보 포함으로 저장소에 포함하지 않음)*

---

## 1. 평가 프레임

```
[Pipeline A · 학습]  WeatherBench2 ERA5 ── 전처리/라벨링 ── 4멤버 앙상블 탐지모델 학습
                                                                    │  (Frozen 가중치 공유)
                                                                    ▼
[Pipeline B · 평가]  ERA5 초기조건 ── GenCast 추론(15일·8멤버) ── 후처리 ── A의 탐지모델로 리드 1~15 평가
```

- **탐지모델 = 분류기**(예보가 아님). 주어진 기후장에서 사건을 판정 → 동아시아 한정 학습 OK.
- **미래 예측은 GenCast 전담**(추론 전용·Frozen). 탐지모델은 GenCast 출력을 채점하는 잣대.
- **하이브리드 윈도우**: detector 30일 창 = ERA5 과거(30−d) + GenCast 미래(d). ERA5 29 + GenCast 15 = 44일 슬라이딩 → 리드 1~15를 한 번에. 아노말리 채널은 윈도우에서 즉석 산출(GenCast 재추론 불필요).
- **기준선**: ERA5=완전예보(상한)·climatology=무정보·persistence=지속성. **절대 CSI 직접비교 금지 · 상대 SS(CSI 기반) 우선**.

---

## 2. 확정 설정

| 항목 | 값 |
|---|---|
| 복합 유형 | Hot-Dry 단일 |
| 공간 | 60°E–180°E, 0°N–70°N, 1° (71 lat × 120 lon) |
| 시간 | 1989–2020, 일평균 |
| 변수 | t2m, tp, Z500(=z/9.80665), t850 **+ 폭염·복합은 t2m doy-아노말리(5채널)** |
| 분할 | Train 89–09 / Val 10–14 / Test 15–20 (시간 기반·누수 방지) |
| 라벨 | 폭염 t2m>p95(doy±7)&3일연속 · 가뭄 SPI-1<−1 · 복합 동시충족 |
| 모델 | **4멤버 앙상블** — CNN-LSTM · ConvLSTM · U-Net · Transformer (전부 Focal) |
| 주 지표 | CSI (보조 AUC · PR-AUC÷base · POD · Reliability) |

> **가뭄 정의 전환**: SPI-3<−1.5 → **SPI-1<−1**. GenCast 15일 예보가 SPI-3(90일창)의 17%만 채워 가뭄 평가 불가 → SPI-1(30일)이면 ~50%. SPI-3 모델은 baseline으로 전부 보존.
>
> **doy-아노말리 (핵심 방법)**: 폭염 라벨은 doy-p95(기후값 상대 기준)인데 입력 t2m은 절대장 z-score라 입력↔라벨 기준이 불일치. **t2m 평년편차(doy 기후값±7, Train-only) 채널을 추가**해 라벨 정의에 정렬 → 폭염 CSI **+60%**, 복합 **+43%**(순수 단일변수 통제).

---

## 3. 주요 결과

### Pipeline A — 탐지모델 자체 성능 (ERA5 Test 2015–2020, 평가 상한)

| 타깃 | base rate | CSI | AUC | POD | Precision | PR-AUC÷base |
|---|---|---|---|---|---|---|
| 가뭄 (4채널) | 15.4% | **0.700** | 0.981 | 0.84 | 0.81 | ~6배 |
| 폭염 (아노말리) | 3.04% | **0.477** | 0.986 | 0.66 | 0.64 | ~23배 |
| 복합 (아노말리) | 0.81% | **0.290** | 0.986 | 0.46 | 0.44 | ~53배 |

복합 CSI가 낮은 건 base 0.81% 극희소에 따른 통계적 난이도이지 무능이 아님(PR-AUC÷base ~53배 = 무작위 대비 큰 lift).

### Pipeline B — GenCast 리드타임 Skill Decay (6년 90 init · 아노말리 detector)

| 타깃 | Day1 AUC | Day5 | Day10 | Day15 | 상대SS(CSI) Day1→15 | 실용 한계 |
|---|---|---|---|---|---|---|
| 가뭄 | 0.985 | 0.971 | 0.927 | **0.862** | 0.98 → 0.51 | **~2주** |
| 복합 | 0.975 | 0.939 | 0.807 | 0.636 | 0.83 → 0.05 | ~1주 |
| 폭염 | 0.966 | 0.918 | 0.775 | 0.630 | 0.79 → 0.08 | ~1주 |

- **가뭄 ≫ 복합 ≈ 폭염**: 가뭄(느린 변수=누적강수/SPI-1 지속성)은 2주 내내 재현, 폭염(빠른 대기변동)은 ~1주 급감. 복합 한계는 폭염 성분이 주요 병목.
- ⭐ **잣대를 아노말리로 개선(Day1 AUC 0.92~0.96 → 0.97~0.99)해도 리드 한계(~1주) 불변** = 병목은 잣대가 아닌 GenCast 예보(통제 실험).
- **GenCast 앙상블 under-dispersion**(RMSE ≫ spread = 과신) → 확률 보정 필요. Reliability·Spread-Error 두 독립 분석이 같은 결론.
- **지속성 분해**: 가뭄 2주 skill 상당부분은 antecedent(지속성), 폭염·복합은 GenCast가 naive persistence를 실제 능가.

> 평가 대상은 "GenCast 직접 예측성"이 아니라 "ERA5 학습 Frozen detector score와 ERA5 기준의 일치도". detector miss/false-alarm이 평가에 전파됨(과대포장 금지). GenCast 학습기간(~2018) 밖인 2019–2020만으로 재확인해도 결과 거의 동일.

전체 수치·해석은 [`RESULTS.md`](RESULTS.md), 그림은 [`notebooks/figures/`](notebooks/figures/) 참조.

---

## 4. 저장소 구조

```
src/
  data/      download_era5 · preprocess · add_doy_anomaly · gencast_postprocess · dataset
  labeling/  heatwave · spi · compound
  models/    cnn_lstm · convlstm · unet · transformer · losses
  eval/      metrics · detector(추론 진입점) · ablation · skill_decay
  train.py · utils.py
config/      data/labeling/model yaml · detectors_spi1.json(추론 manifest)
scripts/     단계별 실행 래퍼 (SCRIPTS.md 매핑)
notebooks/   라벨 검증 · 그림 생성 스크립트 · figures/ · html2pptx(발표자료 빌드)
checkpoints/ 활성 가중치 .pt (4멤버×3타깃 + baseline)
docs/        report.docx · report.md · slides.pptx
data/        (gitignore — 대용량, 아래 재현 흐름으로 재생성)
```

---

## 5. 환경

| env | 용도 |
|---|---|
| `data` | WeatherBench2 다운로드 (gcsfs, xarray) |
| `cecd` | 전처리·라벨링·탐지모델 학습·평가 (torch 2.11.0 cu128, RTX 5090) |
| `gencast` | GenCast 추론 (JAX, Pipeline B) |

```bash
conda create -n cecd python=3.11 && conda activate cecd
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## 6. 추론 (학습된 탐지모델 사용)

가중치(`checkpoints/*.pt`)와 manifest(`config/detectors_spi1.json`)가 동봉돼 있어 **로드만으로 재현**된다.

```bash
# 3타깃(가뭄·폭염·복합) 앙상블 탐지기를 Test 셋에 적용
conda run -n cecd python -m src.eval.detector \
  --manifest config/detectors_spi1.json --split test
# → 가뭄 CSI 0.700 · 폭염 0.477 · 복합 0.290 재현
```
⚠️ 폭염·복합 detector는 입력에 doy-아노말리 채널(`era5_daily_anom_*.nc`)이 필요(manifest `proc` 필드 참조).

---

## 7. 재현 흐름 (스테이지별)

```
Stage 0 얇은 슬라이스 → 게이트0 ✅ → Stage 1 Pipeline A → 게이트1 ✅
        → Stage 2 Pipeline B → 게이트2 ✅ → Stage 3 정리·배포
```

```bash
# Stage 0 — 파이프라인 관통 (다운로드→전처리→라벨링→smoke)
bash scripts/run_smoke.sh 2015-07

# Stage 1 — 30년 학습
conda run -n data python -m src.data.download_era5 --start 1989 --end 2020
conda run -n cecd python -m src.data.preprocess --tag 1989-2020
conda run -n cecd python -m src.data.add_doy_anomaly --tag 1989-2020   # 폭염·복합용 아노말리 채널
conda run -n cecd python -m src.labeling.compound --tag 1989-2020
bash scripts/run_spi1_hw_dr_members.sh    # 4멤버 앙상블 (가뭄·폭염)
bash scripts/run_heatwave_anom.sh         # 폭염 doy-아노말리
bash scripts/run_compound_anom.sh         # 복합 doy-아노말리

# Stage 2 — GenCast 추론 → 후처리 → 리드타임 평가
bash scripts/run_summer_inference.sh                                   # 90 init 추론 (gencast env)
conda run -n cecd python -m src.data.gencast_postprocess --raw-dir data/gencast_raw
bash scripts/run_pipelineB_reeval.sh                                   # 하이브리드 윈도우 리드타임

# 그림 생성
conda run -n cecd python notebooks/plot_skill_decay.py    # + plot_reliability / plot_detector_confusion / plot_ablation ...
```

스크립트 단계 매핑 → [`scripts/SCRIPTS.md`](scripts/SCRIPTS.md) · 가중치 목록 → [`checkpoints/CHECKPOINTS.md`](checkpoints/CHECKPOINTS.md) · 데이터 산출물 → `data/DATA.md`.

---

## 8. 데이터

대용량 원본 데이터(ERA5 6h, GenCast zarr 등 ~350GB)는 저장소에 포함하지 않으며 위 흐름으로 재생성한다.

- **ERA5**: WeatherBench2 공개 zarr — `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr` (gcsfs 익명 `token='anon'`)
- **GenCast**: [google-deepmind/graphcast](https://github.com/google-deepmind/graphcast) 사전학습 가중치로 추론(별도 환경).

---

## 9. 한계

동아시아·1°·여름·Hot-Dry 전용(범용 아님) · 복합 event detection은 극희소로 상한 제한적 · GenCast 앙상블 under-dispersion에 대한 확률 보정 미적용 · 평가가 detector 편향을 일부 포함 · 하이브리드 윈도우라 리드 후반도 ERA5 antecedent를 포함(GenCast 단독 재현이 아닌 운용조건 성능).

## 참고

핵심 선행연구: Zscheischler et al. 2018 (*Nat. Clim. Change*, 복합극한) · Mazdiyasni & AghaKouchak 2015 (*PNAS*, 동시 가뭄·폭염) · Prabhat et al. 2021 (*GMD*, ClimateNet) · Price et al. 2025 (*Nature*, GenCast). 상세는 보고서 참조.
