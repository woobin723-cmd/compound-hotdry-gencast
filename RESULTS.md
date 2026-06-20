# 결과 종합 1-pager (PPT 제작용)

> 복합 극한기후(Hot-Dry) 탐지 + GenCast 평가 — 전체 결과 요약.
> 슬라이드 단위 상세는 메모리 `report-material.md`, 그림은 `notebooks/figures/`.

## 0. 한 줄 결론
ERA5 학습 **4멤버 앙상블 탐지모델**(AUC 0.98)을 Frozen 잣대로, GenCast 15일 예측을 리드타임 평가 →
**가뭄은 2주 내내, 복합·폭염은 ~1주까지 재현. GenCast 앙상블은 과신(under-dispersion)**.
고온 타깃(폭염·복합)은 **doy-아노말리(평년편차) 입력 채널로 라벨 정의에 정렬** → 폭염 CSI +60%·복합 +43%(순수 통제).

## 1. 평가 프레임
- **Pipeline A**: ERA5 → 전처리/라벨링 → 탐지모델 학습(**4멤버 앙상블: CNN-LSTM·ConvLSTM·U-Net·Transformer·전부 Focal**, PyTorch). 입력 = t2m·tp·Z500·t850 **+ 폭염·복합은 t2m doy-아노말리(5채널)**.
- **Pipeline B**: ERA5 초기조건 → GenCast 추론(Frozen·8멤버·15일) → 후처리 → A의 탐지모델로 리드 1~15 평가. 아노말리 채널은 하이브리드 윈도우에서 즉석 산출(GenCast 재추론 불필요).
- **하이브리드 윈도우**: detector 30일 창 = ERA5 과거(30−d) + GenCast 미래(d). ERA5 29+GenCast 15=44일 슬라이딩 → 리드 1~15 한 번에.
- 기준선: ERA5=perfect(상한)·climatology=no-skill. **절대 CSI 금지·상대 SS(CSI 기반) 우선**.

## 2. 핵심 수치

### Pipeline A — 탐지모델 자체 (ERA5 Test 2015–2020, 평가 상한 · 4멤버·폭염/복합 doy-아노말리)
| 타깃 | base | CSI | AUC | POD | Prec | PR-AUC÷base |
|---|---|---|---|---|---|---|
| 가뭄(4채널) | 15.4% | **0.700** | 0.981 | 0.84 | 0.81 | ~6배 |
| 폭염(아노말리) | 3.04% | **0.477** | 0.986 | 0.66 | 0.64 | ~23배 |
| 복합(아노말리) | 0.81% | **0.290** | 0.986 | 0.46 | 0.44 | ~53배 |

- **doy-아노말리 순수 통제효과**(4멤버·30일·Focal 동일·입력채널만): 폭염 CSI 0.2985→**0.4770(+60%)**·복합 0.2031→**0.2897(+43%)**·Recall 폭염 0.40→0.66·복합 0.30→0.46. heat가 복합 병목이라는 진단 직접 입증.
- **복합 앙상블 선정**: ablation에서 4멤버 전체(CVUT) 채택(Val·Test 일치). 멤버 확률상관 0.53~0.85.

### Pipeline B — GenCast 리드타임 (6년 90 init · 아노말리 detector, 2026-06-15)
| 타깃 | Day1 AUC | Day5 | Day10 | Day15 | 상대SS(CSI) Day1→15 |
|---|---|---|---|---|---|
| 가뭄(4채널) | 0.985 | 0.971 | 0.927 | **0.862** | 0.98→0.51 |
| 복합(아노말리) | 0.975 | 0.939 | 0.807 | 0.636 | 0.83→0.05 |
| 폭염(아노말리) | 0.966 | 0.918 | 0.775 | 0.630 | 0.79→0.08 |

- ERA5 perfect 전 리드 평탄(AUC~0.985)=상한선 · climatology AUC 0.78~0.90(강한 계절 baseline)·CSI=0.
- **Spread-Error**: 모든 타깃 RMSE≫spread = 심한 under-dispersion(앙상블 과신).
- ⭐ **잣대를 아노말리로 개선(Day1 AUC 0.92~0.96→0.97~0.99)해도 리드 한계(~1주) 불변** = 병목은 잣대 아닌 GenCast 예보(통제 실험).

## 3. 핵심 해석 (PPT 메시지)
1. **가뭄 ≫ 복합 ≈ 폭염**: 가뭄(느린 변수=누적강수/SPI-1 지속성)은 15일 재현 → **SPI-1 전환의 과학적 가치 입증**(SPI-3였다면 가뭄 평가 불가). 폭염(빠른 대기변동)은 ~1주 급감.
2. **4멤버 앙상블 = 다양성 결합**: 시간·공간 처리가 다른 4구조의 오류 탈상관(확률상관 0.53~0.68)으로 단독 최강보다 +16%. 단순 멤버 수가 아니라 **구조적 다양성**이 이득의 원천(ablation·확률상관으로 입증).
3. **복합 한계는 폭염 성분이 주요 병목**: 복합 Day15 상대SS 0.05가 가뭄 0.51이 아닌 폭염 0.08에 근접. **doy-아노말리로 폭염·복합을 동반 개선(+60%/+43%)한 것이 heat 병목을 직접 입증**. 단 잣대를 개선해도 리드 한계(~1주)는 불변 = 병목은 예보 모델 쪽(통제 실험).
7. **입력↔라벨 정렬이 핵심 레버**: 가뭄=입력창을 SPI 누적기간(30일)에 정렬, 폭염·복합=입력에 doy-아노말리(평년편차) 채널 추가로 라벨(doy-p95) 정렬. 같은 원리의 단일변수 통제로 큰 향상(앙상블·손실보다 효과 큼).
4. **GenCast 앙상블 과신**: under-dispersion + Reliability 과신 두 독립 분석이 같은 결론 → 확률 보정 필요.
5. **평가 대상 명확화**: "GenCast 직접 예측성"이 아니라 "ERA5 학습 Frozen detector score와 ERA5 기준의 일치도". detector miss/false-alarm이 평가에 전파(과대포장 금지).
6. **AUC 단독 서사 금지**: 희귀사건은 CSI/POD/Precision/calibration 병기로 방어(상대SS는 CSI 주지표 — AUC 기반은 강한 clim 대비 후반 음수=ranking 약화이지 "무가치" 아님).

## 4. 그림 (notebooks/figures/)
| 파일 | 내용 | 슬라이드 |
|---|---|---|
| `pipeline_overview.png` | Pipeline A→Frozen 탐지기→B 전체 구조도(가중치 공유·복합 4멤버) | 1 |
| `diversity_matrix.png` | 4멤버 다양성 매트릭스(시간×공간 평면·개념도) | 6/7-3 |
| `ablation_csi.png` | 멤버셋 15조합 Test CSI 바(CVUT 0.203 최적) | 7-3 |
| `member_corr.png` | 멤버쌍 확률상관 히트맵(C↔T 0.53 최다양) | 7-3 |
| `learning_curves.png` | 복합 4멤버 Focal 학습곡선(train_loss 절대비교·val_CSI ★best) | 7-1 |
| `skill_decay_curves.png` | 리드별 AUC·상대SS(CSI 주지표), 3타깃 | 8-12 |
| `confusion_matrices.png` | 타깃×리드(Day1/5/10/15) confusion, POD decay | 8-12 |
| `spread_error.png` | 리드별 spread vs RMSE, under-dispersion 진단 | 8-12/8-13 |
| `reliability_diagram.png` | 확률 보정 2패널(Pipeline A 상한 vs B GenCast) | 8-13 |
| `detector_confusion_era5.png` | 탐지모델 자체 confusion(ERA5 입력=모델 상한·폭염 0.477·복합 0.290 아노말리) | 8-13 |
| `01~05_*.png` | 라벨 검증(월별·공간·2016/2018 사례) | 5 |

## 5. 한계 (정직한 서술)
동아시아·1°·여름·Hot-Dry 전용(범용 아님) · 2005는 Train 누수 plumbing 검증(본평가 2015–2020과 분리) ·
복합 event detection 상한 제한적(극희소) · 앙상블 under-dispersion 확률 보정 미적용 · 평가가 detector 편향을 포함.

## 6. 진행 상태
게이트 0/1/2 통과 ✅. 4멤버 앙상블 확장(2026-06-09)·폭염·가뭄 4멤버 통일(06-14)·**doy-아노말리로 폭염 CSI 0.215→0.477·복합 0.203→0.290(06-15)**·**Pipeline B 아노말리 재평가 완료(06-15)**. Stage 3 정리 완료. 배포(GitHub 공개)는 보류.
