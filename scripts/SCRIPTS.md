# 실행 스크립트 인벤토리

단계별 재현 래퍼. 스테이지 순서대로 매핑.

| 스크립트 | 스테이지 | 용도 |
|---|---|---|
| `run_smoke.sh` | Stage 0 | 다운로드→전처리→라벨링→CNN-LSTM smoke(파이프라인 관통·과적합 확인) |
| `run_stage1_training.sh` | Stage 1 | 본학습 테스트 라운드(CNN-LSTM→Transformer 순차, `--full`) |
| `run_stage1_formal.sh` | Stage 1 | 정식 재학습(CSI 극대화: 오버샘플·cosine+warmup·조기종료·grad clip) |
| `run_singletarget.sh` | Stage 1 | 폭염·가뭄 단독 학습(구성요소별 비교) |
| `run_multiscale.sh` | Stage 1 | SPI-3 멀티스케일(25스텝/90일) 가뭄·복합 학습 |
| `run_spi1.sh` | Stage 1 | 🔄 SPI-1 전환 재학습(30일 일별 단일창) 가뭄·복합 |
| `run_spi1_compound.sh` | Stage 1 | SPI-1 복합 재시작 복구(`--no-capture-output` 실시간 로그) |
| `ensemble_eval.py` | Stage 1 | 앙상블 평가(CNN-LSTM+Transformer 확률 평균·Val에서 thr 최적화) |
| `download_summer_input.py` | Stage 2 | GenCast 여름 init 입력 ERA5 다운로드(2015–2020·90 init) |
| `run_summer_inference.sh` | Stage 2 | GenCast mini 15일 추론(90 init×8멤버→predictions.zarr) |

## 후처리·평가 (scripts 아님 — `python -m`)
- 후처리: `python -m src.data.gencast_postprocess --raw-dir data/gencast_raw --year {Y} ...`
- 리드타임 평가: `python -m src.eval.skill_decay --mode leadtime --target {t} --year 2015..2020 --save-json ...`
- 탐지기 핸드오프: `python -m src.eval.detector --manifest config/detectors_spi1.json`

## 그림 생성 (notebooks/)
`plot_pipeline.py` · `plot_skill_decay.py` · `plot_detector_confusion.py` · `plot_reliability.py` → `notebooks/figures/`.
