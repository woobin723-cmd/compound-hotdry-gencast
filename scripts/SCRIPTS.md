# 실행 모듈 인벤토리

단계별 진입점(`python -m ...`)과 설정 매핑. 실행·환경(conda env·GPU·경로)은 사용자 몫.
모델별 하이퍼파라미터는 `config/*.yaml`, 추론 manifest는 `config/detectors_spi1.json`.

## Stage 1 — 데이터·라벨
| 단계 | 명령 |
|---|---|
| 다운로드 | `python -m src.data.download_era5 --start 1989 --end 2020` (env: data) |
| 전처리 | `python -m src.data.preprocess --tag 1989-2020` |
| doy-아노말리 채널 | `python -m src.data.add_doy_anomaly --tag 1989-2020` (폭염·복합 입력용) |
| 라벨링 | `python -m src.labeling.compound --tag 1989-2020` (폭염·SPI-1 가뭄·복합) |

## Stage 1 — 4멤버 앙상블 학습
`python -m src.train --full --target {drought|heatwave|compound} --config {config} --proc {nc} --labels {labels} --out {ckpt}`

| 멤버 | config(아노말리=폭염·복합) |
|---|---|
| CNN-LSTM | `model_cnn_lstm_anom_spi1_focal.yaml` (가뭄: `model_cnn_lstm_spi1_focal.yaml`) |
| ConvLSTM | `model_convlstm_anom_spi1_focal.yaml` (가뭄: `model_convlstm_spi1_focal.yaml`) |
| U-Net | `model_unet_anom_spi1_focal.yaml` (가뭄: `model_unet_spi1_focal.yaml`) |
| Transformer | `model_transformer_anom_spi1.yaml` (가뭄: `model_transformer_spi1.yaml`) |

- proc: 폭염·복합 = `data/processed/era5_daily_anom_1989-2020.nc`, 가뭄 = `data/processed/era5_daily_1989-2020.nc`
- labels: `data/labels/labels_spi1_1989-2020.nc`
- 멤버 선정·임계값: `python -m src.eval.ablation --members-target {target}` → manifest 갱신
- 앙상블 단발 평가: `scripts/ensemble_eval.py`

## Stage 2 — GenCast 추론·후처리·평가
| 단계 | 명령 |
|---|---|
| 입력 다운로드 | `scripts/download_summer_input.py` (2015–2020 · 90 init) |
| GenCast 추론 | gencast env에서 90 init×8멤버 15일 → `predictions.zarr` |
| 후처리 | `python -m src.data.gencast_postprocess --raw-dir data/gencast_raw` |
| 리드타임 평가 | `python -m src.eval.skill_decay --mode leadtime --target {t} --year 2015..2020 --save-json ...` |
| 탐지기 재현 | `python -m src.eval.detector --manifest config/detectors_spi1.json --split test` |

## 그림 (notebooks/)
`plot_pipeline.py` · `plot_skill_decay.py` · `plot_detector_confusion.py` · `plot_reliability.py` · `plot_ablation.py` · `plot_diversity.py` · `plot_learning_curve.py` → `notebooks/figures/`.
