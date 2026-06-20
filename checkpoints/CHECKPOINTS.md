# 체크포인트 인벤토리

탐지 모델(PyTorch) 가중치 목록. **활성(루트)** = manifest가 참조하는 평가용 가중치, **archive/** = 보존하되 비활성인 구버전·중간단계.

> 보존 원칙: SPI-3·SPI-1 모델 전부 보존(삭제 금지). 정리는 물리 이동만, manifest 경로 불변.

## 활성 (checkpoints/ 루트) — manifest 참조

### SPI-1 정식 (주력 — `config/detectors_spi1.json`)
| 파일 | 타깃 | 구성 | Test 성능 |
|---|---|---|---|
| `cnn_lstm_compound_spi1.pt` | 복합 | 앙상블 멤버(CNN-LSTM) | 앙상블 CSI 0.1691 / AUC 0.950 |
| `transformer_compound_spi1.pt` | 복합 | 앙상블 멤버(Transformer) | (앙상블 mean, thr 0.5174) |
| `cnn_lstm_drought_spi1.pt` | 가뭄 | 단일(CNN-LSTM) | CSI 0.5565 / AUC 0.948 |
| `cnn_lstm_heatwave.pt` | 폭염 | 단일(CNN-LSTM, SPI 독립·공유) | CSI 0.2150 / AUC 0.922 |

### SPI-3 baseline (보존 — `config/detectors.json`, 멀티스케일)
| 파일 | 타깃 | 구성 | Test 성능 |
|---|---|---|---|
| `cnn_lstm_compound_ms.pt` | 복합 | 앙상블 멤버(CNN-LSTM·멀티스케일) | 앙상블 CSI 0.0803 / AUC 0.930 |
| `transformer_compound_ms.pt` | 복합 | 앙상블 멤버(Transformer·멀티스케일) | (앙상블 mean, thr 0.5045) |
| `cnn_lstm_drought_ms.pt` | 가뭄 | 단일(멀티스케일) | CSI 0.3427 / AUC 0.917 |

> `cnn_lstm_heatwave.pt`는 SPI-1·SPI-3 manifest가 **공유**(폭염은 SPI 정의와 무관).

## archive/ — 보존(비활성)

| 파일 | 단계 | 비고 |
|---|---|---|
| `cnn_lstm.pt` | Stage 0 smoke | 2015-07 슬라이스 파이프라인 관통 |
| `transformer_smoke.pt` | smoke | Transformer 초기 smoke(AUC 0.81) |
| `cnn_lstm_full.pt` | Stage 1 테스트 라운드 | 30년 15ep, Test CSI 0.0362 |
| `transformer_full.pt` | Stage 1 테스트 라운드 | 30년 40ep, Test CSI 0.0278 |
| `cnn_lstm_formal.pt` | SPI-3 정식(14일) | 복합 Test CSI 0.0469 |
| `transformer_formal.pt` | SPI-3 정식(14일) | 복합 Test CSI 0.0252 |
| `cnn_lstm_drought.pt` | SPI-3 가뭄(14일·ms 전) | CSI 0.100 — 멀티스케일로 0.343 개선 전 baseline |
| `cnn_lstm_compound_spi1.pt.interrupted_bak` | 중단 백업 | SPI-1 복합 학습 중단 시 백업(정식 완성본으로 대체됨) |

## 명명 규칙
`{model}_{target}_{variant}.pt` — model∈{cnn_lstm, transformer}, target∈{compound, drought, heatwave}, variant∈{(없음)=Stage0/formal, full=테스트, ms=SPI-3 멀티스케일, spi1=SPI-1 정식}.
