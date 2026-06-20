#!/usr/bin/env bash
# GPU가 비는 즉시(타세션 raw_solar 종료) 폭염·가뭄 4멤버 학습을 시작.
# raw_solar는 PID가 바뀌며 재시작되므로 PID 대신 "여유 메모리 > THRESH 지속"으로 판정.
# 사용자 지시(2026-06-13): "지금 돌고있는 작업 끝나는대로 시작해".
set -uo pipefail
cd /data/woobin/deeplearning
THRESH=28000   # MiB. raw_solar(18.5GB) 종료 시 여유 ~32GB로 점프.
GPU=0          # 대상 GPU index(다중 GPU 대비 명시).
LOG=logs/wait_gpu_then_train.log
mkdir -p logs
echo "[wait] 시작 $(date '+%F %T') — GPU${GPU} 여유 > ${THRESH}MiB 대기" >> "$LOG"

# nvidia-smi 여유 메모리(MiB)를 정수로 안전 추출. 실패/비정상 출력이면 -1(=대기 지속) 반환.
free_mem() {
  local v
  v=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo -1
}

while true; do
  if [ "$(free_mem)" -gt "$THRESH" ]; then
    sleep 20   # 일시적 빈틈(프로세스 교체) 배제 위해 재확인
    if [ "$(free_mem)" -gt "$THRESH" ]; then
      echo "[wait] GPU${GPU} 여유 확인($(free_mem)MiB) → 학습 시작 $(date '+%F %T')" >> "$LOG"
      break
    fi
  fi
  sleep 30
done

bash scripts/run_spi1_hw_dr_members.sh >> "$LOG" 2>&1
rc=$?
echo "[wait] 학습 스크립트 종료(exit=$rc) $(date '+%F %T')" >> "$LOG"
