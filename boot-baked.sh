#!/bin/bash
# dd-ltx23-worker-slim boot (2026-08-31) — the baked-image contract, minus baked weights.
# Repo/venv/patches are IN the image (/opt/LTX-2, pin asserted below); server.py arrives
# from the signed SERVER_URL bundle exactly as always; weights come from HuggingFace via
# the proven parallel hf_transfer stanza (boot.sh 2026-08-11) into HF_HOME=/workspace/hf —
# on volume-mounted pods that's the network volume, so a warm volume makes this a
# seconds-long cache check. Idempotent; logs to /workspace/boot.log.
set -uo pipefail
mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
echo "=== slim boot start $(date -u) ==="

if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
  (service ssh start || /usr/sbin/sshd) >/dev/null 2>&1 || true
fi

export PATH=/root/.local/bin:$PATH
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
printf '%s' "${HF_TOKEN:?HF_TOKEN env missing}" > /workspace/.hf_token
chmod 600 /workspace/.hf_token

# Pin assert — the byte-exact law: a slim image whose repo drifted from the recipe pin
# must die loudly, never serve.
WANT_SHA=4f8905737aac86a554637cac86c178877a39c744
GOT_SHA=$(cat /opt/LTX2_SHA 2>/dev/null || echo missing)
if [ "$GOT_SHA" != "$WANT_SHA" ]; then
  echo "!!! LTX-2 pin mismatch: want $WANT_SHA got $GOT_SHA — refusing to serve"
  exit 1
fi
cp /opt/LTX2_SHA /workspace/LTX2_SHA

curl -fsSL "${SERVER_URL:?SERVER_URL env missing}" -o /workspace/server.py
echo "server.py fetched: $(wc -c < /workspace/server.py) bytes"

# DISK GUARD: a cold weights download needs ~90GB under /workspace. The volumeless baked
# round deploys 30GB container disk — die in seconds (cheap, the deploy walks on) instead
# of filling the disk at minute 12. A warm cache passes regardless of free space.
NEED_GB=90
FREE_GB=$(df -BG /workspace | awk 'NR==2 {gsub("G","",$4); print $4}')
CACHE_HINT=$(find /workspace/hf -name "*.safetensors" -size +1G 2>/dev/null | head -1)
if [ -z "$CACHE_HINT" ] && [ "${FREE_GB:-0}" -lt "$NEED_GB" ]; then
  echo "!!! ${FREE_GB}GB free under /workspace, cold cache needs ~${NEED_GB}GB — this round cannot hold the weights (mount the volume or raise the container disk)"
  exit 1
fi

# WEIGHTS (the proven 2026-08-11 stanza, verbatim semantics): hf download is idempotent
# and resumable — a warm cache turns each call into a fast no-op, so this runs
# unconditionally and self-heals partial caches. Parallel via hf_transfer; a failure
# retries serially without it (a slow boot beats a dead pod).
cd /opt/LTX-2
echo "=== weights start $(date -u) ==="
PIDS=""
i=0
for f in ltx-2.3-22b-dev.safetensors ltx-2.3-22b-distilled-lora-384-1.1.safetensors ltx-2.3-spatial-upscaler-x2-1.1.safetensors; do
  i=$((i + 1))
  uv run hf download Lightricks/LTX-2.3 "$f" > "/workspace/dl_$i.log" 2>&1 &
  PIDS="$PIDS $!"
done
uv run hf download google/gemma-3-12b-it-qat-q4_0-unquantized > /workspace/dl_gemma.log 2>&1 &
PIDS="$PIDS $!"
DL_FAIL=0
for p in $PIDS; do wait "$p" || DL_FAIL=1; done
if [ "$DL_FAIL" != 0 ]; then
  echo "!!! a parallel download failed — retrying serially without hf_transfer"
  unset HF_HUB_ENABLE_HF_TRANSFER
  for f in ltx-2.3-22b-dev.safetensors ltx-2.3-22b-distilled-lora-384-1.1.safetensors ltx-2.3-spatial-upscaler-x2-1.1.safetensors; do
    uv run hf download Lightricks/LTX-2.3 "$f" >/dev/null && echo "GOT $f"
  done
  uv run hf download google/gemma-3-12b-it-qat-q4_0-unquantized >/dev/null && echo "GOT gemma"
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi
echo "=== weights done $(date -u) ==="

echo "=== starting server $(date -u) ==="
exec uv run uvicorn server:app --host 0.0.0.0 --port 8188 --app-dir /workspace
