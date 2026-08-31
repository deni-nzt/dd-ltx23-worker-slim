# dd-ltx23-worker-slim — the host lane's code+env image, NO WEIGHTS (2026-08-31).
#
# WHY: the 32GB weights-baked image (dd-ltx23-worker:20260822-v31) dies in a nightly
# GHCR→RunPod throughput wall — pods bill while pulling and never start (two nights
# running, 08-30 + 08-31). The 6.6GB image class boots through the same wall every
# time. So: bake exactly what boot.sh used to build at runtime (repo @ pin, venv,
# stage-1 scale patch) and leave the ~78GB of weights to the boot-time HuggingFace
# stanza that ran nightly for weeks before the bake without ever hitting this wall.
# Weights land in HF_HOME=/workspace/hf — on volume-mounted pods that is the network
# volume, so a warm volume turns the whole download into a seconds-long cache check.
#
# Contract parity with the baked image (lib/pipeline/host/ltx-host-client.ts):
#   - ships /opt/boot-baked.sh (the client's dockerStartCmd runs exactly that);
#   - fetches server.py from the signed SERVER_URL bundle at start (always current);
#   - serves on :8188; WORKER_MODE picks a2v/i2v as before.
# runpod/base, NOT nvidia/cuda:*-runtime (canary 2026-08-31, pod lhxti2xho5w49n, $0.10):
# triton JIT-compiles kernels at first gen and needs the compiler toolchain the runtime
# image lacks ("include_dirs/libraries/ccflags" death in triton.runtime.build). This is
# the exact base the self-bootstrap mode ran on for weeks — zero unknown-unknowns.
FROM runpod/base:1.1.0-cuda1281-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive PATH=/root/.local/bin:$PATH
RUN apt-get update -qq && apt-get install -y -qq git ffmpeg jq curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# LTX-2 at the SAME pin boot.sh has always used — the locked recipe's ground truth.
WORKDIR /opt
RUN git clone https://github.com/Lightricks/LTX-2.git && cd LTX-2 && \
    git checkout 4f8905737aac86a554637cac86c178877a39c744 && \
    git rev-parse HEAD > /opt/LTX2_SHA

# STAGE-1 SCALE PATCH (source tree) — byte-identical to boot.sh's PATCH block.
COPY patch_stage1.py /opt/patch_stage1.py
RUN python3 /opt/patch_stage1.py /opt/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py

WORKDIR /opt/LTX-2
RUN uv python install 3.11 && uv sync && \
    uv pip install hf_transfer fastapi uvicorn pydantic && uv cache clean

# STAGE-1 SCALE PATCH (runtime venv copy) — uv sync installs ltx_pipelines into the
# venv and the interpreter loads THAT copy; patching only the source tree is inert
# (boot.sh learned this 2026-08-13).
RUN RUNTIME_A2VID=$(uv run python -c "import ltx_pipelines.a2vid_two_stage as m; print(m.__file__)" | tail -1) && \
    python3 /opt/patch_stage1.py "$RUNTIME_A2VID"

ENV HF_HUB_ENABLE_HF_TRANSFER=1
COPY boot-baked.sh /opt/boot-baked.sh
RUN chmod +x /opt/boot-baked.sh
CMD ["bash", "/opt/boot-baked.sh"]
