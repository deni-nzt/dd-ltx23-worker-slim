# dd-ltx23-worker-slim

Slim HOST (A2V) worker image for denidose — **code + env baked, NO weights** (~7GB vs the
32GB weights-baked image whose GHCR pull dies in the nightly registry-throughput wall).
Weights come at boot from HuggingFace via the proven parallel hf_transfer stanza
(cache-hits instantly when /workspace is the network volume and already warm).

Drop-in for the baked-image contract: ships `/opt/boot-baked.sh`, fetches the signed
server.py bundle (SERVER_URL) at start, serves on :8188. LTX-2 pinned at 4f890573.
