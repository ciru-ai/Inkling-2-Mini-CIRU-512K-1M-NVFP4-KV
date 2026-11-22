# syntax=docker/dockerfile:1
FROM vllm/vllm-openai:inkling-2-mini@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
COPY overlay/ /usr/local/lib/python3.14/dist-packages/vllm/models/inkling2/nvidia/
ENV INKLING_SM120_NVFP4_KV=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1
