FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gcc-10 \
    g++-10 \
    clang \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV CC=gcc-10
ENV CXX=g++-10

RUN git clone https://github.com/iankorovinsky/autoresearch.git /opt/autoresearch \
    && git -C /opt/autoresearch checkout "85d252915b65d06003bed2e7c9f1969e6d3cbbba" \
    && cd /opt/autoresearch/KernelBench \
    && uv python install 3.10 \
    && uv sync --extra gpu

WORKDIR /workspace
