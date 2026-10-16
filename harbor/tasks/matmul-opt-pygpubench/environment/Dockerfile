FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libseccomp2 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

RUN git clone https://github.com/iankorovinsky/autoresearch.git /opt/autoresearch \
    && git -C /opt/autoresearch checkout "50e99ad"

RUN uv venv --python 3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python torch --index-url https://download.pytorch.org/whl/cu130 \
    && uv pip install --python /opt/venv/bin/python pygpubench==0.0.3 numpy requests pydantic python-dotenv tqdm openai litellm ninja

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH=/opt/autoresearch/KernelBench/src
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV TORCH_EXTENSIONS_DIR=/tmp/kb_build

WORKDIR /workspace
