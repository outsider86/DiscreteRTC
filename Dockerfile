# Real-Time Chunking Kinetix - GPU training image
# Requires: docker buildx with --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 for submodule support
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=0
ENV UV_LINK_MODE=copy

# Install Python 3.12 and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy project files (submodule must be initialized before build)
COPY pyproject.toml uv.lock ./
COPY third_party/ third_party/
COPY src/ src/
COPY worlds/ worlds/

# Use Python 3.12 for dm-tree wheel compatibility
RUN uv sync --python 3.12

# Default: run train_expert.py
ENTRYPOINT ["uv", "run", "src/train_expert.py"]
CMD []
