# Lightweight Ubuntu LTS base. Python is installed once in the image;
# project dependencies live in the project .venv (a Docker named volume).
FROM ubuntu:24.04

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Sao_Paulo \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Native libraries and tools only. Python packages are installed into .venv later.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        git-lfs \
        openssh-client \
        sudo \
        build-essential \
        pkg-config \
        cmake \
        ninja-build \
        ffmpeg \
        libgl1 \
        libglib2.0-0t64 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        jq \
        unzip \
        ripgrep \
        fd-find \
        procps \
        iproute2 \
        dnsutils \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# Pin the interpreter used to create the project's virtual environment.
RUN uv python install 3.13.18 \
    && PYTHON_BIN="$(uv python find --managed-python 3.13.18)" \
    && ln -sfn "${PYTHON_BIN}" /usr/local/bin/python \
    && ln -sfn "${PYTHON_BIN}" /usr/local/bin/python3 \
    && ln -sfn "${PYTHON_BIN}" /usr/local/bin/python3.13 \
    && /usr/local/bin/python --version \
    && rm -rf /root/.cache/uv

ENV PATH="/usr/local/bin:${PATH}"
