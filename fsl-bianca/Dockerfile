FROM brainlife/fsl:6.0.4

ENV FSLDIR=/usr/local/fsl
ENV PATH="${FSLDIR}/bin:${PATH}"
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV DEBIAN_FRONTEND=noninteractive

# Python 3.8 と pip インストール（Ubuntu 18.04 Bionic対応）
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-distutils python3-pip \
    dc bc \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && rm -rf /var/lib/apt/lists/*

# pip アップグレード後にライブラリ追加（Python 3.8対応バージョン）
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    nibabel==3.2.2 \
    numpy==1.24.4 \
    scipy==1.9.3 \
    pandas==1.5.3

RUN mkdir -p /input /output /workspace

WORKDIR /workspace
COPY bianca_pipeline.sh quantify_wmh.py ./
RUN chmod +x bianca_pipeline.sh

CMD ["bash", "bianca_pipeline.sh", "--help"]
