FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV nnUNet_raw="/data/nnUNet_raw"
ENV nnUNet_preprocessed="/data/nnUNet_preprocessed"
ENV nnUNet_results="/data/nnUNet_results"

RUN apt-get update && apt-get install -y \
    git curl wget libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    nnunetv2==2.4.1 \
    SimpleITK==2.3.1 \
    pydicom==2.4.3 \
    nibabel==5.2.0 \
    numpy==1.26.4 \
    scipy==1.12.0

RUN mkdir -p /data/nnUNet_raw /data/nnUNet_preprocessed /data/nnUNet_results \
             /input /output /workspace

WORKDIR /workspace
COPY inference.py .

CMD ["python3", "inference.py", "--help"]
