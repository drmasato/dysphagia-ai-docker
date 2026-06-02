FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl wget libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    TotalSegmentator==2.2.1 \
    SimpleITK==2.3.1 \
    pydicom>=3.0.0 \
    nibabel==5.2.0 \
    numpy==1.26.4 \
    scipy==1.12.0 \
    pandas==2.2.0

RUN mkdir -p /input /output /workspace

WORKDIR /workspace
COPY quantify_muscles.py .

CMD ["python3", "quantify_muscles.py", "--help"]
