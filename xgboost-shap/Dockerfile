FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl wget build-essential \
    ca-certificates gnupg lsb-release \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
       | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/debian $(lsb_release -cs) stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    xgboost==2.0.3 \
    shap==0.44.1 \
    scikit-learn==1.4.0 \
    imbalanced-learn==0.12.0 \
    optuna==3.5.0 \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.5.3 \
    numpy==1.26.4 \
    pandas==2.2.0 \
    matplotlib==3.8.2 \
    joblib==1.3.2 \
    nibabel==5.2.0 \
    SimpleITK==2.3.1

RUN mkdir -p /workspace/models /workspace/data /workspace/plots

WORKDIR /workspace
COPY train_model.py predict.py api_server.py visualize.py ./

EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
