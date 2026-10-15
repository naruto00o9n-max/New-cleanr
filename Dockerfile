FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONUNBUFFERED=1 CLEANER_DATA_DIR=/app/data CLEANER_MODELS_DIR=/app/models
COPY pyproject.toml README.md ./
COPY manga_cleaner ./manga_cleaner
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir .
RUN manhwa-cleaner prepare-models
RUN useradd --uid 10001 --create-home cleaner && mkdir -p /app/data && chown -R cleaner:cleaner /app
USER cleaner
CMD ["manhwa-cleaner", "bot"]
