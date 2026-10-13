FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 CLEANER_DATA_DIR=/app/data CLEANER_MODELS_DIR=/app/models
COPY pyproject.toml README.md ./
COPY manga_cleaner ./manga_cleaner
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir .
RUN manhwa-cleaner prepare-models
RUN useradd --uid 10001 --create-home cleaner && mkdir -p /app/data && chown -R cleaner:cleaner /app
USER cleaner
CMD ["manhwa-cleaner", "bot"]
