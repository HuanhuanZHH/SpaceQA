FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN python -m pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 spaceqa
USER spaceqa

ENTRYPOINT ["spaceqa"]
CMD ["--help"]
