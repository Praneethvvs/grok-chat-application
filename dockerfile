#------------------INTERMEDIATE BUILD------------------------
FROM python:3.10.0-slim as builder

RUN pip install poetry==1.7.1

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=${POETRY_VERSION} \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install --without dev --no-root


#------------------FINAL BUILD------------------------
FROM python:3.10.0-slim

WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:${PWD}
ENV PYTHONUNBUFFERED=1

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY . /app

CMD ["uvicorn", "azure_function:app", "--host", "0.0.0.0", "--port", "8080"]