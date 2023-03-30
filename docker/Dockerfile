# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.10.10
ARG POETRY_VERSION=1.4.1
ARG APP_NAME=tradeforce
ARG APP_PATH=/opt/${APP_NAME}

# Stage 1: staging
FROM python:${PYTHON_VERSION} AS staging

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1
ARG POETRY_VERSION
ARG APP_PATH

RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} POETRY_HOME=${POETRY_HOME} python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
WORKDIR ${APP_PATH}

# Stage 2: development
FROM staging AS development
COPY pyproject.toml poetry.lock ./
RUN poetry install
COPY . .
# ENTRYPOINT ["poetry", "run", "python", "examples/live_trader.py"]

# Stage 3: build
FROM staging AS build
COPY pyproject.toml poetry.lock ./
RUN poetry install && \
    poetry build && \
    mv dist/*.whl /opt/${APP_NAME}.whl

# Stage 4: production
FROM python:${PYTHON_VERSION} AS production
COPY --from=build /opt/${APP_NAME}.whl /opt/
RUN pip install /opt/${APP_NAME}.whl && rm /opt/${APP_NAME}.whl
CMD ["tradeforce"]
