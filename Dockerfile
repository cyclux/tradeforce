# syntax=docker/dockerfile:1

#----------------------
# Setup build arguments
#----------------------

ARG DEBIAN_VERSION=bullseye
ARG PYTHON_VERSION=3.11.3
ARG POETRY_VERSION=1.4.1
ARG APP_NAME=tradeforce
ARG APP_PATH=/opt/${APP_NAME}

#----------------------
# Stage 1: staging
#----------------------

FROM python:${PYTHON_VERSION}-${DEBIAN_VERSION} AS staging

ARG POETRY_VERSION
ARG APP_PATH

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$POETRY_HOME/bin:$PATH"
WORKDIR ${APP_PATH}

#----------------------
# Stage 2: development
#----------------------
FROM staging AS development
ARG APP_PATH

WORKDIR ${APP_PATH}
COPY . .
RUN --mount=type=cache,target=/root/.cache \
    poetry install --with dev,docs

ENTRYPOINT ["/bin/bash"]

#----------------------
# Stage 3: build
#----------------------

FROM staging AS build
ARG POETRY_HOME
ARG APP_NAME
ARG APP_PATH

WORKDIR ${APP_PATH}

COPY pyproject.toml poetry.lock ./
COPY --from=development ${APP_PATH} ${APP_PATH}
RUN --mount=type=cache,target=/root/.cache \
    poetry install --without dev,docs \
    && poetry build --format wheel \
    && poetry export --without dev,docs --format requirements.txt --output requirements.txt

#----------------------
# Stage 4: production
#----------------------

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS production
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# User and group id for non-root user
ARG USER=tf_docker
ARG USER_UID=1000
ARG USER_GID=1000

ARG APP_NAME
ARG APP_PATH=/home/${USER}/${APP_NAME}

RUN apt-get update && apt-get install --no-install-recommends -y git

# Create non-root user
RUN groupadd --gid ${USER_GID} ${USER} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --home-dir /home/${USER} ${USER}
USER ${USER}
ENV PATH="/home/${USER}/.local/bin:$PATH"

COPY --from=build --chown=${USER}:${USER} /opt/${APP_NAME}/dist/*.whl /home/${USER}/
RUN pip install --user /home/${USER}/${APP_NAME}*.whl \
    && rm /home/${USER}/${APP_NAME}*.whl

WORKDIR /home/${USER}/user_code

ENTRYPOINT ["/bin/bash"]
