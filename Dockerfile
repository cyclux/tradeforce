# syntax=docker/dockerfile:1

# https://gist.github.com/usr-ein/c42d98abca3cb4632ab0c2c6aff8c88a

ARG PYTHON_VERSION=3.10.10
# ARG PYTHON_VERSION=3.11.2 # Waiting for numba 0.57.0rc1 -> https://github.com/numba/numba/issues/8841
ARG POETRY_VERSION=1.4.1
ARG APP_NAME=tradeforce
ARG APP_PATH=/opt/${APP_NAME}

####################
# Stage 1: staging #
####################

FROM python:${PYTHON_VERSION} AS staging

ENV POETRY_HOME="/opt/poetry" \
    #POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on
ARG POETRY_VERSION
ARG APP_PATH

# POETRY_VERSION=${POETRY_VERSION} POETRY_HOME=${POETRY_HOME}
RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python3 -
# RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
WORKDIR ${APP_PATH}

########################
# Stage 2: development #
########################
FROM staging AS development
ARG APP_PATH
WORKDIR ${APP_PATH}
COPY . .
# COPY pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=/root/.cache \
    poetry install --with=dev
# RUN poetry install --with=dev

ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["poetry", "run", "python", "examples/live_trader.py"]

##################
# Stage 3: build #
##################

FROM staging AS build
ARG POETRY_HOME
ARG APP_NAME
ARG APP_PATH

WORKDIR ${APP_PATH}

COPY pyproject.toml poetry.lock ./
COPY --from=development $APP_PATH $APP_PATH

RUN --mount=type=cache,target=/root/.cache \
    poetry install --without dev \
    && poetry build --format wheel

#######################
# Stage 4: production #
#######################

FROM python:${PYTHON_VERSION} AS production
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ARG APP_NAME
ARG APP_PATH

COPY --from=build ${APP_PATH}/dist/*.whl ./
RUN pip install ./${APP_NAME}*.whl \
    && rm ./${APP_NAME}*.whl

COPY ./docker/* /docker/
WORKDIR /docker

ENTRYPOINT ["python"]
