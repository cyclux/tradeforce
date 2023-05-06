# Tradeforce

Tradeforce is a comprehensive Python trading framework designed for
high-performance backtesting, hyperparameter optimization, and live trading.
By leveraging JIT machine code compilation and providing a robust set of features,
Tradeforce enables users to develop, test, and deploy trading strategies
with efficiency and optional scalability.

### Key features
- Numba accelerated simulations, processing 100k+ records / second / thread
- Customizable trading strategies
- Dedicated market server supporting 100+ simultaneous WebSocket connections
- Backend support for PostgreSQL and MongoDB
- Local caching of market data in Arrow format for rapid loading times
- Hyperparameter optimization through the Optuna framework
- Live trading capabilities
- Easy and flexible deployment via Docker
- Scalable to cluster environments, such as Kubernetes
- Jupyter Notebook integration for analysis and visualization of results


## Installation

### pip

#### Create virtual environment (optional)

```
# Note: The environment name "tradeforce" is arbitrary, can be changed.

# Create environment
python3 -m venv tradeforce

# Activate environment
source tradeforce/bin/activate
```

#### Install via pip

```
pip install tradeforce
```

### Docker

Docker needs to be installed on the host system. See https://docs.docker.com/get-docker/

## Usage

#### Run dedicated market server

##### Window Powershell

```
$env:RUN_MARKET_SERVER="True"; docker-compose up; Remove-Item Env:\RUN_MARKET_SERVER
```

##### Linux Bash

```
RUN_MARKET_SERVER="True" docker-compose up
```

#### Run JupyterLab

##### Window Powershell

```
$env:RUN_JUPYTERLAB="True"; docker-compose up; Remove-Item Env:\RUN_JUPYTERLAB
```

##### Linux Bash

```
RUN_JUPYTERLAB="True" docker-compose up
```

#### Get bash shell in container
```
docker exec -it tradeforce bash
```

### Configuration

Tradeforce is either configured through a Python dictionary or a YAML file.

See folder `examples`.

See `documentation` for more details on all the configuration options.


## DISCLAIMER
Use at your own risk! Tradeforce is currently in beta, and bugs may occur.
Furthermore, there is no guarantee that strategies that have performed well
in the past will continue to do so in the future.


## Build

### Get requirements.txt from container

cd /to/path/tradeforce/root
docker build -t tradeforce:build --target build .
docker create --name tradeforce_build tradeforce:build
docker cp tradeforce_build:/opt/tradeforce/requirements.txt ./requirements.txt

### Build production stage

cd /to/path/tradeforce/root
docker build -t cyclux/tradeforce:latest -t cyclux/tradeforce:0.0.1 --target production .
docker push -a cyclux/tradeforce