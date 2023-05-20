Deployment
==========

Tradeforce offers various deployment options. The most convenient and powerful
method is using the provided Docker Compose stack, which retrieves the latest image
from Docker Hub and sets up a database backend required for most use cases. There is
also the option to spin up a Jupyter Notebook server for analyzing and visualizing
results. Furthermore, the Docker ecosystem enables scalability through cluster
environments like Kubernetes.

Alternatively, Tradeforce can be installed via pip from PyPI in a virtual environment,
combined with a self-hosted database backend. Running without a database backend is
possible, but its use cases are limited. Generally, a local cache of market data should
be available at minimum (per default `local_cache` is set `True` and dumps the DB in
`arrow` format).