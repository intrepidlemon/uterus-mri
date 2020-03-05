## Deep learning based on magnetic resonance imaging for predictingoutcome of uterine fibroid embolization
### Running this project

This project can be initiated using simple scripts.

1. Inspect and edit the [`config.py`](./config.py) file to match your local installation preferences, i.e. where the project should expect to find your data files as well as where the project should output your results
2. The project expects your data to be in a folder referenced by the environment variable: `$DATA_DIR`
3. Initialize the project (creating the necessary directory and databases) using [`setup.sh`](./setup.sh)
4. Run the project using [`run.sh`](./run.sh)
5. (optional) create a `pushover-secret.sh` file to receive [pushover](https://pushover.net/) notifications when your run is completed:

### Comments
- This project uses Pipenv for dependency management. To get the necessary packages, run pipenv install --skip-lock

