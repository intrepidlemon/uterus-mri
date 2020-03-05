pipenv run python run.py --description $1 --model v1 --trials $2 --form all --label outcome --hyperparameters hyperparameters.json --split a5bdb694-b086-41d1-9105-34b65c52593a
pipenv run python run.py --description $1 --model v1 --trials $2 --form t1 --label outcome --hyperparameters hyperparameters.json --split a5bdb694-b086-41d1-9105-34b65c52593a
pipenv run python run.py --description $1 --model v1 --trials $2 --form t2 --label outcome --hyperparameters hyperparameters.json --split a5bdb694-b086-41d1-9105-34b65c52593a
pipenv run python run.py --description $1 --model v1 --trials $2 --form features --label outcome --hyperparameters hyperparameters.json --split a5bdb694-b086-41d1-9105-34b65c52593a
bash notify.sh "deep-mri all trials complete"
