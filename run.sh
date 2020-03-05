pipenv run python scripts/calculate_uae_features.py --folder $3 --features $4

pipenv run python error_analysis.py > errors

pipenv run python preprocess.py --n4

bash notify.sh "deep-mri preprocess complete"

bash run-model.sh $1 $2
