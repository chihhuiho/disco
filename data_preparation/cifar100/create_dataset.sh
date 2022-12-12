# Remove --debug to create the full dataset
# The --attack flag creates adv-clean pairs using different attack methods
python create_dataset.py --attack "pgd" --debug
python create_dataset.py --attack "fgsm" --debug
python create_dataset.py --attack "bim" --debug
