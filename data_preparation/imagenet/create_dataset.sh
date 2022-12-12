# Remove --debug to create the full dataset
# The --attack flag creates adv-clean pairs using different attack methods
python create_dataset.py --attack "pgd"  --debug 
python create_dataset.py --attack "fgsm" --debug 
python create_dataset.py --attack "bim"  --debug 

'''
# since the clean image is the same for all dataset
# use the skip clean flag to skip the clean image generation and reuse the folder that contains the clean images
python create_dataset.py --attack "pgd" --skip_clean --debug 
python create_dataset.py --attack "fgsm" --skip_clean --debug 
python create_dataset.py --attack "bim" --skip_clean --debug 
'''
