if [ $# -eq 0 ]
  then

# No defense
# log/results/cifar10/no_input_defense/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="no_input_defense" 

# Adversarailly trained baselines
# Robust classifier Carmon2019Unlabeled
# log/results/cifar10/no_input_defense/Carmon2019Unlabeled/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="no_input_defense" --model_name 'Carmon2019Unlabeled' 

# Robust classifer Rebuffi2021Fixing_70_16_cutmix_extra
# log/results/cifar10/no_input_defense/Rebuffi2021Fixing_70_16_cutmix_extra/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="no_input_defense" --model_name 'Rebuffi2021Fixing_70_16_cutmix_extra' --batch_size=50  

# Adversarailly trained baselines
# log/results/cifar10/bit_depth_reduction/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="bit_depth_reduction"  
# log/results/cifar10/jpeg/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="jpeg"  
# log/results/cifar10/randomization/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="randomization"  
# log/results/cifar10/medianfilter/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="medianfilter"  



else # debug mode


# No defense
# log/results/debug/cifar10/no_input_defense/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="no_input_defense" --debug 

# Adversarailly trained baselines
# Robust classifier Carmon2019Unlabeled
# log/results/debug/cifar10/no_input_defense/Carmon2019Unlabeled/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="no_input_defense" --model_name 'Carmon2019Unlabeled' --debug 

# Robust classifer Rebuffi2021Fixing_70_16_cutmix_extra
# log/results/debug/cifar10/no_input_defense/Rebuffi2021Fixing_70_16_cutmix_extra/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="no_input_defense" --model_name 'Rebuffi2021Fixing_70_16_cutmix_extra' --batch_size=50 --debug 

# Adversarailly trained baselines
# log/results/debug/cifar10/bit_depth_reduction/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="bit_depth_reduction" --debug 
# log/results/debug/cifar10/jpeg/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="jpeg" --debug 
# log/results/debug/cifar10/randomization/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="randomization" --debug 
# log/results/debug/cifar10/medianfilter/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="medianfilter" --debug 


fi

