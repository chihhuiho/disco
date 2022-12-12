bs=100

if [ $# -eq 0 ]
  then

# Adversarially trained baselines
# log/results/imagenet/no_input_defense/Salman2020Do_R18/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="no_input_defense" --batch_size=$bs --model_name 'Salman2020Do_R18'  
# log/results/imagenet/no_input_defense/Engstrom2019Robustness/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="no_input_defense" --batch_size=$bs --model_name="Engstrom2019Robustness"  

# Transformation based baselines
# log/results/imagenet/bit_depth_reduction/Standard_R50/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="bit_depth_reduction" --batch_size=$bs 
# log/results/imagenet/jpeg/Standard_R50/autoattack/result_1.txt 
python imagenet.py --trial 1 --input_defense="jpeg" --batch_size=$bs 
# log/results/imagenet/randomization/Standard_R50/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="randomization" --batch_size=$bs 
# log/results/imagenet/medianfilter/Standard_R50/autoattack/result_1.txt 
python imagenet.py --trial 1 --input_defense="medianfilter"  --batch_size=$bs  

else # debug mode

# Adversarially trained baselines
# log/results/debug/imagenet/no_input_defense/Salman2020Do_R18/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="no_input_defense" --batch_size=$bs --model_name 'Salman2020Do_R18' --debug 
# log/results/debug/imagenet/no_input_defense/Engstrom2019Robustness/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="no_input_defense" --batch_size=$bs --model_name="Engstrom2019Robustness"  --debug 

# Transformation based baselines
# log/results/debug/imagenet/bit_depth_reduction/Standard_R50/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="bit_depth_reduction" --batch_size=$bs --debug
# log/results/debug/imagenet/jpeg/Standard_R50/autoattack/result_1.txt 
python imagenet.py --trial 1 --input_defense="jpeg" --batch_size=$bs --debug 
# log/results/debug/imagenet/randomization/Standard_R50/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="randomization" --batch_size=$bs --debug
# log/results/debug/imagenet/medianfilter/Standard_R50/autoattack/result_1.txt 
python imagenet.py --trial 1 --input_defense="medianfilter"  --batch_size=$bs --debug 

fi
