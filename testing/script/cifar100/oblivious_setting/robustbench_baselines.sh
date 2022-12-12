batchsize=32
model="WideResNet-34-10"
#model="WideResNet-28-10"

if [ $# -eq 0 ]
  then

# log/results/cifar100/no_input_defense/WideResNet-34-10/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="no_input_defense" --model_name=$model --batch_size=$batchsize 

# Transformation based baselines
# log/results/cifar100/medianfilter/WideResNet-34-10/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="medianfilter" --model_name=$model --batch_size=$batchsize  
# log/results/cifar100/bit_depth_reduction/WideResNet-34-10/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="bit_depth_reduction" --model_name=$model --batch_size=$batchsize 
# log/results/cifar100/jpeg/WideResNet-34-10/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="jpeg" --model_name=$model --batch_size=$batchsize 
# log/results/cifar100/randomization/WideResNet-34-10/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="randomization" --model_name=$model --batch_size=$batchsize  

else # debug mode

# log/results/debug/cifar100/no_input_defense/WideResNet-34-10/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="no_input_defense" --model_name=$model --batch_size=$batchsize --debug 


# Transformation based baselines
# log/results/debug/cifar100/medianfilter/WideResNet-34-10/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="medianfilter" --model_name=$model --batch_size=$batchsize --debug 
# log/results/debug/cifar100/bit_depth_reduction/WideResNet-34-10/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="bit_depth_reduction" --model_name=$model --batch_size=$batchsize --debug 
# log/results/debug/cifar100/jpeg/WideResNet-34-10/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="jpeg" --model_name=$model --batch_size=$batchsize --debug 
# log/results/debug/cifar100/randomization/WideResNet-34-10/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="randomization" --model_name=$model --batch_size=$batchsize  --debug 

fi
