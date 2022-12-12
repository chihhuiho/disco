root=$DISCOPATH"/training/save/cifar100/"
repeat=1
batchsize=50


if [ $# -eq 0 ]
  then


for model in "WideResNet-28-10"  "WideResNet-34-10"
do

# log/results/cifar100/disco/<model>/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="disco" --model_name=$model --batch_size=$batchsize  --disco_path=$root"/pgd/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat 
# log/results/cifar100/disco/<model>/autoattack/result_2.txt
python cifar100.py --trial 2 --input_defense="disco" --model_name=$model --batch_size=$batchsize  --disco_path=$root"/bim/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat 
# log/results/cifar100/disco/<model>/autoattack/result_3.txt
python cifar100.py --trial 3 --input_defense="disco" --model_name=$model --batch_size=$batchsize  --disco_path=$root"/fgsm/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat 
done


else # debug mode


for model in "WideResNet-28-10"  "WideResNet-34-10"
do

# log/results/debug/cifar100/disco/<model>/autoattack/result_1.txt
python cifar100.py --trial 1 --input_defense="disco" --model_name=$model --batch_size=$batchsize  --disco_path=$root"/pgd/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat --debug
# log/results/debug/cifar100/disco/<model>/autoattack/result_2.txt
python cifar100.py --trial 2 --input_defense="disco" --model_name=$model --batch_size=$batchsize  --disco_path=$root"/bim/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat --debug
# log/results/debug/cifar100/disco/<model>/autoattack/result_3.txt
python cifar100.py --trial 3 --input_defense="disco" --model_name=$model --batch_size=$batchsize  --disco_path=$root"/fgsm/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat --debug
done


fi
