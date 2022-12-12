root=$DISCOPATH"/training/save/cifar10/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"

repeat=1
attack="bpda"
model="vgg16_bn"
adaptive_iter=1
trial=1
recusive_num=1


if [ $# -eq 0 ]
  then


# log/adaptive_attack_results/cifar10/disco/vgg16_bn/bpda/result_1.txt
python cifar10_adaptive_attack.py --trial $trial --adaptive --recursive_num=$recusive_num --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat --attack=$attack --model_name=$model  --adaptive_iter=$adaptive_iter --batch_size=50 

else # debug mode

# log/adaptive_attack_results/debug/cifar10/disco/vgg16_bn/bpda/result_1.txt
python cifar10_adaptive_attack.py --trial $trial --adaptive --recursive_num=$recusive_num --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat --attack=$attack --model_name=$model  --adaptive_iter=$adaptive_iter --batch_size=50 --debug

fi
