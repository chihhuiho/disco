root=$DISCOPATH"/training/save/cifar10/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"

repeat=1
attack="bpda"
model="vgg16_bn"


if [ $# -eq 0 ]
  then


for r in 1 2 3 
do
# log/bpda_speed/defense/result_r.txt
python cifar10_adaptive_attack_speed.py  --adaptive --recursive_num=$r --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat --attack=$attack --model_name=$model  --adaptive_iter=1  --measure="defense" 

# log/bpda_speed/attack/result_r.txt
python cifar10_adaptive_attack_speed.py  --adaptive --recursive_num=1 --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat --attack=$attack --model_name=$model  --adaptive_iter=$r --measure="attack" 
done

else # debug mode

for r in 1 2 3 
do
# log/bpda_speed/debug/defense/result_r.txt
python cifar10_adaptive_attack_speed.py  --adaptive --recursive_num=$r --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat --attack=$attack --model_name=$model  --adaptive_iter=1  --measure="defense" --debug

# log/bpda_speed/debug/attack/result_r.txt
python cifar10_adaptive_attack_speed.py  --adaptive --recursive_num=1 --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat --attack=$attack --model_name=$model  --adaptive_iter=$r --measure="attack" --debug
done

fi

