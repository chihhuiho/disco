root=$DISCOPATH"/training/save/cifar100/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"

if [ $# -eq 0 ]
  then


for attack in "FGSM" "PGD" "BIM" "RFGSM"  "eotpgd" "tpgd" "ffgsm" "mifgsm" "apgd" "jitter" 
do
# log/defense_transfer/cifar100/disco/WideResNet-28-10/<attack>/result_1.txt
python cifar100_transfer_defense.py --attack=$attack  --trial 1 --input_defense="disco"  --disco_path=$DiscoPth 

# log/defense_transfer/cifar100/no_input_defense/Gowal2020Uncovering_extra/<attack>/result_1.txt
python cifar100_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Gowal2020Uncovering_extra' --batch_size=50 

# log/defense_transfer/cifar100/no_input_defense/Rebuffi2021Fixing_28_10_cutmix_ddpm/<attack>/result_1.txt
python cifar100_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Rebuffi2021Fixing_28_10_cutmix_ddpm' 
done

else # debug mode

for attack in "FGSM" #"PGD" "BIM" "RFGSM"  "eotpgd" "tpgd" "ffgsm" "mifgsm" "apgd" "jitter" 
do
# log/defense_transfer/debug/cifar100/disco/WideResNet-28-10/<attack>/result_1.txt
python cifar100_transfer_defense.py --attack=$attack  --trial 1 --input_defense="disco"  --disco_path=$DiscoPth --debug

# log/defense_transfer/debug/cifar100/no_input_defense/Gowal2020Uncovering_extra/<attack>/result_1.txt
python cifar100_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Gowal2020Uncovering_extra' --batch_size=50 --debug 

# log/defense_transfer/debug/cifar100/no_input_defense/Rebuffi2021Fixing_28_10_cutmix_ddpm/<attack>/result_1.txt
python cifar100_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Rebuffi2021Fixing_28_10_cutmix_ddpm' --debug 
done

fi
