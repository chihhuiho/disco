root=$DISCOPATH"/training/save/cifar10/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"
repeat=1

if [ $# -eq 0 ]
  then

for attack in "FGSM" "PGD" "BIM" "RFGSM"  "eotpgd" "tpgd" "ffgsm" "mifgsm" "apgd" "jitter" 
do
# log/defense_transfer/cifar10/disco/Standard/<attack>/result_1.txt
python cifar10_transfer_defense.py --attack=$attack  --trial 1 --input_defense="disco" --disco_path $DiscoPth  

# cd log/defense_transfer/cifar10/no_input_defense/Rebuffi2021Fixing_70_16_cutmix_extra/<attack>/result_1.txt 
python cifar10_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Rebuffi2021Fixing_70_16_cutmix_extra'  --batch_size=50

# cd log/defense_transfer/cifar10/no_input_defense/Gowal2021Improving_28_10_ddpm_100m/<attack>/result_1.txt
python cifar10_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Gowal2021Improving_28_10_ddpm_100m' --batch_size=50
done


else # debug mode

for attack in "FGSM" #"PGD" "BIM" "RFGSM"  "eotpgd" "tpgd" "ffgsm" "mifgsm" "apgd" "jitter" 
do
# log/defense_transfer/debug/cifar10/disco/Standard/<attack>/result_1.txt
python cifar10_transfer_defense.py --attack=$attack  --trial 1 --input_defense="disco" --disco_path $DiscoPth  --debug

# cd log/defense_transfer/debug/cifar10/no_input_defense/Rebuffi2021Fixing_70_16_cutmix_extra/<attack>/result_1.txt 
python cifar10_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Rebuffi2021Fixing_70_16_cutmix_extra' --batch_size=50 --debug 

# cd log/defense_transfer/debug/cifar10/no_input_defense/Gowal2021Improving_28_10_ddpm_100m/<attack>/result_1.txt
python cifar10_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Gowal2021Improving_28_10_ddpm_100m' --batch_size=50  --debug 
done

fi
