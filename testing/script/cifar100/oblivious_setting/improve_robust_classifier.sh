root=$DISCOPATH"/training/save/cifar100/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"

repeat=1

if [ $# -eq 0 ]
  then


# log/results/cifar100/disco/Gowal2020Uncovering_extra/autoattack/result_1.txt
python cifar100.py --trial 1  --input_defense="disco" --disco_path=$DiscoPth --repeat=$repeat  --model_name='Gowal2020Uncovering_extra' --batch_size=25 
# log/results/cifar100/disco/Rebuffi2021Fixing_70_16_cutmix_ddpm/autoattack/result_1.txt
python cifar100.py --trial 1  --input_defense="disco" --disco_path=$DiscoPth --repeat=$repeat  --model_name='Rebuffi2021Fixing_70_16_cutmix_ddpm' --batch_size=25  
# log/results/cifar100/disco/Pang2022Robustness_WRN70_16/autoattack/result_1.txt
python cifar100.py --trial 1  --input_defense="disco" --disco_path=$DiscoPth --repeat=$repeat  --model_name='Pang2022Robustness_WRN70_16' --batch_size=25  


else # debug mode

# log/results/debug/cifar100/disco/Gowal2020Uncovering_extra/autoattack/result_1.txt
python cifar100.py --trial 1  --input_defense="disco" --disco_path=$DiscoPth --repeat=$repeat  --model_name='Gowal2020Uncovering_extra' --batch_size=25 --debug
# log/results/debug/cifar100/disco/Rebuffi2021Fixing_70_16_cutmix_ddpm/autoattack/result_1.txt
python cifar100.py --trial 1  --input_defense="disco" --disco_path=$DiscoPth --repeat=$repeat  --model_name='Rebuffi2021Fixing_70_16_cutmix_ddpm' --batch_size=25  --debug
# log/results/debug/cifar100/disco/Pang2022Robustness_WRN70_16/autoattack/result_1.txt
python cifar100.py --trial 1  --input_defense="disco" --disco_path=$DiscoPth --repeat=$repeat  --model_name='Pang2022Robustness_WRN70_16' --batch_size=25  --debug

fi
