root=$DISCOPATH"/training/save/cifar10/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"
repeat=1

if [ $# -eq 0 ]
  then

# log/results/cifar10/disco/Rebuffi2021Fixing_70_16_cutmix_extra/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="disco" --repeat=$repeat --disco_path=$DiscoPth --model_name='Rebuffi2021Fixing_70_16_cutmix_extra' --batch_size=50    

# log/results/cifar10/disco/Rebuffi2021Fixing_70_16_cutmix_extra/autoattack_L2/result_1.txt
python cifar10.py --trial 1 --input_defense="disco" --repeat=$repeat --disco_path=$DiscoPth --model_name='Rebuffi2021Fixing_70_16_cutmix_extra' --batch_size=50 --norm="L2"  


else # debug mode

# log/results/debug/cifar10/disco/Rebuffi2021Fixing_70_16_cutmix_extra/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="disco" --repeat=$repeat --disco_path=$DiscoPth --model_name='Rebuffi2021Fixing_70_16_cutmix_extra' --batch_size=50  --debug  

# log/results/debug/cifar10/disco/Rebuffi2021Fixing_70_16_cutmix_extra/autoattack_L2/result_1.txt
python cifar10.py --trial 1 --input_defense="disco" --repeat=$repeat --disco_path=$DiscoPth --model_name='Rebuffi2021Fixing_70_16_cutmix_extra' --batch_size=50 --norm="L2"  --debug 


fi
 
