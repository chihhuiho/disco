root=$DISCOPATH"/training/save/cifar100/"

repeat=1


if [ $# -eq 0 ]
  then

# log/results/cifar100/disco/WideResNet-34-10/autoattack/result_4.txt
python cifar100.py --trial 4 --input_defense="disco" --disco_path $root"pgd/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  
# log/results/cifar100/disco/WideResNet-34-10/autoattack/result_5.txt
python cifar100.py --trial 5 --input_defense="disco" --disco_path $root"fgsm/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat   
# log/results/cifar100/disco/WideResNet-34-10/autoattack/result_6.txt
python cifar100.py --trial 6 --input_defense="disco" --disco_path $root"bim/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat   


else # debug mode

# log/results/debug/cifar100/disco/WideResNet-34-10/autoattack/result_4.txt
python cifar100.py --trial 4 --input_defense="disco" --disco_path $root"pgd/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  --debug
# log/results/debug/cifar100/disco/WideResNet-34-10/autoattack/result_5.txt
python cifar100.py --trial 5 --input_defense="disco" --disco_path $root"fgsm/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat --debug  
# log/results/debug/cifar100/disco/WideResNet-34-10/autoattack/result_6.txt
python cifar100.py --trial 6 --input_defense="disco" --disco_path $root"bim/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat --debug  

fi



