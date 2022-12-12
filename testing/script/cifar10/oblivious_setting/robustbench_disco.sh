root=$DISCOPATH"/training/save/cifar10/"

repeat=1

if [ $# -eq 0 ]
  then

#log/results/cifar10/disco/Standard/autoattack/result_6.txt
python cifar10.py --trial 6 --input_defense="disco" --disco_path $root"pgd/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  
#log/results/cifar10/disco/Standard/autoattack/result_7.txt
python cifar10.py --trial 7 --input_defense="disco" --disco_path $root"fgsm/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat  
#log/results/cifar10/disco/Standard/autoattack/result_8.txt
python cifar10.py --trial 8 --input_defense="disco" --disco_path $root"bim/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat   


else # debug mode
#log/results/debug/cifar10/disco/Standard/autoattack/result_6.txt
python cifar10.py --trial 6 --input_defense="disco" --disco_path $root"pgd/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  --debug
#log/results/debug/cifar10/disco/Standard/autoattack/result_7.txt
python cifar10.py --trial 7 --input_defense="disco" --disco_path $root"fgsm/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat --debug  
#log/results/debug/cifar10/disco/Standard/autoattack/result_8.txt
python cifar10.py --trial 8 --input_defense="disco" --disco_path $root"bim/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat --debug  

fi

