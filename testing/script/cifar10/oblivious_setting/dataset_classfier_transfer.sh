repeat=1
# Reproducing Table 7 in the paper
Cifar10Root=$DISCOPATH"/training/save/cifar10/"


if [ $# -eq 0 ]
  then


# DISCO PGD CIFAR10 + Res18
# log/results/cifar10/disco/resnet18/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="disco" --disco_path $Cifar10Root"pgd/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat --model_name="resnet18"  

# DISCO PGD CIFAR10 + VGG16
# log/results/cifar10/disco/vgg16_bn/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="disco" --disco_path $Cifar10Root"pgd/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat --model_name="vgg16_bn"  

# DISCO PGD CIFAR10 + WRN28
# log/results/cifar10/disco/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="disco" --disco_path $Cifar10Root"pgd/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  

# DISCO BIM CIFAR10 + WRN28
# log/results/cifar10/disco/Standard/autoattack/result_2.txt
python cifar10.py --trial 2 --input_defense="disco" --disco_path $Cifar10Root"bim/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  

# DISCO FGSM CIFAR10 + WRN28
# log/results/cifar10/disco/Standard/autoattack/result_3.txt
python cifar10.py --trial 3 --input_defense="disco" --disco_path $Cifar10Root"fgsm/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  

# DISCO FGSM CIFAR100 + WRN28
# log/results/cifar10/disco/Standard/autoattack/result_4.txt
Cifar100Root=$DISCOPATH"/training/save/cifar100/"
python cifar10.py --trial 4 --input_defense="disco" --disco_path $Cifar100Root"fgsm/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  

# DISCO FGSM Imagenet + WRN28
# log/results/cifar10/disco/Standard/autoattack/result_5.txt
ImageNetRoot=$DISCOPATH"/training/save/imagenet/"
python cifar10.py --trial 5 --input_defense="disco" --disco_path $ImageNetRoot"fgsm/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  


else # debug mode


# DISCO PGD CIFAR10 + Res18
# log/results/debug/cifar10/disco/resnet18/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="disco" --disco_path $Cifar10Root"pgd/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat --model_name="resnet18"  --debug

# DISCO PGD CIFAR10 + VGG16
# log/results/debug/cifar10/disco/vgg16_bn/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="disco" --disco_path $Cifar10Root"pgd/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat --model_name="vgg16_bn"  --debug

# DISCO PGD CIFAR10 + WRN28
# log/results/debug/cifar10/disco/Standard/autoattack/result_1.txt
python cifar10.py --trial 1 --input_defense="disco" --disco_path $Cifar10Root"pgd/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  --debug

# DISCO BIM CIFAR10 + WRN28
# log/results/debug/cifar10/disco/Standard/autoattack/result_2.txt
python cifar10.py --trial 2 --input_defense="disco" --disco_path $Cifar10Root"bim/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  --debug

# DISCO FGSM CIFAR10 + WRN28
# log/results/debug/cifar10/disco/Standard/autoattack/result_3.txt
python cifar10.py --trial 3 --input_defense="disco" --disco_path $Cifar10Root"fgsm/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  --debug

# DISCO FGSM CIFAR100 + WRN28
# log/results/debug/cifar10/disco/Standard/autoattack/result_4.txt
Cifar100Root=$DISCOPATH"/training/save/cifar100/"
python cifar10.py --trial 4 --input_defense="disco" --disco_path $Cifar100Root"fgsm/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  --debug

# DISCO FGSM Imagenet + WRN28
# log/results/debug/cifar10/disco/Standard/autoattack/result_5.txt
ImageNetRoot=$DISCOPATH"/training/save/imagenet/"
python cifar10.py --trial 5 --input_defense="disco" --disco_path $ImageNetRoot"fgsm/imnet_mlp/trial_1/epoch-best.pth"  --repeat=$repeat  --debug

fi
