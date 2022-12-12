root=$DISCOPATH"/training/save/imagenet/"
repeat=1


if [ $# -eq 0 ]
  then

# log/results/imagenet/disco/Standard_R50/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=50  --disco_path=$root"pgd/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat   
# log/results/imagenet/disco/Standard_R50/autoattack/result_2.txt
ipython imagenet.py --trial 2 --input_defense="disco" --batch_size=50  --disco_path=$root"bim/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat   
# log/results/imagenet/disco/Standard_R50/autoattack/result_3.txt
python imagenet.py --trial 3 --input_defense="disco" --batch_size=50  --disco_path=$root"fgsm/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat   

else # debug mode

# log/results/debug/imagenet/disco/Standard_R50/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=50  --disco_path=$root"pgd/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat  --debug 
# log/results/debug/imagenet/disco/Standard_R50/autoattack/result_2.txt
python imagenet.py --trial 2 --input_defense="disco" --batch_size=50  --disco_path=$root"bim/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat  --debug 
# log/results/debug/imagenet/disco/Standard_R50/autoattack/result_3.txt
python imagenet.py --trial 3 --input_defense="disco" --batch_size=50  --disco_path=$root"fgsm/imnet_mlp/trial_1/epoch-best.pth" --repeat=$repeat  --debug 


fi
