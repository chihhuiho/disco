root=$DISCOPATH"/training/save/imagenet/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"

repeat=1

if [ $# -eq 0 ]
  then

# log/results/imagenet/disco/inceptionv3/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=50  --disco_path=$DiscoPth --repeat=$repeat --model_name="inceptionv3"  
# log/results/imagenet/disco/resnet18/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=50  --disco_path=$DiscoPth --repeat=$repeat --model_name="resnet18"  
# log/results/imagenet/disco/wide_resnet50_2/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=50  --disco_path=$DiscoPth --repeat=$repeat --model_name="wide_resnet50_2"  

else # debug mode

# log/results/debug/imagenet/disco/inceptionv3/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=50  --disco_path=$DiscoPth --repeat=$repeat --model_name="inceptionv3" --debug 
# log/results/debug/imagenet/disco/resnet18/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=50  --disco_path=$DiscoPth --repeat=$repeat --model_name="resnet18" --debug 
# log/results/debug/imagenet/disco/wide_resnet50_2/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=50  --disco_path=$DiscoPth --repeat=$repeat --model_name="wide_resnet50_2" --debug 

fi
