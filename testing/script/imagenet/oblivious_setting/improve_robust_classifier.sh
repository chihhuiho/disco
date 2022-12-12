root=$DISCOPATH"/training/save/imagenet/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"

repeat=1

if [ $# -eq 0 ]
  then

# log/results/imagenet/disco/Salman2020Do_R50/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=10  --disco_path=$DiscoPth --repeat=$repeat --model_name="Salman2020Do_R50" 
# log/results/imagenet/disco/Engstrom2019Robustness/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=10  --disco_path=$DiscoPth --repeat=$repeat --model_name="Engstrom2019Robustness"  
# log/results/imagenet/disco/Wong2020Fast/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=10  --disco_path=$DiscoPth --repeat=$repeat --model_name="Wong2020Fast" 

else # debug mode

# log/results/debug/imagenet/disco/Salman2020Do_R50/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=10  --disco_path=$DiscoPth --repeat=$repeat --model_name="Salman2020Do_R50" --debug 
# log/results/debug/imagenet/disco/Engstrom2019Robustness/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=10  --disco_path=$DiscoPth --repeat=$repeat --model_name="Engstrom2019Robustness" --debug 
# log/results/debug/imagenet/disco/Wong2020Fast/autoattack/result_1.txt
python imagenet.py --trial 1 --input_defense="disco" --batch_size=10  --disco_path=$DiscoPth --repeat=$repeat --model_name="Wong2020Fast" --debug 

fi
