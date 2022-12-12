root=$DISCOPATH"/training/save/imagenet/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"

batch_size=10
repeat=1

if [ $# -eq 0 ]
  then

for attack in "FGSM" "PGD" "BIM" "RFGSM"  "eotpgd" "tpgd" "ffgsm" "mifgsm" "apgd" "jitter" 
do
# log/defense_transfer/imagenet/disco/Standard_R50/<attack>/result_1.txt
python imagenet_transfer_defense.py --attack=$attack  --trial 1 --input_defense="disco" --disco_path $DiscoPth --batch_size=$batch_size  
# log/defense_transfer/imagenet/no_input_defense/Salman2020Do_R50/<attack>/result_1.txt
python imagenet_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Salman2020Do_R50' --batch_size=$batch_size  
# log/defense_transfer/imagenet/no_input_defense/Engstrom2019Robustness/<attack>/result_1.txt
python imagenet_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Engstrom2019Robustness' --batch_size=$batch_size  
done


else # debug mode

for attack in "FGSM" #"PGD" "BIM" "RFGSM"  "eotpgd" "tpgd" "ffgsm" "mifgsm" "apgd" "jitter" 
do
# log/defense_transfer/debug/imagenet/disco/Standard_R50/FGSM/result_1.txt
python imagenet_transfer_defense.py --attack=$attack  --trial 1 --input_defense="disco" --disco_path $DiscoPth --batch_size=$batch_size  --debug 
# log/defense_transfer/debug/imagenet/no_input_defense/Salman2020Do_R50/FGSM/result_1.txt
python imagenet_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Salman2020Do_R50' --batch_size=$batch_size --debug 
# log/defense_transfer/debug/imagenet/no_input_defense/Engstrom2019Robustness/FGSM/result_1.txt
python imagenet_transfer_defense.py --attack=$attack  --trial 1 --input_defense="no_input_defense" --model_name='Engstrom2019Robustness' --batch_size=$batch_size --debug 
done

fi
