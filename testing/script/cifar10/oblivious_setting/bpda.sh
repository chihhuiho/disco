root=$DISCOPATH"/training/save/cifar10/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"

repeat=1

if [ $# -eq 0 ]
  then


# log/bpda_results/cifar10/disco/Standard/bpda/result_x.txt
python cifar10_bpda.py --trial 1 --adaptive --recursive_num 0 --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat
python cifar10_bpda.py --trial 2 --adaptive --recursive_num 1 --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat 
python cifar10_bpda.py --trial 3 --adaptive --recursive_num 2 --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat 


else # debug mode

# log/bpda_results/debug/cifar10/disco/Standard/bpda/result_x.txt
python cifar10_bpda.py --trial 1 --adaptive --recursive_num 0 --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat --debug
python cifar10_bpda.py --trial 2 --adaptive --recursive_num 1 --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat --debug
python cifar10_bpda.py --trial 3 --adaptive --recursive_num 2 --input_defense="disco" --disco_path=$DiscoPth --defense_disco_path=$DiscoPth --repeat=$repeat --debug

fi

