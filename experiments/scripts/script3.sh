CUDA_VISIBLE_DEVICES=0 python rewinding.py --dataset1 cifar10 --noise_1 0.05 --num_epochs 100 --sched cosine --lr1 0.01 --model_type vit --seed 1 --augmentation 0 &

CUDA_VISIBLE_DEVICES=1 python rewinding.py --dataset1 cifar10 --noise_1 0.1 --num_epochs 100 --sched cosine --lr1 0.01 --model_type vit --seed 1 --augmentation 0 &

CUDA_VISIBLE_DEVICES=2 python rewinding.py --dataset1 mnist --noise_1 0.05 --num_epochs 100 --sched cosine --lr1 0.01 --model_type vit --seed 1 --augmentation 0 &

CUDA_VISIBLE_DEVICES=3 python rewinding.py --dataset1 mnist --noise_1 0.1 --num_epochs 100 --sched cosine --lr1 0.01 --model_type vit --seed 1 --augmentation 0 &

CUDA_VISIBLE_DEVICES=4 python rewinding.py --dataset1 svhn --noise_1 0.05 --num_epochs 100 --sched cosine --lr1 0.01 --model_type vit --seed 1 --augmentation 0 &

CUDA_VISIBLE_DEVICES=5 python rewinding.py --dataset1 svhn --noise_1 0.1 --num_epochs 100 --sched cosine --lr1 0.01 --model_type vit --seed 1 --augmentation 0 &

CUDA_VISIBLE_DEVICES=6 python rewinding.py --dataset1 mnist --noise_1 0.1 --num_epochs 100 --sched cosine --lr1 0.01 --model_type vit --seed 1 --augmentation 1 &

CUDA_VISIBLE_DEVICES=7 python rewinding.py --dataset1 mnist --noise_1 0.05 --num_epochs 100 --sched cosine --lr1 0.01 --model_type vit --seed 1 --augmentation 1 &

