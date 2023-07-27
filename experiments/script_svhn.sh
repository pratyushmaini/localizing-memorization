id=0
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)

while [ $free_mem -lt 30000 ]; do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
    sleep 5
    echo $free_mem 
done

for NOISE in 0.05 0.1 0.2
do
    for SEED in 0 1 2
    do
        for DATASET in svhn
        do
            for MODEL in resnet50 resnet9
            do
                # echo $COUNTER
                CUDA_VISIBLE_DEVICES=$SEED python rewinding.py --dataset1 $DATASET --noise_1 $NOISE --num_epochs 50 --sched cosine --lr1 0.001 --model_type $MODEL --seed $SEED --augmentation 0 &
            done
        done
    done
done

