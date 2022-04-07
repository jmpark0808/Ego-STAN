#!/bin/bash
#SBATCH --gres=gpu:1      
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32000M       
#SBATCH --time=1-12:00     
#SBATCH --account=rrg-pfieguth

module load python/3.9 cuda cudnn

# Prepare virtualenv
source ~/torch/bin/activate

# Download training set
for f in ~/projects/def-pfieguth/mo2cap/TrainSet/*;
 do
  tar -xvf $f -C $SLURM_TMPDIR;
 done

# Download test set
tar -xvf ~/projects/def-pfieguth/mo2cap/TestSet.tar.gz -C $SLURM_TMPDIR

logdir=/home/s42hossa/scratch/experiments/throwaway

# Start training
tensorboard --logdir=${logdir} --host 0.0.0.0 --load_fast false & \
    python ~/projects/def-pfieguth/s42hossa/xREgoPose/train.py \
    --model xregopose \
    --dataloader mo2cap2 \
    --logdir ${logdir} \
    --dataset_tr $SLURM_TMPDIR/TrainSet \
    --dataset_val $SLURM_TMPDIR/TestSet \
    --dataset_test $SLURM_TMPDIR/TestSet \
    --batch_size 16 \
    --epoch 1 \
    --num_workers 24 \
    --lr 0.001 \
    --es_patience 7 \
    --display_freq 64 \
    --load_resnet /home/s42hossa/projects/def-pfieguth/s42hossa/resnet101-63fe2227.pth \
    --eval True
