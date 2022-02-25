#!/bin/bash
#SBATCH --gres=gpu:1      
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32000M       
#SBATCH --time=1-12:00     
#SBATCH --account=rrg-pfieguth
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.9 cuda cudnn

# Prepare virtualenv
source ~/torch/bin/activate


# Prepare data
dataset_dir=~/projects/def-pfieguth/xREgoPose/xR-EgoPose/data/Dataset
declare -a a_test=("female_004_a_a"
				"female_008_a_a"
				"female_010_a_a"
				"female_012_a_a"
				"female_012_f_s"
				"male_001_a_a"
				"male_002_a_a"
				"male_004_f_s"
				"male_006_a_a"
				"male_007_f_s"
				"male_010_a_a"
				"male_014_f_s")
declare -a max_test=("i" "i" "i" "f" "a" "i" "j" "a" "i" "a" "i" "a")
declare -a a_train=("female_001_a_a"
				"female_002_a_a"
				"female_002_f_s"
				"female_003_a_a"
				"female_005_a_a"	
				"female_006_a_a"
				"female_007_a_a"
				"female_009_a_a"
				"female_011_a_a"
				"female_014_a_a"
				"female_015_a_a"
				"male_003_f_s"
				"male_004_a_a"
				"male_005_a_a"
				"male_006_f_s"
				"male_007_a_a"
				"male_008_f_s"
				"male_009_a_a"
				"male_010_f_s"
				"male_011_f_s"
				"male_014_a_a")
declare -a max_train=("i" "j" "a" "f" "i" "i" "h" "i" "f" "f" "j" "a" "i" "j" "a" "i" "a" "h" "a" "a" "i")
declare -a a_val=("male_008_a_a")
declare -a max_val=("i")

download_set () {

	case "$1" in

		ValSet)
			echo "ValSet"
			arr=("${a_val[@]}")
			max=("${max_val[@]}")
			;;

		TestSet)
			echo "TestSet"
			arr=("${a_test[@]}")
			max=("${max_test[@]}")
			;;

		TrainSet)
			echo "TrainSet"
			arr=("${a_train[@]}")
			max=("${max_train[@]}")
			;;

		*)
			break
			;;

	esac
	  
	cd $1

	for i in "${!arr[@]}"
	do
		s=${arr[$i]}
		m=${max[$i]}
        
	mkdir -p $SLURM_TMPDIR/$1
        # extract data
        cat $s.tar.gz.part?? | unpigz -p 32  | tar -xvC $SLURM_TMPDIR/$1

   	done

	cd ..
}

# Abort on error
set -e

# Create the destination directory if it doesn't exist yet
mkdir -p ${dataset_dir}
cd ${dataset_dir}

# Download and process Train set
download_set "TrainSet"
download_set "ValSet"
download_set "TestSet"

logdir=/home/kkaai/projects/def-pfieguth/kkaai/

# Start training
tensorboard --logdir=${logdir} --host 0.0.0.0 --load_fast false & \
    python ~/projects/def-pfieguth/kkaai/xREgoPose/train.py \
    --model direct_regression \
    --dataloader baseline \
    --eval True \
    --logdir ${logdir} \
    --dataset_tr $SLURM_TMPDIR/TrainSet \
    --dataset_val $SLURM_TMPDIR/ValSet \
    --dataset_test $SLURM_TMPDIR/TestSet \
    --seed 42 \
    --batch_size 16 \
    --epoch 20 \
    --num_workers 24 \
    --lr 0.001 \
    --es_patience 7 \
    --display_freq 64 \
    --val_freq 2000 \
    --load_resnet /home/kkaai/projects/def-pfieguth/xREgoPose/xR-EgoPose/resnet101-63fe2227.pth
