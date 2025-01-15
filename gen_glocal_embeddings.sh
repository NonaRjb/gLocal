#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 150GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 2-00:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/gLocal/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/gLocal/logs/%J_slurm.err

data_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_meg/images"
embeddings_dir="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_meg_embeddings_tmp/gLocal"
save_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_meg/image_embeddings"
transform_path="/proj/rep-learning-robotics/users/x_nonra/gLocal/transforms/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/visual/transform.npz"
model_name="dino-vit-base-p16"
dataset="things-meg"

cd /proj/rep-learning-robotics/users/x_nonra/gLocal

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

export HF_HOME="/proj/rep-learning-robotics/users/x_nonra/.cache"
export TORCH_HOME="/proj/rep-learning-robotics/users/x_nonra/.cache"
export XDG_CACHE_HOME="/proj/rep-learning-robotics/users/x_nonra/.cache"

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate glocal

python main_gen_orig_embeddings.py --embeddings_dir "$embeddings_dir" --data_root "$data_path" --model_name "$model_name" --dataset "$dataset" --split "train"
python main_gen_orig_embeddings.py --embeddings_dir "$embeddings_dir" --data_root "$data_path" --model_name "$model_name" --dataset "$dataset" --split "test"

python main_things_embedding_gen.py --embeddings_dir "$embeddings_dir" --data_root "$data_path" --transform_path "$transform_path" \
--model_name "$model_name" --update_transforms

python split_embeddings.py --embeddings_dir "$embeddings_dir" --output_dir "$save_path" --model_name "$model_name" --split "train"
python split_embeddings.py --embeddings_dir "$embeddings_dir" --output_dir "$save_path" --model_name "$model_name" --split "test"