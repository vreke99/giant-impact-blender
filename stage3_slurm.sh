#!/bin/bash -l

#
#   COSMA SLURM job array for Stage 3: each array task takes one GPU node and
#   renders its share of the frame range there with render_chunks.sh, which
#   runs several Blender processes at once on that node's GPU.
#
#   Usage
#   -----
#   sbatch stage3_slurm.sh                  # frames 0-2160 on up to 4 GPU nodes
#   sbatch stage3_slurm.sh 0 599            # a sub-range (START END)
#   sbatch --array=0 stage3_slurm.sh 0 599  # all on one node
#   sbatch -w mad06 --array=0 stage3_slurm.sh   # on mad06 only
#
#   Set MODE and CAMERA (the template) in render_chunks.sh as before.  The frame
#   range is split evenly over the array tasks, so --array alone sets how many
#   nodes are used.  Each task writes stage3_JOB_TASK.out/.err, and each of its
#   Blender processes its own render_S_to_E.log; a task fails (and says which
#   frames) if any of its frames failed.
#

#SBATCH --array=0-2                   # one task per GPU node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                  # check the name with: sinfo -p hardwarelab-gpu -o "%N %G"
#SBATCH --cpus-per-task=64            # MAX_PARALLEL renders share these cores
#SBATCH --mem=96G
#SBATCH --exclude=ga008               # AMD MI300A: no GPU support in this Blender build
#SBATCH -J Stage3_render
#SBATCH -o /cosma5/data/dp004/vreke/blender/outs_and_errs/stage3_%A_%a.out
#SBATCH -e /cosma5/data/dp004/vreke/blender/outs_and_errs/stage3_%A_%a.err
#SBATCH -p hardwarelab-gpu
#SBATCH -A durham                     # check the account for this queue
#SBATCH --mail-type=END
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH -t 3:00:00

#   This task's chunk of the frame range
START=${1:-0}
END=${2:-2160}
NCHUNK=${SLURM_ARRAY_TASK_COUNT:-1}
ID=${SLURM_ARRAY_TASK_ID:-0}
STEP=$(( (END - START + NCHUNK) / NCHUNK ))
S=$(( START + ID * STEP ))
E=$(( S + STEP - 1 )); (( E > END )) && E=$END
if (( S > END )); then echo "Task $ID: nothing to do"; exit 0; fi

#   Blender processes per GPU: 8 suits a 40 GB card for meshes and panoramas;
#   use ~5 for PARTICLES modes.  Threads share out the node's cores.
CORES=${SLURM_CPUS_ON_NODE:-$(nproc)}
export MAX_PARALLEL=${MAX_PARALLEL:-8}
export NTHREAD=$(( CORES / MAX_PARALLEL ))

echo "Job ${SLURM_ARRAY_JOB_ID}_${ID} on $(hostname): frames $S-$E, $MAX_PARALLEL x $NTHREAD threads"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /cosma/home/planet/vreke/blender
./render_chunks.sh $S $E
status=$?

sacct -j ${SLURM_JOB_ID} --format=JobID,JobName,Partition,NodeList,Elapsed,ExitCode
exit $status
