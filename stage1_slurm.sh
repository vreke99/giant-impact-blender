#!/bin/bash -l

#
#   COSMA SLURM job array for Stage 1 (combined_01_make.py)
#
#   Usage
#   -----
#   sbatch stage1_slurm.sh                 # frames 0-2160
#   sbatch stage1_slurm.sh 0 400           # a sub-range (START END)
#   sbatch --array=0-19 stage1_slurm.sh    # 20 chunks
#   sbatch --cpus-per-task=1 --mem=4G stage1_slurm.sh   # light modes or reused meshes/mist
#
#   Each array task is one Stage 1 process on a chunk of frames, with its own
#   cores and memory on a shared node (no --exclusive).  The frame range is
#   split evenly across however many array tasks there are, so changing
#   --array is all that is needed to use more or fewer chunks.
#

#SBATCH --array=0-19                  # 20 chunks; %N suffix (e.g. 0-19%5) limits how many run at once
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # threads per chunk (splashsurf, neighbour search)
#SBATCH --mem=19G                    # true peaks: ~10G MESH_MIST, ~1G others; ~4.8G/cpu is free
#SBATCH -J Stage1_make
#SBATCH -o /cosma5/data/dp004/vreke/blender/outs_and_errs/stage1_%A_%a.out
#SBATCH -e /cosma5/data/dp004/vreke/blender/outs_and_errs/stage1_%A_%a.err
#SBATCH -p cosma5
#SBATCH -A durham
#SBATCH --mail-type=END
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH -t 3:00:00

module purge
module load python/3.12.4

#   This task's chunk of the frame range
START=${1:-0}
END=${2:-2160}
NCHUNK=${SLURM_ARRAY_TASK_COUNT:-1}
ID=${SLURM_ARRAY_TASK_ID:-0}
STEP=$(( (END - START + NCHUNK) / NCHUNK ))
S=$(( START + ID * STEP ))
E=$(( S + STEP - 1 )); (( E > END )) && E=$END
if (( S > END )); then echo "Task $ID: nothing to do"; exit 0; fi

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

SCRIPT_DIR=/cosma/home/planet/vreke/blender
cd "$SCRIPT_DIR"
echo "Job ${SLURM_ARRAY_JOB_ID}_${ID} on $(hostname): frames $S-$E with $RAYON_NUM_THREADS threads"
python3 --version

# /usr/bin/time -v reports the true peak memory of the Python processes
# ("Maximum resident set size", in the .err file).  sacct's MaxRSS can
# include the node's file cache, which fills to the --mem limit.
/usr/bin/time -v python3 combined_01_make.py $S $E

echo "Task done!"
sacct -j ${SLURM_JOB_ID} --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
