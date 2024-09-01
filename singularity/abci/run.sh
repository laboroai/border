module load cuda/11.8
module load cudnn/8.8.1
SINGULARITY_TMPDIR=$SGE_LOCALDIR singularity run --nv --fakeroot border.sif "$1"
