module load singularitypro
SINGULARITY_TMPDIR=$SGE_LOCALDIR
singularity build --fakeroot border_base.sif border_base.def
singularity build --fakeroot border.sif border.def
