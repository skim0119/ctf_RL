#!/bin/bash

FILE=$1
BatchSize=$2
i=0
while IFS= read -r line
do
  if [ $i = 0 ]
  then
    echo "Creating File for Batch"
    touch batch.pbs
    echo "#!/bin/bash
#PBS -q eng-research
#PBS -N PPO
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=24
#PBS -j oe
#PBS -M nealeav2@illinois.edu
#PBS -o 1FRAME_CONFID1.out
#PBS -e cluster_error.out

module load anaconda/3
module load vim/8.1
module load cuda/10.0
module load git/2.19.0

module list
conda activate /projects/tran-research-group/skim449/conda/ctf

cd \$PBS_O_WORKDIR
which python
which pip
conda env list
pwd" >> batch.pbs
  fi
  echo "$line" >> batch.pbs
  i=$((i+1))
  if [ $i -gt 2 ]
  then
    echo "Sending Batch to the queue"
    # qsub batch.pbs
    echo "Deleting File for Batch"
    # rm batch.pbs
    i=0
  fi
done < "$1"
