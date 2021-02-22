#!/bin/bash

FILE=$1
BatchSize=$2
Name=$3

Time=$4
if [ -z "$Time" ]
then
  Time=2880
fi

QueueName=$5
if [ -z "$QueueName" ]
then
  QueueName="eng-research"
fi

EnvName=$6
if [ -z "$EnvName" ]
then
  EnvName="tf23"
fi

ExtraArgs=$7
if [ -z "$ExtraArgs" ]
then
  ExtraArgs=""
fi

fileCounter=1
i=1
while IFS= read -r line
do
  if [ $i = 1 ]
  then
    echo "Creating File for Batch"
    echo "#!/bin/bash
#SBATCH -p $QueueName
#SBATCH --job-name=$Name.$fileCounter
#SBATCH -t $Time
#SBATCH --mail-type=BEGIN,END
#SBATCH -o tmp.out
#SBATCH -e cluster_error.out
#SBATCH $ExtraArgs

module load anaconda/3
module load vim/8.1
module load cuda/10.0
module load git/2.19.0
module load gcc

module list
. /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate /home/nealeav2/trg/nealeav2/conda/$EnvName
which python
which pip
conda env list
pwd
" >> batch.slurm
  fi
  if [ $i -lt $BatchSize ]; then
    echo "${line} &" >> batch.pbs
    i=$((i+1))
  else
    echo "${line}" >> batch.slurm
    echo "Sending Batch to the queue"
    sbatch batch.slurm
    echo "Deleting File for Batch"
    rm batch.slurm
    i=1
    fileCounter=$((fileCounter+1))
  fi
done < "$1"
