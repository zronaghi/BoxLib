#!/bin/bash
#Job script for summitdev multigrid_c
#BSUB -P CSC190PORT
#BSUB -J mg400runtime
#BSUB -o mg400runtime.o%J
#BSUB -W 01:00
#BSUB -nnodes 1

module load gcc/5.4.0
module load cuda/8.0.54
jsrun -n 1 -a 1 -g 1 ./main3d.kokkos.ex inputs
