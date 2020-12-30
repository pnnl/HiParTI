#!/bin/bash

declare -a s3tsrs=("vast-2015-mc1" "nell2" "choa700k" "1998DARPA" "freebase_music" "flickr" "freebase_sampled" "delicious" "nell1")
declare -a s3tsrs_pl=("3D_irregular_small" "3D_irregular_medium" "3D_irregular_large" "3D_regular_small" "3D_regular_medium" "3D_regular_large")
# declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")

declare -a s4tsrs=("chicago-crime-comm-4d" "nips-4d" "uber-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a s4tsrs_pl=("4D_irregular_small" "4D_irregular_medium" "4D_irregular_large" "4D_regular_small" "4D_regular_medium" "4D_regular_large" "4D_i_small" "4D_i_medium" "4D_i_large")
# declare -a s4tsrs_pl=("4D_regular_large" "4D_regular_medium" "4D_regular_small" "4D_i_large" "4D_i_medium" "4D_i_small")

declare -a test_tsr_names=("4D_i_small")