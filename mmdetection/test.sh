#!/usr/bin/env bash

# for VARIABLE in 294
# do
#     tools/dist_test.sh /sda1/yhk/MM/mmdetection/custom_configs/lg_best_epoch_294/lg_final_best.py /sda1/yhk/MM/mmdetection/custom_configs/lg_best_epoch_294/epoch_${VARIABLE}.pth 2 --show-dir result/lg_final_epoch${VARIABLE} --format-only --eval-options "jsonfile_prefix=./work_dirs/lg_best_final/0.45_0.15_0.65_0.1_lg_final_best_epoch${VARIABLE}predict"
# done



for VARIABLE in 294
do
    python tools/test.py /sda1/yhk/MM/mmdetection/custom_configs/lg_best_epoch_294/lg_final_best.py /sda1/yhk/MM/mmdetection/custom_configs/lg_best_epoch_294/epoch_${VARIABLE}.pth --show-dir result/lg_final_best_diou_clusternms_epoch${VARIABLE} --format-only --eval-options "jsonfile_prefix=./work_dirs/lg_best_final/wcnms0.45_0.15_0.65_0.1_lg_final_best_epoch${VARIABLE}predict"
done