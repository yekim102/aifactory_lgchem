#!/usr/bin/env bash

for VARIABLE in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    tools/dist_test.sh /home/synergy/yhk/MM/mmdetection/custom_configs/h2res2net101_bigger_size/htcres2net101.py /home/synergy/yhk/MM/mmdetection/custom_configs/h2res2net101_bigger_size/epoch_${VARIABLE}.pth 2 --show-dir result/htcres2net_epoch${VARIABLE} --format-only --eval-options "jsonfile_prefix=./work_dirs/htcres2net_larger_epoch${VARIABLE}/0.8_0.15_htcres2net_larger_epoch${VARIABLE}predict"
done

