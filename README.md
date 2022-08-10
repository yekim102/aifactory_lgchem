# AiFactory.space LG-chem Particle Instance Segmentation Private Challenge

LG-chem private challenge: https://aifactory.space/competition/detail/2067.
Instance segmentation using mmdetection

## Run commands
In the mmdetection directory:
- train: ./tools/dist_train.sh {config_file} {# gpus to use}
- test: ./tools/dist_test.sh {config_file} {model_path} {gpu #} --show-dir {image_output_dir} --format-only --eval-options='jsonfile_prefix={output_dir}'
