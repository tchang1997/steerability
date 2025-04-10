#!/bin/bash

set -x
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_lopo/checkpoint-192_train.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_lopo/checkpoint-288_train.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_lopo/checkpoint-384_train.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_lopo/checkpoint-480_train.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_lopo/checkpoint-576_train.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_lopo/checkpoint-672_train.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_lopo/checkpoint-768_train.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_lopo/checkpoint-96_train.yml --api-config /home/ctrenton/api/vllm_oai.config
