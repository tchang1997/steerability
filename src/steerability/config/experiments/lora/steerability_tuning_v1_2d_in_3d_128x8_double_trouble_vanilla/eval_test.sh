#!/bin/bash

set -x
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_vanilla/checkpoint-192_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_vanilla/checkpoint-288_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_vanilla/checkpoint-384_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_vanilla/checkpoint-480_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_vanilla/checkpoint-576_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_vanilla/checkpoint-672_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_vanilla/checkpoint-768_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_vanilla/checkpoint-96_test.yml --api-config /home/ctrenton/api/vllm_oai.config
