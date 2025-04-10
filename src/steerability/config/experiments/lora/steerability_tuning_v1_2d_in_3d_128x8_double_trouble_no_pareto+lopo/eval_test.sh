#!/bin/bash

set -x
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_pareto+lopo/checkpoint-192_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_pareto+lopo/checkpoint-288_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_pareto+lopo/checkpoint-384_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_pareto+lopo/checkpoint-480_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_pareto+lopo/checkpoint-576_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_pareto+lopo/checkpoint-672_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_pareto+lopo/checkpoint-768_test.yml --api-config /home/ctrenton/api/vllm_oai.config
python create_steerability_report.py --config ./config/experiments/lora/steerability_tuning_v1_2d_in_3d_128x8_double_trouble_no_pareto+lopo/checkpoint-96_test.yml --api-config /home/ctrenton/api/vllm_oai.config
