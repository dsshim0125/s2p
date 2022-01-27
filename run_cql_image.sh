python examples/iql/mujoco_finetune.py \
    --env_name cheetah-run-mixed_first_500k\
    --exp_name test \
    --algo_type 'cql' \
    --image_rl \
    --no_curl_contrastive_learning \
    --gpu_id 0 \
    --slac_representation \
    --slac_policy_input_type 'feature_action' \
    --slac_latent_model_load_dir '/cheetah-run/slac-cheetah-run-mixed_first_500k-seed0-20220102-1241/model' \
    --data_mix_type 'all_state_1step_random_action' \
    --data_mix_num_real 1000 \
    --data_mix_num_gen 1000 \
    --uncertainty_penalty_lambda 2 \
    --uncertainty_type 'aleatoric' \
    --use_tiny_data \
    
