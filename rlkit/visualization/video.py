

from rlkit.core import trainer
from rlkit.util.video import dump_video_custom
from rlkit.samplers.rollout_functions import rollout
def VideoSaveFunction(savedir, deterministic_policy=False, env = None, env_name = None, image_rl = False, \
    max_path_length = None, curl_crop = False, curl_crop_output_size = 84, render_kwargs = None, \
    slac_algo = None, slac_policy_input_type = None, slac_obs_reset_w_same_obs = False, generalization_test = None):
    
    def video_func(RL_class, epoch):
        if deterministic_policy:
            from rlkit.torch.sac.policies import MakeDeterministic
            policy = MakeDeterministic(RL_class.trainer.policy)
        else:
            policy = RL_class.trainer.policy

        if epoch % 5 ==0:
            dump_video_custom(env=env,# RL_class.eval_env,
                            policy = policy,#RL_class.trainer.policy,
                            filename=savedir+'/evaluate_rollout_'+str(epoch)+'.mp4',
                            rollout_function= rollout, # RL_class.trainer._rollout_fn,                    
                            horizon = max_path_length,
                            curl_crop = curl_crop,
                            curl_crop_output_size = curl_crop_output_size,
                            image_rl = image_rl,
                            render_kwargs = render_kwargs,
                            env_name = env_name,
                            # for SLAC
                            slac_algo = slac_algo, 
                            slac_policy_input_type = slac_policy_input_type,
                            slac_obs_reset_w_same_obs = slac_obs_reset_w_same_obs,
                            generalization_test = generalization_test,
                            )
        else:
            pass  # do nothing
        



    return video_func #(self, epoch)
    
    