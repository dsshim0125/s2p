import os
import os.path as osp
import time

import numpy as np
import scipy.misc
import skvideo.io


def get_image(goal, obs, recon_obs, imsize=84, pad_length=1, pad_color=255):
    if len(goal.shape) == 1:
        goal = goal.reshape(-1, imsize, imsize).transpose()
        obs = obs.reshape(-1, imsize, imsize).transpose()
        recon_obs = recon_obs.reshape(-1, imsize, imsize).transpose()
    img = np.concatenate((goal, obs, recon_obs))
    img = np.uint8(255 * img)
    if pad_length > 0:
        img = add_border(img, pad_length, pad_color)
    return img


def add_border(img, pad_length, pad_color, imsize=84):
    H = 3 * imsize
    W = imsize
    img = img.reshape((3 * imsize, imsize, -1))
    img2 = np.ones((H + 2 * pad_length, W + 2 * pad_length, img.shape[2]),
                   dtype=np.uint8) * pad_color
    img2[pad_length:-pad_length, pad_length:-pad_length, :] = img
    return img2



def dump_video_custom(
        env,
        policy,
        filename,
        rollout_function,
        rows=3,
        columns=6,
        pad_length=0,
        pad_color=255,
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,
        num_channels=3,
        curl_crop = False,
        curl_crop_output_size = 84,
        image_rl = False,
        render_kwargs = None,
        env_name = None,
        # for SLAC
        slac_algo = None,
        slac_policy_input_type = None,
        slac_obs_reset_w_same_obs = False,
        generalization_test = None,
        
):
    frames = []
    H = 3 * imsize
    W = imsize
    N = 1 # rows * columns
    l = []
    
    for i in range(N):
        start = time.time()
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
            curl_crop = curl_crop,
            curl_crop_output_size = curl_crop_output_size,
            render_image_for_video_when_state_rl = False if image_rl else True,
            render_kwargs = render_kwargs,
            env_name = env_name,
            slac_algo = slac_algo, 
            slac_policy_input_type = slac_policy_input_type,
            slac_obs_reset_w_same_obs = slac_obs_reset_w_same_obs,
            generalization_test = generalization_test,

        )
        
        # l = []
        image_path = path['full_observations'] if image_rl else path['image_observations']
        for d in image_path:            
            recon = d
            # d : [C*stack, H, W]
            
            l.append(np.transpose(d, (1,2,0))) #[C,H,W] -> [H,W,C]
                
        if do_timer:
            print(i, time.time() - start)

    outputdata = np.stack(l, axis =0) # [T, H,W,C]
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)

