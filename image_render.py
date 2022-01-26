
import PIL
import dmc2gym
import copy
import numpy as np

dm_control_env_dict = { 'cheetah-run-mixed' : dict(domain_name='cheetah', task_name='run', frame_skip=4),                        
                        }


def test_augment_state2image_random_sample_wrapper():    
    env_dict = dm_control_env_dict
    is_dmc = True if env_dict==dm_control_env_dict else False
    if is_dmc:
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb()
        vdisplay.start()
    
    for env_name, env_kwargs in env_dict.items():
        test_augment_state2image_random_sample(env_name, env_kwargs, is_dmc=is_dmc)
    
    if is_dmc:
        vdisplay.stop()


def test_augment_state2image_random_sample(env_name=None, env_kwargs=None, is_dmc = False):
    
    import gym    
    import numpy as np
    import os    
    import time    
    
    # Create the environment
    
    save_image_type = 'stack_3_images'
    # debug
    # n_image_generation = int(2e3)
    n_image_generation = int(2e4) 
    
    if is_dmc:
        import dmc2gym
        import torch        
        env = dmc2gym.make(visualize_reward = False, **env_kwargs)
        
        
        payload = torch.load('./data/cheetah-run-mixed_first_500k_data.pth')
        dataset = {}
        custom_range_mixed_data = True
        if custom_range_mixed_data:
            original_size = payload[0].shape[0]
            custom_range = int(5e5) # first 500k
            # custom_range = int(1e5) # first 100k
            print('@@@@@@@@@@ Currently use custom range : ', custom_range)
            dataset['observations'] = payload[0][:custom_range]
            dataset['next_observations'] = payload[1][:custom_range]
            dataset['actions'] = payload[2][:custom_range]
            dataset['rewards'] = payload[3].squeeze()[:custom_range]
            dataset['terminals'] = (payload[4]==0).astype(payload[4].dtype).squeeze()[:custom_range] # 0,1  뒤바꾸기 as dm control saves not_dones
            dataset['infos'] = payload[5][:custom_range]
            dataset['timeouts'] = payload[6].squeeze()[:custom_range]

        else:
            dataset['observations'] = payload[0]
            dataset['next_observations'] = payload[1]
            dataset['actions'] = payload[2]
            dataset['rewards'] = payload[3].squeeze()
            dataset['terminals'] = (payload[4]==0).astype(payload[4].dtype).squeeze() # 0,1  뒤바꾸기 as dm control saves not_dones
            dataset['infos'] = payload[5]
            dataset['timeouts'] = payload[6].squeeze()
        
    
    custom_dataset = {}
    
    
    if env_name in dm_control_env_dict.keys():         
        original_data_size = dataset['observations'].shape[0]
        print('env_name : {} , data size : {}'.format(env_name, original_data_size))
        
        print('n image generation : ', n_image_generation)
        timeout_indices = np.where(dataset['timeouts']==1)[0]
        terminal_indices = np.where(dataset['terminals']==1)[0]
        timeout_indices = np.sort(np.concatenate([timeout_indices, terminal_indices]))
        n_trajs = timeout_indices.shape[0]
        print('n_trajs : ', n_trajs)
        
        num_stack = int(save_image_type.replace('stack_','')[:1])
        
        sample_type='except_for_stack_'+str(num_stack)

        save_trajwise = True

        if save_trajwise:
            print('save trajwise!')
            if sample_type=='except_for_stack_3':
                # episode end indice중에 random하게 골라야
                # Assume bigger than 100k dataset
                trajectories = []                
                start_idx = 0
                num_steps = 0
                temp_episode_end_indices = timeout_indices.copy()
                np.random.shuffle(temp_episode_end_indices) # random sample traj
                t0_indices = []
                pseudo_timeouts = []
                # t1_indices = []
                for end_idx in temp_episode_end_indices: # if 마지막에 남는거 있다면 무시
                    idx = np.where(timeout_indices==end_idx)[0]
                    if idx ==0 :
                        start_idx = 0
                    else:
                        start_idx = timeout_indices[idx-1]+1
                    import copy
                    t0_indices.append(copy.deepcopy(num_steps))
                    current_traj_length = (end_idx-start_idx) # +1
                    num_steps += current_traj_length
                    trajectories.append(np.arange(start_idx, end_idx)) # e.g. end_idx = 249, [0, .. 248]. except end idx for next obs -> terminal이든 timeout이든 안들어가게됨 (뒤에 next_qpos_qvel render 때문에)
                    # temp_terminal = np.zeros(current_traj_length)
                    # temp_terminal[-1] = 1.0
                    temp_timeouts = np.zeros_like(np.arange(start_idx, end_idx)) 
                    temp_timeouts[-1] = 1.0 #[0,0,... 0, 1] : size : [249]
                    pseudo_timeouts.append(temp_timeouts)
                    # t1_indices.append(start_idx+1)
                    if num_steps > int(n_image_generation):
                        break
                t0_indices = np.array(t0_indices, dtype=np.int32)
                # pseudo_terminals = np.concatenate(pseudo_terminals, axis =-1)
                pseudo_timeouts = np.concatenate(pseudo_timeouts, axis =-1)
                random_indices = np.concatenate(trajectories, axis =-1) # [0,1,2,... 98, || 100,101,...]
                random_indices_for_tm1 = random_indices-1 # [-1,0,1,... 98, || 99,100,...]
                random_indices_for_tm2 = random_indices-2 # [-2,-1,0,... 97, || 98,99,...]
                random_indices_for_tp1 = random_indices+1

                random_indices_for_tm1[t0_indices] = random_indices_for_tm1[t0_indices+1]
                random_indices_for_tm2[t0_indices] = random_indices_for_tm2[t0_indices+2]
                random_indices_for_tm2[t0_indices+1] = random_indices_for_tm2[t0_indices+2]

                assert num_steps==random_indices.shape[0]
                print('sampled dataset size (n image generation) : ', num_steps)
                n_image_generation = int(num_steps)

                custom_dataset['terminals'] = dataset['terminals'][random_indices]
                # custom_dataset['timeouts'] = dataset['timeouts'][random_indices]
                custom_dataset['timeouts'] = pseudo_timeouts
                assert custom_dataset['terminals'].shape==pseudo_timeouts.shape

                custom_dataset['observations'] = dataset['observations'][random_indices]
                custom_dataset['next_observations'] = dataset['next_observations'][random_indices]
                custom_dataset['actions'] = dataset['actions'][random_indices]
                custom_dataset['rewards'] = dataset['rewards'][random_indices]                    
                custom_dataset['indices'] = random_indices
                
                image_indices = random_indices
                next_qpos_qvel = []
                qpos_qvel = []
                qpos_qvel_tm1 = []
                qpos_qvel_tm2 = []

                for k in random_indices_for_tp1:
                    next_qpos_qvel.append(dataset['infos'][k,0]['internal_state'])
                for k in random_indices:
                    qpos_qvel.append(dataset['infos'][k,0]['internal_state'])
                for k in random_indices_for_tm1:
                    qpos_qvel_tm1.append(dataset['infos'][k,0]['internal_state'])
                for k in random_indices_for_tm2:
                    qpos_qvel_tm2.append(dataset['infos'][k,0]['internal_state'])
                

                next_qpos_qvel = np.stack(next_qpos_qvel, axis =0)
                qpos_qvel = np.stack(qpos_qvel, axis =0)    
                qpos_qvel_tm1 = np.stack(qpos_qvel_tm1, axis =0)
                qpos_qvel_tm2 = np.stack(qpos_qvel_tm2, axis =0)

                custom_dataset['image_indices'] = image_indices
                custom_dataset['qpos_qvel']=qpos_qvel
                if 'reacher' in env_name: # due to goal visualization
                    temp_dataset = {}
                    image_indices_tm1 = random_indices_for_tm1
                    image_indices_tm2 = random_indices_for_tm2
                    temp_dataset['observations_tm1'] = dataset['observations'][image_indices_tm1]
                    temp_dataset['observations_tm2'] = dataset['observations'][image_indices_tm2]

                
    
    image_size = 100
    
    import matplotlib.pyplot as plt
    from PIL import Image
    
    if save_image_type=='stack_3_images':
        save_figure_dir_prefix = './data/trajwise/'     
    
    if custom_range_mixed_data:
        save_figure_dir = save_figure_dir_prefix+env_name+'_first_500k'
        # save_figure_dir = save_figure_dir_prefix+env_name+'_first_100k'
    else:
        save_figure_dir = save_figure_dir_prefix+env_name
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    env.reset()
    adroit_env = False
    
    if 'image' in save_image_type:        
        images, images_tp1, images_tm1, images_tm2 = [], [], [] , []
        
        for i in range(n_image_generation):
            print('{}th image ...'.format(i)) if i %500 == 0 else None
            save_figure = True if i < 300 else False
            t = image_indices[i]
            if is_dmc:
                qp_qv = qpos_qvel[i]
                qp = qv = None
                special_obs = custom_dataset['observations'][i][2:4] if 'reacher' in env_name else None
                img = render_image(env_name, env, qp, qv, image_size, is_dmc=True, qpos_qvel = qp_qv, special_obs = special_obs)

            
            images.append(img)
            if save_figure:
                
                if not os.path.exists(save_figure_dir):
                    os.makedirs(save_figure_dir)
                
                img = Image.fromarray(img.astype(np.uint8))
                img.save(save_figure_dir+'/'+str(t)+'.jpg')
                del img

            
            if save_image_type=='with_next_image' or save_image_type=='stack_3_images' :
                if is_dmc:
                    qp_qv = next_qpos_qvel[i]
                    qp = qv = None
                    special_obs = custom_dataset['next_observations'][i][2:4] if 'reacher' in env_name else None
                    img = render_image(env_name, env, qp, qv, image_size, is_dmc=True, qpos_qvel = qp_qv, special_obs = special_obs)
                
                
                
                images_tp1.append(img)
                if save_figure and (not adroit_env):                    
                    if not os.path.exists(save_figure_dir):
                        os.makedirs(save_figure_dir)                    
                
                    img = Image.fromarray(img.astype(np.uint8))
                    img.save(save_figure_dir+'/'+str(t+1)+'.jpg')
                    del img
            
            if save_image_type=='stack_3_images':
                if is_dmc:
                    qp_qv = qpos_qvel_tm1[i]
                    qp = qv = None
                    special_obs = temp_dataset['observations_tm1'][i][2:4] if 'reacher' in env_name else None
                    img = render_image(env_name, env, qp, qv, image_size, is_dmc=True, qpos_qvel = qp_qv, special_obs = special_obs)
                
                
                images_tm1.append(img)
                if save_figure and (not adroit_env):                                        
                    if not os.path.exists(save_figure_dir):
                        os.makedirs(save_figure_dir)                    
                    
                    img = Image.fromarray(img.astype(np.uint8))
                    img.save(save_figure_dir+'/'+str(t-1)+'.jpg')
                    del img
                
                if is_dmc:
                    qp_qv = qpos_qvel_tm2[i]
                    qp = qv = None
                    special_obs = temp_dataset['observations_tm2'][i][2:4] if 'reacher' in env_name else None
                    img = render_image(env_name, env, qp, qv, image_size, is_dmc=True, qpos_qvel = qp_qv, special_obs = special_obs)
                                
                images_tm2.append(img)
                if save_figure and (not adroit_env):                    
                    if not os.path.exists(save_figure_dir):
                        os.makedirs(save_figure_dir)                    
                    
                    img = Image.fromarray(img.astype(np.uint8))
                    img.save(save_figure_dir+'/'+str(t-2)+'.jpg')
                    del img
                
        images = np.stack(images, axis=0) # [bs,h,w,c]
        custom_dataset['image_observations'] = images        
        if save_image_type=='with_next_image' or save_image_type=='stack_3_images':
            images_tp1 = np.stack(images_tp1, axis=0) # [bs,h,w,c]
            custom_dataset['image_observations_tp1'] = images_tp1
        if save_image_type=='stack_3_images':
            images_tm1 = np.stack(images_tm1, axis=0) # [bs,h,w,c]
            images_tm2 = np.stack(images_tm2, axis=0) # [bs,h,w,c]
            custom_dataset['image_observations_tm1'] = images_tm1
            custom_dataset['image_observations_tm2'] = images_tm2
            

        custom_dataset['image_indices'] = image_indices # np.stack(image_indices, axis =0) #[bs]
        print('saving subsampled data...')
        
    
    
    start = time.time()
    import h5py
    if save_image_type=='stack_3_images':
        h5f_w = h5py.File(save_figure_dir+'/image_numpy_dataset_stack3_imgsize_'+str(image_size)+'.hdf5', 'w')    
    elif save_image_type=='stack_3_states':
        h5f_w = h5py.File(save_figure_dir+'/state_numpy_dataset_stack3.hdf5', 'w')
    else:
        h5f_w = h5py.File(save_figure_dir+'/image_numpy_dataset_imgsize_'+str(image_size)+'.hdf5', 'w')
    for k,v in custom_dataset.items():
        h5f_w.create_dataset(str(k), data=v)
    if 'image' in save_image_type:
        print('{} images save time : {}'.format(custom_dataset['image_observations'].shape[0], time.time() - start))
    else:
        print('{} states save time : {}'.format(custom_dataset['observations'].shape[0], time.time() - start))
    # time.sleep(5)
    h5f_w.close()
    del h5f_w


def render_image(env_name, env, qp, qv, image_size=100, is_dmc = False, qpos_qvel = None, special_obs = None):
    
    if is_dmc:
        env.physics.set_state(qpos_qvel)
        env.physics.forward()        
        # or env.physics.step()
        if 'reacher' in env_name: # to visualize target goal                        
            finger_to_target_xy = special_obs
            finger_pos = env.physics.named.data.geom_xpos['finger']                        
            target_pos = finger_pos +np.concatenate([finger_to_target_xy, np.array([0.0])], axis =-1)            
            env.physics.named.data.geom_xpos['target'] = target_pos
            # print('target pos : ', target_pos)
    
    if is_dmc:
        img = env.render(mode='rgb_array', width =image_size, height =image_size) # tracking camera, [h,w,c]

    return img




if __name__=='__main__':
    
    test_augment_state2image_random_sample_wrapper()
    
    
    
    
    
    