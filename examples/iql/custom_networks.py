import torch
import torch.nn as nn
import numpy as np
# for 224 x 224 : 380192 -> filter 32, 109, 109 OUT_DIM  \
OUT_DIM_224 = {2: 109, 4 : 105, 6 : 101}

# for 128 x 128 : 119072 -> filter 32, OUT_DIM 61
OUT_DIM_128 = {2: 61, 4 : 57, 6 : 53}

# for 84 x 84 inputs
OUT_DIM_84 = {2: 39, 4: 35, 6: 31}

# for 64 x 64 inputs
# OUT_DIM_64 = {2: 29, 4: 25, 6: 21}



from rlkit.torch.networks.mlp import Mlp 


class Qfunction(Mlp):
    def __init__(self,  *args, dim=1, encoder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.encoder = encoder

    def forward(self, states, actions, detach_encoder =False, **kwargs):
        
        if self.encoder is not None:
            features = self.encoder(states, detach=detach_encoder) #[bs, dim]
        else:
            features = states
        flat_inputs = torch.cat((features, actions), dim=self.dim)
        return super().forward(flat_inputs, **kwargs)

class Vfunction(Mlp):    
    def __init__(self,  *args, dim=1, encoder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.encoder = encoder
        

    def forward(self, states, detach_encoder=False, **kwargs):
        
        if self.encoder is not None:
            features = self.encoder(states, detach=detach_encoder) #[bs, dim]
        else:
            features = states
        flat_inputs = features
        return super().forward(flat_inputs, **kwargs)

class Critic(nn.Module):
    def __init__(self, qf1, qf2, target_qf1, target_qf2, encoder = None, encoder_target=None):
        super().__init__()
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        
        self.encoder = encoder
        self.encoder_target = encoder_target
    
    def forward(self, obs, action, detach_encoder = False):
        if self.encoder is not None:
            obs_encoder = self.encoder(obs, detach=detach_encoder)
        else:
            obs_encoder = obs
        if self.encoder_target is not None:
            obs_encoder_target = self.encoder_target(obs, detach=detach_encoder)
        else:
            obs_encoder_target = obs
        q1 = self.qf1(obs_encoder, action)
        q2 = self.qf2(obs_encoder, action)
        
        
        target_q1 = self.target_qf1(obs_encoder_target, action)
        target_q2 = self.target_qf2(obs_encoder_target, action)
        
        return q1, q2, target_q1, target_q2

        

class VFunctionCritic(nn.Module):
    def __init__(self, vf, encoder=None):
        super().__init__()
        self.vf = vf                
        self.encoder = encoder
            
    def forward(self, obs,detach_encoder = False):
        if self.encoder is not None:
            obs_encoder = self.encoder(obs, detach=detach_encoder)
        else:
            obs_encoder = obs
        
        v = self.vf(obs_encoder)        
        
        return v


class CriticSLAC(nn.Module):
    def __init__(self, qf1, qf2, target_qf1, target_qf2, vf=None):
        super().__init__()
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vf = vf
        
    def forward(self, obs_encoded, action=None, detach_encoder = False):
        # Assume encoded input is given
        vf = self.vf(obs_encoded)
        if action is not None:
            q1 = self.qf1(obs_encoded, action)
            q2 = self.qf2(obs_encoded, action)
            
            target_q1 = self.target_qf1(obs_encoded, action)
            target_q2 = self.target_qf2(obs_encoded, action)
            return q1, q2, target_q1, target_q2, vf

        return vf

        



from rlkit.torch.sac.policies.gaussian_policy import GaussianPolicy, TanhGaussianPolicy
# noinspection PyMethodOverriding
class TanhGaussianPolicyWithEncoder(TanhGaussianPolicy):
    def __init__(self, encoder=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

    def forward(self, inputs, detach_encoder=False, **kwargs):
        if self.encoder is not None:
            features = self.encoder(inputs, detach=detach_encoder)
        else:
            features = inputs
        
        return super().forward(features, **kwargs)


class GaussianPolicyWithEncoder(GaussianPolicy):
    def __init__(self, encoder=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

    def forward(self, inputs, detach_encoder=False):
        if self.encoder is not None:
            features = self.encoder(inputs, detach=detach_encoder)
        else:
            features = inputs
        
        return super().forward(features)

