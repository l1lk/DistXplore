import torch
import numpy as np
from attack.mmd import MMDLoss
import attack.mmd_v2 as mmdv2
'''
ne_feats: N * sample feats
ae_feats: N * sample feats
'''
# --- Create mmd kernel ---
mmd_loss = MMDLoss()


# --- Calculate MMD ---
def cal_mmd_linear(ae_feats, ne_feats):
    ae_feats = torch.Tensor(np.array(ae_feats))
    ne_feats = torch.Tensor(np.array(ne_feats))
    return mmdv2.mmd_linear(torch.flatten(ae_feats, start_dim=1), torch.flatten(ne_feats, start_dim=1))

def cal_mmd_rbf(ae_feats, ne_feats):
    ae_feats = torch.Tensor(np.array(ae_feats))
    ne_feats = torch.Tensor(np.array(ne_feats))
    return mmdv2.mmd_rbf(torch.flatten(ae_feats, start_dim=1), torch.flatten(ne_feats, start_dim=1))

def cal_mmd_poly(ae_feats, ne_feats):
    ae_feats = torch.Tensor(np.array(ae_feats))
    ne_feats = torch.Tensor(np.array(ne_feats))
    return mmdv2.mmd_poly(torch.flatten(ae_feats, start_dim=1), torch.flatten(ne_feats, start_dim=1))


def cal_mmd(ae_feats, ne_feats):
    ae_feats = torch.Tensor(np.array(ae_feats))
    ne_feats = torch.Tensor(np.array(ne_feats))
    return mmd_loss(torch.flatten(ae_feats, start_dim=1), torch.flatten(ne_feats, start_dim=1))



# --debug--  
# diff = np.linalg.norm(ne_feats[0] - ne_feats[1]) 
# print('--------diff is {} --------'.format(diff)) 
# print('--->> losses is {} <<---'.format(losses)) 
   
   
 


  
    
