# -*- coding: utf-8 -*-
# @Time        : 15/08/2024 17:10 PM
# @Author      : li zezeng
# @Email       : zezeng.lee@gmail.com


import torch
import numpy as np
import scipy.io as sio
import os
from glob import glob
from ot_utils.optimal_transport import OptimalTransport

torch.set_printoptions(precision=8)
#  generate latent code P
def get_adversarial_sample(args,OT_solver,return_att_target=False):
    topk = args.topk
    I_all = OT_solver.find_topk_neib(topk)
    numX =  I_all.shape[1]
    I_all_2 = -torch.ones([2, (topk-1) * numX], dtype=torch.long, device=args.device)
    for ii in range(topk-1):
        I_all_2[0, ii * numX:(ii+1) * numX] = I_all[0,:]
        I_all_2[1, ii * numX:(ii+1) * numX] = I_all[ii + 1, :]
    I_all = I_all_2
    
    if torch.sum(I_all < 0) > 0:
        print('Error: numX is not a multiple of bat_size_n')

    ###compute angles
    P = OT_solver.tg_fea.view(OT_solver.num_tg, -1)   
    nm = torch.cat([P, -torch.ones([OT_solver.num_tg,1],device=args.device)], dim=1)
    nm /= torch.norm(nm,dim=1).view(-1,1)
    cs = torch.sum(nm[I_all[0,:],:] * nm[I_all[1,:],:], 1) #element-wise multiplication
    cs = torch.min(torch.ones([cs.shape[0]],device=args.device), cs)
    theta = torch.acos(cs)
    print(torch.max(theta))
    #theta = (theta-torch.min(theta))/(torch.max(theta)-torch.min(theta))

    ###filter out sigularity with theta larger than threshold
    #flag0 = (theta >= args.angle_thresh).view(numX,-1)#I_all[:, theta <= args.angle_thresh]
    #flag = (torch.sum(flag0,dim=1)>args.neib_nums)
    print('theta=',theta)
    print('theta.max()=',theta.max())
    print('theta.min()=',theta.min())
    print('theta.mean()=',theta.mean())
    
    flag = (theta >= args.angle_thresh)#I_all[:, theta <= args.angle_thresh]
    
    I_SA = I_all[:, flag]
    I_SA, _ = torch.sort(I_SA, dim=0)
    #_, uni_SA_id = np.unique(I_SA[0,:].cpu().numpy(), return_index=True)
    #np.random.shuffle(uni_SA_id)
    #I_SA = I_SA[:, torch.from_numpy(uni_SA_id)]
     
    numSA = I_SA.shape[1]
    if args.adv_samp_nums is not None:
        numSA = min(numSA, args.adv_samp_nums)
        idc= np.random.randint(0,I_SA.shape[1],numSA)
        I_SA = I_SA[:,idc]
    #I_SA = I_SA[:,:numSA]
    print('OT successfully generated {} adversarial samples'.format(numSA))
    
    
    ###target features transfer   
    P_org = OT_solver.tg_fea[I_SA[0,:],:]
    P_org_Nb = OT_solver.tg_fea[I_SA[1,:],:]
    id_SA = I_SA[0,:].squeeze().cpu().numpy().astype(int)
    
    '''generate adversarial features'''
    # rand_w = torch.rand([numSA,1])    
    rand_w = args.dissim * torch.ones([numSA,1],device=args.device)
    P_SA = torch.mul(P_org, 1 - rand_w) + torch.mul(P_org_Nb, rand_w)
    P_SA_org = np.concatenate((P_SA.cpu().numpy(),P_org.cpu().numpy()))
    
    SA_feature_path = os.path.join(args.save_dir,'ot_attacked_and_original_features.mat')
    sio.savemat(SA_feature_path, {'features':P_SA_org, 'ids':id_SA})
    if return_att_target:
        id_SAT = I_SA[1,:].squeeze().cpu().numpy().astype(int)
        return P_SA,P_org,id_SA,P_org_Nb,id_SAT
    else:
        return P_SA,P_org,id_SA
    
    

def get_OT_solver(args,tg_fea):
    #arguments for training OT
    
    num_fea = tg_fea.shape[0]
    print('tg_fea.shape:',tg_fea.shape)
    
    bat_size_tg = args.bat_size_tg
    tg_fea = tg_fea[0:num_fea//bat_size_tg*bat_size_tg,:]
    num_fea = tg_fea.shape[0]
    #tg_measures = (torch.ones(num_fea)/num_fea).to(args.device)
    tg_measures = (torch.zeros(num_fea)/num_fea).to(args.device)
    
    
    ot_dir=os.path.join(args.save_dir, 'ot')
    if not os.path.exists(ot_dir):       
        os.makedirs(ot_dir)

    '''train ot'''
    OT_solve = OptimalTransport(tg_fea, args, args.device, ot_dir)
    if args.h_name is None:
        #OT_solve.set_h(torch.load('./results/modelnet40/points2048/ot/h_best.pt')) #h_best
        #OT_solve.set_h(torch.load('./results/shapenetpart8/AE_airplane_1727602099/ot/h_best.pt')) #h_best
        #
        #OT_solve.train_ot(tg_measures,8000)#8000
        OT_solve.train_ot(tg_measures)#8000
    else:
        OT_solve.set_h(torch.load(args.h_name)) 
    print('OT have been successfully solved')
    return OT_solve


def get_adversarial_sample_with_target(args,OT_solver,input_code):
    topk = args.topk
    I_all = OT_solver.find_topk_neib(topk)
    numX =  I_all.shape[1]
    I_all_2 = -torch.ones([2, (topk-1) * numX], dtype=torch.long, device=args.device)
    for ii in range(topk-1):
        I_all_2[0, ii * numX:(ii+1) * numX] = I_all[0,:]
        I_all_2[1, ii * numX:(ii+1) * numX] = I_all[ii + 1, :]
    I_all = I_all_2
    
    if torch.sum(I_all < 0) > 0:
        print('Error: numX is not a multiple of bat_size_n')

    ###compute angles
    P = OT_solver.tg_fea.view(OT_solver.num_tg, -1)   
    nm = torch.cat([P, -torch.ones([OT_solver.num_tg,1],device=args.device)], dim=1)
    nm /= torch.norm(nm,dim=1).view(-1,1)
    cs = torch.sum(nm[I_all[0,:],:] * nm[I_all[1,:],:], 1) #element-wise multiplication
    cs = torch.min(torch.ones([cs.shape[0]],device=args.device), cs)
    theta = torch.acos(cs)
    print(torch.max(theta))
    #theta = (theta-torch.min(theta))/(torch.max(theta)-torch.min(theta))
    
    numSA = input_code.shape[0]
    dist = torch.zeros([numSA,OT_solver.num_tg],device=args.device)
    for i in range(numSA):
        for j in range(OT_solver.num_tg):
            dist[i,j]=4-torch.sum(input_code[i]*P[j])
    val, idxes = torch.topk(dist, topk, dim=1)
    
    rand_w = args.dissim * torch.ones([numSA,1],device=args.device)
    adv_code = None
    for i in range(1,topk):
        idx = idxes[:,i]
        target_code = P[idx]
        P_SA = torch.mul(input_code, 1 - rand_w) + torch.mul(target_code, rand_w)
        if adv_code is None:
            adv_code = P_SA
        else:
            adv_code = torch.cat((adv_code,P_SA),dim=0)
            
    return adv_code