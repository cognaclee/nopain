import os
import time
import argparse
import torch
import sys
from tqdm.auto import tqdm
from utils.dataset import *
from utils.ShapeNetDataLoader import *
from utils.misc import *
from utils.data import *
#from utils.ShapeNetDataLoader import *
from models.autoencoder_shapenetpart import *
from evaluation import EMD_CD
#from models.pointnetpp_cls.classification import get_classifier
from ot_utils.ot_util import get_OT_solver, get_adversarial_sample
#from models.classification.pointnet_cls import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, PCT
from models.classifier.ShapeNetPart.pointnet_cls import PointNetCls
from models.classifier.ShapeNetPart.pointnet2_cls_msg import PointNet2ClsMsg
from models.classifier.ShapeNetPart.dgcnn import DGCNN
from models.classifier import PointConvDensityClsSsg,PCT
from defense import *
from models.config import BEST_WEIGHTS

from models.pointFlow.networks import PointFlow




NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
LAYERS = ["ignore", "concat", "concat_v2", "squash", "concatsquash", "scale", "concatscale"]
'''
def pc_normalize(pc):
    #tmp_pc=pc.cpu().numpy()
    centroid = torch.mean(pc, dim=1,keepdim=True)
    tmp_pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(tmp_pc ** 2, dim=2)),dim=1)[0]
    tmp_pc = tmp_pc / m.unsqueeze(dim=1).unsqueeze(dim=2)
    print('tmp_pc.max()=',tmp_pc.max())
    print('tmp_pc.min()=',tmp_pc.min())
    print('tmp_pc.mean()=',tmp_pc.mean())
    return tmp_pc
'''    
def pc_normalize(pc):
    tmp_pc=pc.cpu().numpy()
    centroid = np.mean(tmp_pc, axis=0)
    tmp_pc = tmp_pc - centroid
    m = np.max(np.sqrt(np.sum(tmp_pc ** 2, axis=1)))
    tmp_pc = tmp_pc / m
    return torch.from_numpy(tmp_pc)

# Arguments
def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="/user38/code/diffusion-attack/pretrained/pointFlow/ae/all/checkpoint.pt")
    parser.add_argument('--classifier', type=str, default='pointconv', metavar='MODEL',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pointconv','PCT'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv].')
    parser.add_argument('--trans_model', type=str, default='pointnet', 
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv].')
    parser.add_argument('--num_class', type=int, default=16, help='class numbers')
    #parser.add_argument('--categories', type=str_list, default=['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar','Knife', 
    #                      'Lamp', 'Laptop', 'Motorbick', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table'])
    parser.add_argument('--categories', type=str_list, default=['airplane'])
    
    
    ####  Arguments for PointFlow
    parser.add_argument('--use_deterministic_encoder', type=bool, default=True,help='Whether to use a deterministic encoder.')
    parser.add_argument('--input_dim', type=int, default=3,
                        help='Number of input dimensions (3 for 3D point clouds)')
    parser.add_argument('--use_latent_flow', action='store_true',
                        help='Whether to use the latent flow to model the prior.')
    parser.add_argument('--dims', type=str, default='512-512-512')
    parser.add_argument('--latent_dims', type=str, default='256')
    parser.add_argument("--num_blocks", type=int, default=1,
                        help='Number of stacked CNFs.')
    parser.add_argument("--latent_num_blocks", type=int, default=1,
                        help='Number of stacked CNFs.')
    parser.add_argument("--layer_type", type=str, default="concatsquash", choices=LAYERS)
    parser.add_argument('--time_length', type=float, default=0.5)
    parser.add_argument('--train_T', type=eval, default=True, choices=[True, False])
    parser.add_argument("--nonlinearity", type=str, default="tanh", choices=NONLINEARITIES)
    parser.add_argument('--use_adjoint', type=eval, default=True, choices=[True, False])
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sync_bn', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)
    parser.add_argument('--zdim', type=int, default=128,help='Dimension of the shape code')
    
    
    parser.add_argument('--save_dir', type=str, default="/user38/code/diffusion-attack/results/shapenetpart-flow/")
    parser.add_argument('--device', type=str, default='cuda')
    # Datasets and loaders
    
    parser.add_argument('--dataset_path', type=str, default="/user38/code/diffusion-attack/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/")
    #parser.add_argument('--scale_mode', type=str, default='shape_unit')
    parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batch_size', type=int, default=4)
    
    parser.add_argument('--defense', type=str, default=None,choices=['sor','srs','dupnet'])
    
    # OT 
    parser.add_argument('--h_name', type=str, default="/user38/code/diffusion-attack/results/shapenetpart-flow/pointflow2/ot/h_best.pt")#None
    #parser.add_argument('--h_name', type=str,default="/user38/code/diffusion-attack/results/shapenetpart6/AE_airplane_bag_cap_car_chair_earphone_guitar_knife_lamp_laptop_motorbick_mug_pistol_rocket_skateboard_table_1726194643/ot/h_best.pt", help='file name of OT Brenier h')
    parser.add_argument('--source_dir', type=str, default="/user38/code/diffusion-attack/results/shapenetpart-flow/pointflow2/")
    parser.add_argument('--max_iter', type=int, default=10000,help='max iters of train ot')#
    parser.add_argument('--lr_ot', type=float, default=1e-2,help='learning rate of OT')#
    parser.add_argument('--bat_size_sr', type=int, default=7200,help='Size of mini-batch of Monte-Carlo source samples on device')
    parser.add_argument('--bat_size_tg', type=int, default=720,help='Size of mini-batch of Monte-Carlo target samples on device')
    parser.add_argument('--topk', type=int, default=11, help='The nearest k samples around current sample')
    parser.add_argument('--angle_thresh', type=float, default=0.9,help='the threshold of the angle between two samples')
    parser.add_argument('--dissim', type=float, default=1.2,help='the degree of dissimilarity with the original sample')
    parser.add_argument('--attn', type=int, default=120,help='num of points to attack')
    parser.add_argument('--adv_samp_nums', type=int, default=4096,help='the required number of adversarial samples')
    args = parser.parse_args()
    return args


def main(args):

    ## ShapenetPart
    #all_categories_order=['airplane','bench','car','chair','sofa','table','telephone']
    
    all_categories_order=['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar','Knife', 
                          'Lamp', 'Laptop', 'Motorbick', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']

    
    
    #args = parse_argument()
    actions = ['ot_based_attack']#'extract_feature', 
    
    
    
    
    
    if 'extract_feature' in actions:
        save_dir = os.path.join(args.save_dir, 'AE_%s_%d' % ('_'.join(args.categories), int(time.time())) )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = args.source_dir
    # Logging
    logger = get_logger('test', save_dir)
    '''
    for k, v in vars(args).items():
        logger.info('[ARGS::%s] %s' % (k, repr(v)))
    '''
    # Checkpoint
    ckpt = torch.load(args.ckpt)
    #print('ckpt=',ckpt)
    #seed_all(ckpt['args'].seed)
    #
    # Model
    #logger.info('Loading model...')
    #model = AutoEncoder(ckpt['args']).to(args.device)
    #model.load_state_dict(ckpt['state_dict'])
    
    model = PointFlow(args).to(args.device)

    model.load_state_dict({k.replace('module.',''):v for k,v in ckpt.items()})
    # Datasets and loaders
    
    transform = None
    if args.rotate:
      transform = RandomRotate(180, ['pointcloud'], axis=1)
    logger.info('Transform: %s' % repr(transform))

    test_dset=ShapeNetCore(
    root=args.dataset_path,
    npoints=2048,
    split='test',
    transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    #print('Dataset length = ',len(test_dset))
    
    


     # build classifier
        
        #classifier = get_classifier().to(args.device)
    WEIGHTS = BEST_WEIGHTS["ShapeNetPart"][1024]   
    if  args.classifier.lower() == 'dgcnn':
        print('loading dgcnn')
        classifier = DGCNN(output_channels=16)
        classifier.load_state_dict(torch.load(WEIGHTS[args.classifier])['model_state_dict'])
        print(torch.load(WEIGHTS[args.classifier])['model_state_dict'])
        classifier = classifier.cuda()
        classifier.eval()
    elif args.classifier.lower() == 'pointnet':
        print('loading pointnet')
        classifier = PointNetCls(k=16).cuda()
        classifier.load_state_dict(torch.load(WEIGHTS[args.classifier])['model_state_dict'])
        classifier.eval()
    elif args.classifier.lower() == 'pointnet2':
        print('loading pointnet2')
        classifier = PointNet2ClsMsg(k=16).cuda()
        classifier.load_state_dict(torch.load(WEIGHTS[args.classifier])['model_state_dict'])
        classifier.eval()
    elif args.classifier.lower() == 'pointconv':
        print('loading pointconv')
        classifier = PointConvDensityClsSsg(num_classes=16)
        classifier = torch.nn.DataParallel(classifier).cuda()
        classifier.load_state_dict(torch.load(WEIGHTS[args.classifier]))
        classifier.eval()
    elif args.classifier.lower() == 'pct':
        print('loading PCT')
        classifier = PCT(output_channels=16)
        classifier = torch.nn.DataParallel(classifier).cuda()
        classifier.load_state_dict(torch.load(WEIGHTS[args.classifier]))
        classifier.eval()
    else:
            print('Model not recognized')
            exit(-1)
    
    
    
    
    
    if 'extract_feature' in actions:
            all_ref = []
            all_recons = []
            all_codes = []
            all_labels = []
            all_shifts = []
            all_scales = []
            for i, batch in enumerate(tqdm(test_loader)):
                ref = batch['pointcloud'].to(args.device)
                shift = batch['shift'].to(args.device)
                scale = batch['scale'].to(args.device)
                label = batch['label'].to(args.device)
                

                std=np.load("/user38/code/diffusion-attack/pretrained/pointFlow/ae/all/val_set_std.npy")
                mean=np.load("/user38/code/diffusion-attack/pretrained/pointFlow/ae/all/val_set_mean.npy")
                std=torch.from_numpy(std).cuda()
                mean=torch.from_numpy(mean).cuda()
                ref=(ref-mean)/std
                
                
                
                
                model.eval()
                with torch.no_grad():
                    code = model.encode(ref)
                    #recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()
                    recons = model.decode(code, ref.size(1)).detach()
    
                ref = ref * scale + shift
                recons = recons * scale + shift
    
                all_ref.append(ref.detach().cpu())
                all_recons.append(recons.detach().cpu())
                all_codes.append(code.detach().cpu())
                
                all_labels.append(label.detach().cpu())
                all_shifts.append(shift.detach().cpu())
                all_scales.append(scale.detach().cpu())
            
    
            all_ref = torch.cat(all_ref, dim=0)
            all_recons = torch.cat(all_recons, dim=0)
            all_codes = torch.cat(all_codes, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_shifts = torch.cat(all_shifts, dim=0)
            all_scales = torch.cat(all_scales, dim=0)
            
        
            logger.info('Saving point clouds...')
            torch.save(all_ref, os.path.join(save_dir, 'ref_pc.pt'))
            torch.save(all_recons, os.path.join(save_dir,'dec_pc.pt'))
            torch.save(all_codes, os.path.join(save_dir,'features.pt'))
            torch.save(all_labels, os.path.join(save_dir,'labels.pt'))
            torch.save(all_shifts, os.path.join(save_dir, 'shifts.pt'))
            torch.save(all_scales, os.path.join(save_dir,'scales.pt'))
            
            args.save_dir = save_dir
        
    
    if 'ot_based_attack' in actions:   
        
        if 'extract_feature' in actions:
            #args.source_dir = save_dir
            OT_solver = get_OT_solver(args,all_codes)
            labels = all_labels
            shifts = all_shifts
            scales = all_scales
            ref_pc = all_ref
            
            
        else:
            args.save_dir = args.source_dir
            tg_fea = torch.load(os.path.join(args.source_dir,'features.pt')).to(args.device)
            labels = torch.load(os.path.join(args.source_dir,'labels.pt')).to(args.device)
            shifts = torch.load(os.path.join(args.source_dir,'shifts.pt')).to(args.device)
            scales = torch.load(os.path.join(args.source_dir,'scales.pt')).to(args.device)
            ref_pc = torch.load(os.path.join(args.source_dir,'ref_pc.pt')).to(args.device)
            
            OT_solver = get_OT_solver(args,tg_fea)
            
            
        
        
        
        if args.defense is not None:
          if args.defense == 'sor':
            pre_head = SORDefense(k=2, alpha=1.1)
          elif args.defense == 'srs':
            pre_head = SRSDefense(drop_num=500)
          elif args.defense == 'dupnet':
            pre_head = DUPNet(sor_k=2, sor_alpha=1.1, npoint=2048, up_ratio=4)
          else:
            raise NotImplementedError
            
            
        adv_code,cle_code,idx = get_adversarial_sample(args,OT_solver)
        #print('adv_code.shape:',adv_code.shape)
        
        chs_labels = labels[idx].squeeze().long()
        chs_shifts = shifts[idx]
        chs_scales = scales[idx]
        ## Note !!!  chs_cle_pc is scale restored
        chs_cle_pc = ref_pc[idx] 
        
        
        
        ## save point cloud as obj file
        ot_dir=os.path.join(args.save_dir, 'pc')
        if not os.path.exists(ot_dir):       
            os.makedirs(ot_dir)
        save_cnts = 0
        save_flag = False
        
       
        
        
        '''
        if args.trans_model.lower() == 'dgcnn':
            trans_model = DGCNN(1024, 20, output_channels=args.num_class).to(args.device)
        elif args.trans_model.lower() == 'pointnet':
            trans_model = PointNetFeatureModel(args.num_class, normal_channel=False).to(args.device)
        elif args.trans_model.lower() == 'pointnet2':
            trans_model = PointNet2ClsSsg(num_classes=args.num_class).to(args.device)
        elif args.trans_model.lower() == 'pointconv':
            trans_model = PointConvDensityClsSsg(num_classes=args.num_class).to(args.device)
        elif args.trans_model.lower() == 'pct':
            trans_model = Pct(args, output_channels=args.num_class).cuda()
        else:
            print('Model not recognized')
            exit(-1)
        WEIGHTS = BEST_WEIGHTS["ShapeNetPart"][1024]
        trans_model.load_state_dict(torch.load(WEIGHTS[args.trans_model])['model_state_dict'])
        trans_model.eval()
        '''
        

        logger.info("\n args.dissim: {}".format(args.dissim))
        logger.info("\n args.attn: {}".format(args.attn))
        
        
        acc, succ, trans_num = 0.0,0.0,0.0
        
        nums_gen = adv_code.shape[0]
        bs = args.batch_size
        nb = nums_gen//bs
        all_adv_pc = []
        all_cle_pc = []
        #robust_flags = torch.zeros(nb*bs, dtype=torch.bool, device=args.device)
        robust_pc_num = 0
        
        attn = args.attn
        
        for i in tqdm(range(nb)) :
            model.eval()
            with torch.no_grad():
                # index of those correctly classified  original point clouds
                #pred,_ = classifier((chs_cle_pc[i*bs:(i+1)*bs]).view(bs,3,-1))
                
                
                #chs_cle_pc_one = ((chs_cle_pc[i*bs:(i+1)*bs])-chs_shifts[i*bs:(i+1)*bs])/chs_scales[i*bs:(i+1)*bs]
                
                chs_cle_pc_one=chs_cle_pc[i*bs:(i+1)*bs]
                for idx in range(chs_cle_pc_one.size()[0]):
                    chs_cle_pc_one[idx]=pc_normalize(chs_cle_pc_one[idx])
                '''
                chs_cle_pc_one=pc_normalize(chs_cle_pc[i*bs:(i+1)*bs])
                '''
                
                #pred,_ = classifier(chs_cle_pc_one.view(bs,3,-1))
                pred= classifier(chs_cle_pc_one.transpose(1,2).cuda())
                pred_label_cle = torch.argmax(pred, 1).detach().cuda()

                one_flag = (pred_label_cle == chs_labels[i*bs:(i+1)*bs])
                #robust_flags[start_idx:end_idx] = one_flag
                robust_pc_num += one_flag.sum()
                start_time=time.time()
                adv_pc = model.decode(adv_code[i*bs:(i+1)*bs], ref_pc.size(1)).detach()
                end_time=time.time()
                #print('generate ',adv_pc.size(0),' adv pointcloud cost time :',end_time-start_time)
                #print('adv_pc.shape:',adv_pc.shape)
                

                for idx in range(adv_pc.size()[0]):
                    adv_pc[idx]=pc_normalize(adv_pc[idx])
                '''
                adv_pc=pc_normalize(adv_pc)
                '''
                
                adv_pc_new=[]
                adv_pc_new1=torch.empty((args.batch_size,attn, 3), dtype=adv_pc.dtype)
                adv_pc_new2=torch.empty((args.batch_size, 2048-attn, 3), dtype=adv_pc.dtype)
                for j in range(adv_pc.size(0)):  
                  indices = torch.randperm(2048)[:attn]   
                  adv_pc_new1[j] = adv_pc[j, indices].clone()  
                
                for k in range(chs_cle_pc_one.size(0)):  
                  indices = torch.randperm(2048)[:2048-attn]   
                  adv_pc_new2[k] = chs_cle_pc_one[k, indices].clone()
                adv_pc_new.append(adv_pc_new1)
                adv_pc_new.append(adv_pc_new2)
                adv_pc=torch.cat(adv_pc_new,dim=1)
                
                #print('advpc:',adv_pc.size())
                
                #pred,_ = classifier(adv_pc.view(bs,3,-1))
                if args.defense is None :
                  pred = classifier(adv_pc.transpose(1,2).cuda())
                else :
                  pred = classifier(pre_head(adv_pc.transpose(1,2)).cuda())
                pred_label = torch.argmax(pred, 1).detach().cuda()
                
                
                one_flag_adv = ( pred_label== chs_labels[i*bs:(i+1)*bs])
                one_acc = (one_flag_adv[one_flag]).sum()

                
                
                
                #adv_pc_wrong=chs_cle_pc_one.clone().cuda()
                indices_ = torch.nonzero(one_flag_adv == 0).squeeze().tolist()
                
                
                #print('indics',len(indices_))
                
                #adv_pc_wrong[indices_]=adv_pc[indices_].cuda()
                #all_adv_pc.append(adv_pc_wrong.detach())
                if indices_ != [] :
                 if adv_pc[indices_].dim() == 2 :
                   adv_pc_wrong = adv_pc[indices_].unsqueeze(0).cuda()
                   cle_pc_wrong = chs_cle_pc_one[indices_].unsqueeze(0).cuda()
                 else :
                   adv_pc_wrong = adv_pc[indices_].cuda()
                   cle_pc_wrong = chs_cle_pc_one[indices_].cuda()
                 all_adv_pc.append(adv_pc_wrong.detach())
                 all_cle_pc.append(cle_pc_wrong.detach())
                
                
                
                
                acc += one_acc
                
                '''
                trans_logits = trans_model(adv_pc)
                trans_pred = torch.argmax(trans_logits, dim=-1)
                trans_num += (trans_preds != chs_labels[i*bs:(i+1)*bs]).sum().item()
                '''
                
                #succ += bs-(pred_label== pred_label_cle).sum().item()
                succ += bs-(pred_label== chs_labels[i*bs:(i+1)*bs]).sum().item()
                if save_flag:
                    #print(save_cnts)
                    for k in range(bs):
                        if one_flag_adv[k]==1 or one_flag[k]==0:
                            continue 
                        elif save_cnts >200:
                            save_flag = False
                            break
                        else:
                            #filename = all_categories_order[chs_labels[i*bs+k]]+'_cle_'+str(i)+'_'+str(save_cnts)+'.obj'
                            #write_obj_pc(os.path.join(save_dir,'pc',filename), chs_cle_pc[i*bs+k])
                            #print('adv_label:',all_categories_order[pred_label[k]],'cle_label:',all_categories_order[chs_labels[k]])
                            filename = all_categories_order[pred_label[k]]+'_'+all_categories_order[chs_labels[i*bs+k]]+str(i)+'_'+str(save_cnts)+'.obj'
                            write_obj_pc(os.path.join(save_dir,'pc',filename), adv_pc[k])
                            save_cnts +=1
                
                
        #robust_flags = torch.nonzero(robust_flags, as_tuple=False)
        try:
            all_adv_pc = torch.cat(all_adv_pc, dim=0)
            all_cle_pc = torch.cat(all_cle_pc, dim=0)
        except:
            logger.info("No adversarial examples generated!")
            return 0
        '''
        for idx in range(chs_cle_pc.size()[0]):
                    chs_cle_pc[idx]=pc_normalize(chs_cle_pc[idx])
        for idx in range(all_adv_pc.size()[0]):
                    all_adv_pc[idx]=pc_normalize(all_adv_pc[idx])
        '''
        #print("\nAccuracy on adversarial examples: {}%".format(acc/chs_labels.shape[0] * 100))
        logger.info("\nAccuracy on adversarial examples: {}%".format(acc/robust_pc_num * 100))
        logger.info("\nThe success rate of point cloud attacks: {}%\n".format(succ/(robust_pc_num) * 100))
        #print("\nThe transfer success rate is: {}%\n".format(trans_num/(bs*nb) * 100))
        #print("\nThe transfer success rate is: {}%\n".format(trans_num/acc * 100))
        
        #'''
        logger.info('Start computing metrics...')
        #metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
        print(all_adv_pc.size(),all_cle_pc.size())
        metrics = EMD_CD(all_adv_pc,all_cle_pc, batch_size=args.batch_size)
        cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
        logger.info('CD:  %.12f' % cd)
        #logger.info('EMD: %.12f' % emd)
        #'''
        return cd
    else:
        return 0
            
            
            
        

if __name__ == '__main__':
    dissims = np.array(range(1,16))/10
    #attns = np.array(range(64,1025,64))
    attns = np.array(range(64,256,8))
    #angle_threshs = 0.8+np.array(range(10))/10#45-120
    #angle_threshs[-1]=1.65
    #topks = range(4,20)

    args = parse_argument()
    args.angle_thresh = 0.9
    args.topk = 11
    args.dissim = 0.1#0.1
    args.attn = 96#152(ASR 100 CD 0.006) /208
    #main(args)
    
    #'''
    for dism in dissims:
    #for attn in attns:
    #for topk in topks:
    #for thresh in angle_threshs:
        args.dissim = dism
        #args.angle_thresh = thresh
        for attn in attns:
            args.attn = attn
            main(args)
    #'''
            
