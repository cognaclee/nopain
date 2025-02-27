import os
import time
import argparse
import torch
import sys
from tqdm.auto import tqdm
import yaml
from utils.dataset import *
from utils.misc import *
from utils.data import *
from utils.ModelNetDataLoader import *
from models.autoencoder import *
from evaluation import EMD_CD
from ot_utils.ot_util import get_OT_solver, get_adversarial_sample, get_adversarial_sample_with_target
#from models.classification import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, PCT
from models.classifier import  PointNetCls,PointNet2ClsSsg,PointNet2ClsMsg,DGCNN,PointConvDensityClsSsg,PCT
from models.config import BEST_WEIGHTS
from defense import *
from utils.set_distance import hausdorff

from utils.scanobjectnn.ScanObjectNNDataset import ScanObjectNN
#from pretrained.checkpoint.scanobjectnn
#from defense.ConvONet.opt_defense import defend_point_cloud


from models.pointmamba.point_mamba import PointMamba
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
    parser.add_argument('--ckpt', type=str, default="/home/duxiaoyu/code/nopain/pretrained/ckpt_0.008750_579000.pt")
    parser.add_argument('--classifier', type=str, default='dgcnn', metavar='MODEL',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pointconv','PCT','pointmamba'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv].')
    parser.add_argument('--trans_model', type=str, default='pointnet', 
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pointconv','pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv].')
    parser.add_argument('--num_class', type=int, default=15, help='class numbers')
    parser.add_argument('--categories', type=str_list, default=['airplane'])
    
    
    parser.add_argument('--save_dir', type=str, default='./results/scanobjectnn_nobg')
    parser.add_argument('--device', type=str, default='cuda')
    # Datasets and loaders
    parser.add_argument('--dataset_path', type=str, default="/home/duxiaoyu/code/nopain/data/ScanObjectNN/h5_files/main_split_nobg/")
    parser.add_argument('--batch_size', type=int, default=16)
    # OT 
    parser.add_argument('--h_name', type=str, default="/home/duxiaoyu/code/nopain/results/scanobjectnn_nobg/AE_airplane_1738066730/ot/h_best.pt", help='file name of OT Brenier h')#None
    #parser.add_argument('--h_name', type=str, default=None, help='file name of OT Brenier h')
    parser.add_argument('--source_dir', type=str, default="/home/duxiaoyu/code/nopain/results/scanobjectnn_nobg/AE_airplane_1738066730/", help='source file directory')
    
    
    parser.add_argument('--max_iter', type=int, default=10000,help='max iters of train ot')#
    parser.add_argument('--lr_ot', type=float, default=1e-3,help='learning rate of OT')#
    parser.add_argument('--bat_size_sr', type=int, default=5000,help='Size of mini-batch of Monte-Carlo source samples on device')
    parser.add_argument('--bat_size_tg', type=int, default=500,help='Size of mini-batch of Monte-Carlo target samples on device')
    parser.add_argument('--topk', type=int, default=11, help='The nearest k samples around current sample')
    parser.add_argument('--angle_thresh', type=float, default=1.6,help='the threshold of the angle between two samples')
    parser.add_argument('--dissim', type=float, default=1.2,help='the degree of dissimilarity with the original sample')
    parser.add_argument('--attn', type=int, default=24,help='num of points to attack')
    parser.add_argument('--adv_samp_nums', type=int, default=1024,help='the required number of adversarial samples')
    
    parser.add_argument('--num_category', default=15, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_point', type=int, default=2048,help='num of points to use')
    
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    
    
    parser.add_argument('--defense', type=str, default=None,choices=['sor','srs','dupnet'])
    args = parser.parse_args()
    return args


def main():
    ## Shapenet55
    """
    all_categories_order = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'bottle', 'bowl',
                            'bus', 'cabinet', 'can', 'camera', 'cap', 'car', 'chair', 'clock',
                            'dishwasher', 'monitor', 'table', 'telephone', 'tin_can', 'tower', 'train', 'keyboard',
                            'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 
                            'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 
                            'pillow', 'pistol', 'pot', 'printer', 'remote_control', 'rifle', 'rocket', 'skateboard',
                            'sofa', 'stove', 'vessel','washer', 'cellphone', 'birdhouse', 'bookshelf']
    """
    ## Modelnet40
    
    all_categories_order = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
                            'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
                            'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                            'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 
                            'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    
    ## ShapenetPart
    #all_categories_order=['airplane','bench','car','chair','sofa','table','telephone']
    """
    all_categories_order=['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar','knife', 
                          'lamp', 'laptop', 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    """
    
    args = parse_argument()
    actions = ['ot_based_attack'] #'extract_feature'
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
    ckpt = torch.load(args.ckpt,map_location='cuda')
    seed_all(ckpt['args'].seed)
    #print('ckpt[args].scale_mode=',ckpt['args'].scale_mode)

    # Datasets and loaders
    logger.info('Loading datasets...')
    #test_dset = ModelNet40(path=args.dataset_path,cates=args.categories,split='test',scale_mode=ckpt['args'].scale_mode)
    #test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)
    test_dset = ScanObjectNN(root=args.dataset_path, subset='test')
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    print('Dataset length = ',len(test_dset))
    
    # Model
    logger.info('Loading model...')
    model = AutoEncoder(ckpt['args']).to('cuda')
    model.load_state_dict(ckpt['state_dict'])
    #torch.backends.cudnn.enabled = True
    
    if 'extract_feature' in actions:
    
    
    
    
    
    
            
            all_ref = []
            all_recons = []
            all_codes = []
            all_labels = []
            #all_shifts = []
            #all_scales = []
            for i, batch in enumerate(tqdm(test_loader)):
                ref = batch['pointcloud'].to(args.device)
                #shift = batch['shift'].to(args.device)
                #scale = batch['scale'].to(args.device)
                label = batch['label'].to(args.device)
                print(label)
                
                
                
                
                
                model.eval()
                with torch.no_grad():
                    code = model.encode(ref)
                    recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()
    
                #ref = ref * scale + shift
                #recons = recons * scale + shift
    
                all_ref.append(ref.detach().cpu())
                all_recons.append(recons.detach().cpu())
                all_codes.append(code.detach().cpu())
                
                all_labels.append(label.detach().cpu())
                #all_shifts.append(shift.detach().cpu())
                #all_scales.append(scale.detach().cpu())
            
    
            all_ref = torch.cat(all_ref, dim=0)
            all_recons = torch.cat(all_recons, dim=0)
            all_codes = torch.cat(all_codes, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            #all_shifts = torch.cat(all_shifts, dim=0)
            #all_scales = torch.cat(all_scales, dim=0)
            
        
            logger.info('Saving point clouds...')
            torch.save(all_ref, os.path.join(save_dir, 'ref_pc.pt'))
            torch.save(all_recons, os.path.join(save_dir,'dec_pc.pt'))
            torch.save(all_codes, os.path.join(save_dir,'features.pt'))
            torch.save(all_labels, os.path.join(save_dir,'labels.pt'))
            #torch.save(all_shifts, os.path.join(save_dir, 'shifts.pt'))
            #torch.save(all_scales, os.path.join(save_dir,'scales.pt'))
            
            args.save_dir = save_dir
        
    
    if 'ot_based_attack' in actions:   
        
        if 'extract_feature' in actions:
            #args.source_dir = save_dir
            OT_solver = get_OT_solver(args,all_codes)
            labels = all_labels
            #shifts = all_shifts
            #scales = all_scales
            ref_pc = all_ref
            
            
        else:
            args.save_dir = args.source_dir
            tg_fea = torch.load(os.path.join(args.source_dir,'features.pt')).to(args.device)
            labels = torch.load(os.path.join(args.source_dir,'labels.pt')).to(args.device)
            #shifts = torch.load(os.path.join(args.source_dir,'shifts.pt')).to(args.device)
            #scales = torch.load(os.path.join(args.source_dir,'scales.pt')).to(args.device)
            ref_pc = torch.load(os.path.join(args.source_dir,'ref_pc.pt')).to(args.device)
            
            
            OT_solver = get_OT_solver(args,tg_fea)
            
            
        
        if args.defense is not None :
          if args.defense == 'sor':
            pre_head = SORDefense(k=2, alpha=1.1)
          elif args.defense == 'srs':
            pre_head = SRSDefense(drop_num=500)
          elif args.defense == 'dupnet':
            pre_head = DUPNet(sor_k=2, sor_alpha=1.1, npoint=2048, up_ratio=4).cuda()
          else:
            raise NotImplementedError
        
        '''
        adv_code,cle_code,idx = get_adversarial_sample(args,OT_solver)
        print('adv_code.shape:',adv_code.shape)
        chs_labels = labels[idx].squeeze() 
        chs_cle_pc = ref_pc[idx] 
        print('chs_cle_pc.shape:',chs_cle_pc.shape)
        '''
        '''
        idx_=[30,726,1097,1195,1322,1524,1979]
        input_code = OT_solver.tg_fea.view(OT_solver.num_tg, -1)[idx_]
        adv_code = get_adversarial_sample_with_target(args,OT_solver,input_code)
        chs_labels = labels[idx_]
        chs_cle_pc = ref_pc[idx_]
        for kk in range(args.topk-2):
            chs_labels = torch.cat((chs_labels,labels[idx_]),dim=0)
            chs_cle_pc = torch.cat((chs_cle_pc,ref_pc[idx_]),dim=0)
        chs_labels = chs_labels.squeeze() 
        '''
        adv_code,cle_code,idx = get_adversarial_sample(args,OT_solver)
        print('adv_code.shape:',adv_code.shape)
        chs_labels = labels[idx].squeeze() 
        #chs_shifts = shifts[idx]
        #chs_scales = scales[idx]
        ## Note !!!  chs_cle_pc is scale restored
        chs_cle_pc = ref_pc[idx] 
        ## save point cloud as obj file
        ot_dir=os.path.join(args.save_dir, 'pc')
        if not os.path.exists(ot_dir):       
            os.makedirs(ot_dir)
        save_cnts = 0
        save_flag = True
        
        # build classifier
        
        #classifier = get_classifier().to(args.device)
        
        if args.classifier.lower() == 'dgcnn':
            classifier = DGCNN(1024, 20, output_channels=args.num_class).to(args.device)
            #classifier = DGCNN(args, output_channels=args.num_class).to(args.device)
        elif args.classifier.lower() == 'pointnet':
            classifier = PointNetCls(k=args.num_class,feature_transform=True).to(args.device)
            #classifier = PointNetFeatureModel(args.num_class, normal_channel=False).to(args.device)
        elif args.classifier.lower() == 'pointnet2':
            classifier = PointNet2ClsSsg(num_classes=args.num_class).to(args.device)
        elif args.classifier.lower() == 'pointconv':
            classifier = PointConvDensityClsSsg(num_classes=args.num_class).to(args.device)
        elif args.classifier.lower() == 'pct':
            classifier = PCT(output_channels=args.num_class).cuda()
        elif args.classifier.lower() == 'pointmamba':
            cfg = yaml.load(open("/home/duxiaoyu/code/PointMamba/cfgs/finetune_scan_objonly.yaml"), Loader=yaml.FullLoader)['model']
            print(cfg)
            classifier = PointMamba(cfg).cuda()
        else:
            print('Model not recognized')
            exit(-1)
        WEIGHTS = BEST_WEIGHTS["scanobjectnn"][1024]
        if args.classifier.lower() == 'pointmamba':
          classifier.load_model_from_ckpt(WEIGHTS[args.classifier])
        else:
          classifier = torch.nn.DataParallel(classifier).cuda()
          classifier.load_state_dict(torch.load(WEIGHTS[args.classifier]))
        #classifier=classifier.cuda()
        classifier.eval()
        #if args.classifier.lower() == 'pointnet':
        #  classifier.load_state_dict(torch.load(WEIGHTS[args.classifier])['model_state_dict']).cuda()
        #else:
        '''
        new_state_dict = {}
        for k,v in torch.load(WEIGHTS[args.classifier]).items():
          new_state_dict[k[7:]] = v
        classifier.load_state_dict(new_state_dict)
        '''
        
        '''
        dgcnn = DGCNN(1024, 20, output_channels=args.num_class).to(args.device)
        dgcnn = torch.nn.DataParallel(dgcnn).cuda()
        dgcnn.load_state_dict(torch.load(WEIGHTS['dgcnn']))
        dgcnn.eval()
        pointnet = PointNetCls(k=args.num_class,feature_transform=True).to(args.device)
        pointnet = torch.nn.DataParallel(pointnet).cuda()
        pointnet.load_state_dict(torch.load(WEIGHTS['pointnet']))
        pointnet.eval()
        pointnet2 = PointNet2ClsSsg(num_classes=args.num_class).to(args.device)
        pointnet2 = torch.nn.DataParallel(pointnet2).cuda()
        pointnet2.load_state_dict(torch.load(WEIGHTS['pointnet2']))
        pointnet2.eval()
        pointconv = PointConvDensityClsSsg(num_classes=args.num_class).to(args.device)
        pointconv = torch.nn.DataParallel(pointconv).cuda()
        pointconv.load_state_dict(torch.load(WEIGHTS['pointconv']))
        pointconv.eval()
        '''
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
        


        
        
        acc, succ, trans_num = 0.0,0.0,0.0
        
        nums_gen = adv_code.shape[0]
        bs = args.batch_size
        nb = nums_gen//bs
        all_adv_pc = []
        all_cle_pc = []
        all_cle_label = []
        all_adv_label = []
        
        
        dgcnn_adv_label = []
        pointnet_adv_label = []
        pointnet2_adv_label = []
        pointconv_adv_label = []
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
                
                
                
                
                #for idx in range(chs_cle_pc_one.size()[0]):
                #    chs_cle_pc_one[idx]=pc_normalize(chs_cle_pc_one[idx])
                
                #pred,_ = classifier(chs_cle_pc_one.view(bs,3,-1))
                if args.classifier.lower() == 'pointmamba':
                  pred = classifier(chs_cle_pc_one.cuda())
                else :
                  pred = classifier(chs_cle_pc_one.transpose(1,2).cuda())
                pred_label_cle = torch.argmax(pred, 1).detach().cuda()
                print(pred_label_cle)
                print(chs_labels[i*bs:(i+1)*bs])
                one_flag = (pred_label_cle == chs_labels[i*bs:(i+1)*bs])
                #robust_flags[start_idx:end_idx] = one_flag
                robust_pc_num += one_flag.sum()
                print('robust:',robust_pc_num)
                start_time=time.time()
                adv_pc = model.decode(adv_code[i*bs:(i+1)*bs], ref_pc.size(1), flexibility=ckpt['args'].flexibility).detach()
                end_time=time.time()
                #print('generate ',adv_pc.size(0),' adv pointcloud cost time :',end_time-start_time)
                #print('adv_pc.shape:',adv_pc.shape)
                ## classification before scale or behind?
                #pred,_ = classifier(adv_pc.view(bs,3,-1))
                
                
                #adv_pc = adv_pc * chs_scales[i*bs:(i+1)*bs] + chs_shifts[i*bs:(i+1)*bs]
                
                

                #for idx in range(adv_pc.size()[0]):
                #    adv_pc[idx]=pc_normalize(adv_pc[idx])
  
                
                adv_pc_new=[]
                adv_pc_new1=torch.empty((args.batch_size,attn, 3), dtype=adv_pc.dtype)
                adv_pc_new2=torch.empty((args.batch_size, 2048-attn,3), dtype=adv_pc.dtype)
                for j in range(adv_pc.size(0)):  
                  indices = torch.randperm(2048)[:attn]   
                  adv_pc_new1[j] = adv_pc[j, indices].clone()  
                
                for k in range(chs_cle_pc_one.size(0)):  
                  indices = torch.randperm(2048)[:2048-attn]   
                  adv_pc_new2[k] = chs_cle_pc_one[k, indices].clone()
                adv_pc_new.append(adv_pc_new1)
                adv_pc_new.append(adv_pc_new2)
                adv_pc=torch.cat(adv_pc_new,dim=1)
                
                
                #pred,_ = classifier(adv_pc.view(bs,3,-1))
                if args.classifier.lower() == 'pointmamba':
                  pred = classifier(adv_pc.cuda())
                else:
                  pred = classifier(adv_pc.transpose(1,2).cuda())
                pred_label = torch.argmax(pred, 1).detach().cuda()
                one_flag_adv = ( pred_label== chs_labels[i*bs:(i+1)*bs])
                one_acc = (one_flag_adv[one_flag]).sum()
                #print('one_pred_label_adv')
                #print(one_flag_adv)
                
                
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
                  one_chs_label = chs_labels[i*bs:(i+1)*bs]
                  if pred_label[indices_].dim() == 0 :    
                    all_adv_label.append(pred_label[indices_].unsqueeze(0))
                    all_cle_label.append(one_chs_label[indices_].unsqueeze(0))
                    '''
                    dgcnn_adv_label.append(dgcnn_label[indices_].unsqueeze(0))
                    pointnet_adv_label.append(pointnet_label[indices_].unsqueeze(0))
                    pointnet2_adv_label.append(pointnet2_label[indices_].unsqueeze(0))
                    pointconv_adv_label.append(pointconv_label[indices_].unsqueeze(0))
                    '''
                  else :
                    all_adv_label.append(pred_label[indices_])
                    all_cle_label.append(one_chs_label[indices_])
                    '''
                    dgcnn_adv_label.append(dgcnn_label[indices_])
                    pointnet_adv_label.append(pointnet_label[indices_])
                    pointnet2_adv_label.append(pointnet2_label[indices_])
                    pointconv_adv_label.append(pointconv_label[indices_])
                    '''
                
                acc += one_acc
                
                
                #succ += bs-(pred_label== pred_label_cle).sum().item()
                succ += one_flag.sum()- ((one_flag_adv == 1) & (one_flag == 1)).sum()
                #succ += bs-(pred_label== chs_labels[i*bs:(i+1)*bs]).sum().item()
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
                            #filename = all_categories_order[pred_label[k]]+'_'+all_categories_order[chs_labels[i*bs+k]]+str(i)+'_'+str(save_cnts)+'.obj'
                            #write_obj_pc(os.path.join(save_dir,'pc',filename), adv_pc[k])
                            save_cnts +=1
                
                
        #robust_flags = torch.nonzero(robust_flags, as_tuple=False)
        
        all_adv_pc = torch.cat(all_adv_pc, dim=0)
        all_cle_pc = torch.cat(all_cle_pc, dim=0)
        all_adv_label = torch.cat(all_adv_label, dim=0)
        all_cle_label = torch.cat(all_cle_label, dim=0)
        '''
        dgcnn_adv_label =torch.cat(dgcnn_adv_label,dim=0)
        pointnet_adv_label = torch.cat(pointnet_adv_label,dim=0)
        pointnet2_adv_label = torch.cat(pointnet2_adv_label,dim=0)
        pointconv_adv_label = torch.cat(pointconv_adv_label,dim=0)
        '''
        #print("\nAccuracy on adversarial examples: {}%".format(acc/chs_labels.shape[0] * 100))
        
        print(succ)
        print(robust_pc_num)
        print("\nAccuracy on adversarial examples: {}%".format(acc/robust_pc_num * 100))
        print("\nThe success rate of point cloud attacks: {}%\n".format(succ/robust_pc_num * 100))
        #print("\nThe transfer success rate is: {}%\n".format(trans_num/(bs*nb) * 100))
        #print("\nThe transfer success rate is: {}%\n".format(trans_num/acc * 100))
        
        #'''
        logger.info('Start computing metrics...')
        
        #metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
        print(all_adv_pc.size(),all_cle_pc.size(),all_cle_label.size())
        metrics = EMD_CD(all_adv_pc, all_cle_pc, batch_size=args.batch_size)
        '''
        metrics = hausdorff.forward(all_adv_pc, all_cle_pc)
        clouds_with_distances = list(zip(metrics.tolist(), all_adv_pc))  
        sorted_clouds = sorted(clouds_with_distances, key=lambda x: x[0], reverse=False)  
        sorted_distances, sorted_advpc = zip(*sorted_clouds) 
        sorted_advpc = torch.stack(sorted_advpc)
        clouds_with_distances = list(zip(metrics.tolist(), all_cle_pc))  
        sorted_clouds = sorted(clouds_with_distances, key=lambda x: x[0], reverse=False)  
        sorted_distances, sorted_clepc = zip(*sorted_clouds) 
        sorted_clepc = torch.stack(sorted_clepc)
        clouds_with_distances = list(zip(metrics.tolist(), all_cle_label))  
        sorted_clouds = sorted(clouds_with_distances, key=lambda x: x[0], reverse=False)  
        sorted_distances, sorted_clelabel = zip(*sorted_clouds) 
        sorted_clelabel = torch.stack(sorted_clelabel)
        clouds_with_distances = list(zip(metrics.tolist(), all_adv_label))  
        sorted_clouds = sorted(clouds_with_distances, key=lambda x: x[0], reverse=False)  
        sorted_distances, sorted_advlabel = zip(*sorted_clouds) 
        sorted_advlabel = torch.stack(sorted_advlabel)
        
        clouds_with_distances = list(zip(metrics.tolist(), dgcnn_adv_label))  
        sorted_clouds = sorted(clouds_with_distances, key=lambda x: x[0], reverse=False)  
        sorted_distances, sorted_dgcnnlabel = zip(*sorted_clouds) 
        sorted_dgcnnlabel = torch.stack(sorted_dgcnnlabel)
        clouds_with_distances = list(zip(metrics.tolist(), pointnet_adv_label))  
        sorted_clouds = sorted(clouds_with_distances, key=lambda x: x[0], reverse=False)  
        sorted_distances, sorted_pointnetlabel = zip(*sorted_clouds) 
        sorted_pointnetlabel = torch.stack(sorted_pointnetlabel)
        clouds_with_distances = list(zip(metrics.tolist(), pointnet2_adv_label))  
        sorted_clouds = sorted(clouds_with_distances, key=lambda x: x[0], reverse=False)  
        sorted_distances, sorted_pointnet2label = zip(*sorted_clouds) 
        sorted_pointnet2label = torch.stack(sorted_pointnet2label)
        clouds_with_distances = list(zip(metrics.tolist(), pointconv_adv_label))  
        sorted_clouds = sorted(clouds_with_distances, key=lambda x: x[0], reverse=False)  
        sorted_distances, sorted_pointconvlabel = zip(*sorted_clouds) 
        sorted_pointconvlabel = torch.stack(sorted_pointconvlabel)
        for i in range(sorted_advpc.size(0)):
          filenameadv = str(i) +'_cle_'+all_categories_order[sorted_clelabel[i]]+\
          '_pointnet_'+all_categories_order[sorted_pointnetlabel[i]]+\
          '_pointnet2_'+all_categories_order[sorted_pointnet2label[i]]+\
          '_pointconv_'+all_categories_order[sorted_pointconvlabel[i]]+\
          '_dgcnn_'+all_categories_order[sorted_dgcnnlabel[i]]+\
          '_pct_'+all_categories_order[sorted_advlabel[i]]+\
          '_adv.obj'
          filenamecle = str(i) +'_cle_'+all_categories_order[sorted_clelabel[i]]+\
          '_pointnet_'+all_categories_order[sorted_pointnetlabel[i]]+\
          '_pointnet2_'+all_categories_order[sorted_pointnet2label[i]]+\
          '_pointconv_'+all_categories_order[sorted_pointconvlabel[i]]+\
          '_dgcnn_'+all_categories_order[sorted_dgcnnlabel[i]]+\
          '_pct_'+all_categories_order[sorted_advlabel[i]]+\
          '_cle.obj'
          write_obj_pc(os.path.join(save_dir,'pc',filenameadv), sorted_advpc[i]) 
          write_obj_pc(os.path.join(save_dir,'pc',filenamecle), sorted_clepc[i])    
        ''' 
        cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
        logger.info('CD:  %.12f' % cd)
        #logger.info('EMD: %.12f' % emd)
        #'''
            
            
        


if __name__ == '__main__':
    main()

