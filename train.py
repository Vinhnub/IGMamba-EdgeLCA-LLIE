import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet_base_w_edge_tiny import CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime

opt = option().parse_args()

def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
def train(epoch):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    
    # Add accumulators for individual losses
    l1_sum = 0
    l2_sum = 0
    d_sum = 0
    p_sum = 0
    e_sum = 0
    lsgd_sum = 0
    exp_sum = 0

    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:

            gamma = (
                random.randint(
                    opt.start_gamma,
                    opt.end_gamma
                ) / 100.0
            )

            input_low = im1 ** gamma
            input_gt = im2 ** gamma

        else:

            input_low = im1
            input_gt = im2

        output_rgb = model(
            input_low
        )
            
        # with torch.no_grad():
        #     _, feat_teacher = model(
        #         input_gt,
        #         return_feats=True
        #     )

        gt_rgb = im2
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)
        # --- Warm-up Loss Weights ---
        warmup_epochs = opt.number_warmup_epoch
        transition_epochs = opt.number_transition_epoch
        
        if epoch <= warmup_epochs:
            warm_up_multiplier = 0.0
        else:
            # Tăng dần tuyến tính từ 0.0 đến 1.0
            warm_up_multiplier = min(1.0, (epoch - warmup_epochs) / (transition_epochs + 1e-8))
            
        # Tính toán riêng biệt từng loss cho RGB
        l1_rgb = L1_loss(output_rgb, gt_rgb)
        l2_rgb = L2_loss(output_rgb, gt_rgb)
        d_rgb = D_loss(output_rgb, gt_rgb)
        p_rgb = opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        e_rgb = warm_up_multiplier * E_loss(output_rgb, gt_rgb)
        lsgd_rgb = warm_up_multiplier * LSGD_loss(output_rgb, gt_rgb)
        
        loss_rgb = l1_rgb + l2_rgb + d_rgb + p_rgb + e_rgb + lsgd_rgb
        
        # Tính toán riêng biệt từng loss cho HVI
        l1_hvi = L1_loss(output_hvi, gt_hvi)
        l2_hvi = L2_loss(output_hvi, gt_hvi)
        d_hvi = D_loss(output_hvi, gt_hvi)
        p_hvi = opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        e_hvi = warm_up_multiplier * E_loss(output_hvi, gt_hvi)
        lsgd_hvi = warm_up_multiplier * LSGD_loss(output_hvi, gt_hvi, is_hvi=True)
        
        # Exposure Loss tính trực tiếp trên kênh I (Intensity) của output_hvi
        exp_hvi = EXP_loss(output_hvi[:, 2:3, :, :])
        
        loss_hvi = l1_hvi + l2_hvi + d_hvi + p_hvi + e_hvi + lsgd_hvi + exp_hvi
        
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        
        # Tích lũy giá trị loss để in ra
        l1_sum += (l1_rgb.item() + opt.HVI_weight * l1_hvi.item())
        l2_sum += (l2_rgb.item() + opt.HVI_weight * l2_hvi.item())
        d_sum += (d_rgb.item() + opt.HVI_weight * d_hvi.item())
        p_sum += (p_rgb.item() + opt.HVI_weight * p_hvi.item())
        e_sum += (e_rgb.item() + opt.HVI_weight * e_hvi.item())
        lsgd_sum += (lsgd_rgb.item() + opt.HVI_weight * lsgd_hvi.item())
        exp_sum += (opt.HVI_weight * exp_hvi.item())
        
        iter += 1
        
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_print = loss_print + loss.item()
        loss_last_10 = loss_last_10 + loss.item()
        #loss_lsgd_print += loss_lsgd.item()
        pic_cnt += 1
        pic_last_10 += 1
        if iter == train_len:
            print("===> Epoch[{}]: Total Loss: {:.4f} || L1: {:.4f} | L2: {:.4f} | D(SSIM): {:.4f} | P(VGG): {:.4f} | Edge: {:.4f} | LSGD: {:.4f} | EXP: {:.4f} || lr={}.".format(
                epoch,
                loss_last_10/pic_last_10, 
                l1_sum/pic_cnt,
                l2_sum/pic_cnt,
                d_sum/pic_cnt,
                p_sum/pic_cnt,
                e_sum/pic_cnt,
                lsgd_sum/pic_cnt,
                exp_sum/pic_cnt,
                optimizer.param_groups[0]['lr']))
            loss_last_10 = 0
            pic_last_10 = 0
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    return loss_print, pic_cnt
                

def checkpoint(epoch):
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print(f'===> Loading datasets: {opt.dataset}')
    if opt.dataset == 'lol_v1':
        train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_v1)
        
    elif opt.dataset == 'lol_blur':
        train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_blur)

    elif opt.dataset == 'lolv2_real':
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_real)
        
    elif opt.dataset == 'lolv2_syn':
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_syn)
    
    elif opt.dataset == 'SID':
        train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_SID)
        
    elif opt.dataset == 'SICE_mix':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
        
    elif opt.dataset == 'SICE_grad':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
        
    elif opt.dataset == 'fivek':
        train_set = get_fivek_training_set(opt.data_train_fivek,size=opt.cropSize)
        test_set = get_fivek_eval_set(opt.data_val_fivek)
    elif opt.dataset == 'small':
        train_set = get_small_training_set(opt.data_train_small, size=opt.cropSize)
        test_set = get_small_eval_set(opt.data_val_small)
    else:
        raise Exception("should choose a dataset")
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = CIDNet().cuda()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))

    # Cấu hình Dark Focus cho các module hỗ trợ (IG_Mamba)
    ig_mamba_count = 0
    for m in model.modules():
        if hasattr(m, 'dark_focus'):
            m.dark_focus = opt.dark_focus
            ig_mamba_count += 1
    focus_target = "DARK regions (vùng tối)" if opt.dark_focus else "BRIGHT regions (vùng sáng)"
    print(f"===> Dark Focus: {opt.dark_focus} -> Tập trung ưu tiên: {focus_target} (đã đồng bộ {ig_mamba_count} modules)")
    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    L2_weight   = opt.L2_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1.0
    LSGD_weight = opt.LSGD_weight
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    L2_loss= L2Loss(loss_weight=L2_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    LSGD_loss = RegionLSGDLoss(loss_weight=LSGD_weight, dark_focus=opt.dark_focus).cuda()
    EXP_loss = ExposureControlLoss(patch_size=16, mean_val=0.6, loss_weight=0.0).cuda()

    return (
        L1_loss,
        L2_loss,
        P_loss,
        E_loss,
        D_loss,
        LSGD_loss,
        EXP_loss
    )

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss, L2_loss, P_loss, E_loss, D_loss, LSGD_loss, EXP_loss = init_loss()
    
    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) 
        
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./results/training/metrics{now}.md", "w") as f:
        f.write("dataset: "+ opt.dataset + "\n")  
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"HVI_weight: {opt.HVI_weight}\n")  
        f.write(f"L1_weight: {opt.L1_weight}\n")  
        f.write(f"L2_weight: {opt.L2_weight}\n")  
        f.write(f"D_weight: {opt.D_weight}\n")  
        f.write(f"E_weight: {opt.E_weight}\n")  
        f.write(f"P_weight: {opt.P_weight}\n")  
        f.write(f"LSGD_weight: {opt.LSGD_weight}\n")  
        f.write(f"dark_focus: {opt.dark_focus}\n")  
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size = True

            # LOL three subsets
            if opt.dataset == 'lol_v1':
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.dataset == 'lolv2_real':
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.dataset == 'lolv2_syn':
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.dataset == 'lol_blur':
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            if opt.dataset == 'SID':
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.dataset == 'SICE_mix':
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.dataset == 'SICE_grad':
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
                
            if opt.dataset == 'fivek':
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False
                
            if opt.dataset == 'small':
                output_folder = 'SMALL/'
                label_dir = opt.data_valgt_small
                norm_size = False

            im_dir = opt.val_folder + output_folder + '*.png'
            is_lol_v1 = (opt.dataset == 'lol_v1')
            is_lolv2_real = (opt.dataset == 'lolv2_real')
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                 norm_size=norm_size, LOL=is_lol_v1, v2=is_lolv2_real, alpha=0.8)
            
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)
            with open(f"./results/training/metrics{now}.md", "a") as f:
                f.write(f"| {epoch} | { avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n") 

            # --- Eval with GT Mean
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=True)
            print("===> Avg.PSNR (GT): {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM (GT): {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS (GT): {:.4f} ".format(avg_lpips))
            with open(f"./results/training/metrics{now}.md", "a") as f:
                f.write(f"| {epoch} | { avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} | GT Mean |\n") 

        torch.cuda.empty_cache()