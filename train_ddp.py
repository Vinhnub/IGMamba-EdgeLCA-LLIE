import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import warnings
warnings.filterwarnings('ignore', message='.*Grad strides do not match bucket view strides.*')
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from net.CIDNet_Mamba_separable_learning import CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime

opt = option().parse_args()

def is_master():
    """Kiểm tra xem process hiện tại có phải là rank 0 (master) không."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def seed_torch():
    # Thêm rank vào seed để mỗi GPU có một chuỗi ngẫu nhiên (chẳng hạn cho gamma augmentation) nhưng có thể lặp lại
    seed = random.randint(1, 1000000) if is_master() else 0
    if dist.is_initialized():
        seed_tensor = torch.tensor([seed], dtype=torch.long, device='cuda')
        dist.broadcast(seed_tensor, src=0)
        seed = seed_tensor.item() + get_rank()
    else:
        seed = 42 + get_rank()
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_init():
    # Khởi tạo tiến trình phân tán (DDP)
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        # Trên Windows (os.name == 'nt'), NCCL không được hỗ trợ chính thức nên tự động chuyển sang 'gloo'
        backend = 'gloo' if os.name == 'nt' else 'nccl'
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            
    seed_torch()
    cudnn.benchmark = False
    cudnn.deterministic = True
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    return local_rank

def get_model_module(model):
    """Lấy module gốc khi model được bọc bởi DDP hoặc DataParallel."""
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model

def train(epoch, local_rank):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    
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
    
    # Thiết lập sampler cho epoch hiện tại trong DDP để shuffle dữ liệu đúng cách
    if dist.is_initialized() and hasattr(training_data_loader, 'sampler') and isinstance(training_data_loader.sampler, DistributedSampler):
        training_data_loader.sampler.set_epoch(epoch)

    # Chỉ hiển thị thanh progress bar tqdm trên tiến trình chính (rank 0)
    data_iterator = tqdm(training_data_loader) if is_master() else training_data_loader
    
    for batch in data_iterator:
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        
        # random gamma function
        if opt.gamma:
            gamma = (random.randint(opt.start_gamma, opt.end_gamma) / 100.0)
            input_low = im1 ** gamma
            input_gt = im2 ** gamma
        else:
            input_low = im1
            input_gt = im2

        output_rgb = model(input_low)
        gt_rgb = im2
        
        model_module = get_model_module(model)
        output_hvi = model_module.HVIT(output_rgb)
        gt_hvi = model_module.HVIT(gt_rgb)
        
        # --- Warm-up Loss Weights ---
        warmup_epochs = opt.number_warmup_epoch
        transition_epochs = opt.number_transition_epoch
        if epoch <= warmup_epochs:
            warm_up_multiplier = 0.0
        else:
            warm_up_multiplier = min(1.0, (epoch - warmup_epochs) / (transition_epochs + 1e-8))
            
        l1_rgb = L1_loss(output_rgb, gt_rgb)
        l2_rgb = L2_loss(output_rgb, gt_rgb)
        d_rgb = D_loss(output_rgb, gt_rgb)
        p_rgb = opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        e_rgb = warm_up_multiplier * E_loss(output_rgb, gt_rgb)
        lsgd_rgb = warm_up_multiplier * LSGD_loss(output_rgb, gt_rgb)
        
        loss_rgb = l1_rgb + l2_rgb + d_rgb + p_rgb + e_rgb + lsgd_rgb
        
        l1_hvi = L1_loss(output_hvi, gt_hvi)
        l2_hvi = L2_loss(output_hvi, gt_hvi)
        d_hvi = D_loss(output_hvi, gt_hvi)
        p_hvi = opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        e_hvi = warm_up_multiplier * E_loss(output_hvi, gt_hvi)
        lsgd_hvi = warm_up_multiplier * LSGD_loss(output_hvi, gt_hvi, is_hvi=True)
        
        exp_hvi = EXP_loss(output_hvi[:, 2:3, :, :])
        
        loss_hvi = l1_hvi + l2_hvi + d_hvi + p_hvi + e_hvi + lsgd_hvi + exp_hvi
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        
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
        
        loss_print += loss.item()
        loss_last_10 += loss.item()
        pic_cnt += 1
        pic_last_10 += 1
        
        if iter == train_len and is_master():
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
            output_img = transforms.ToPILImage()((output_rgb)[0].detach().cpu().squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].detach().cpu().squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.makedirs(opt.val_folder+'training', exist_ok=True) 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
            
    return loss_print, pic_cnt

def checkpoint(epoch):
    if not is_master():
        return "./weights/train/epoch_{}.pth".format(epoch)
        
    if not os.path.exists("./weights"):          
        os.makedirs("./weights", exist_ok=True) 
    if not os.path.exists("./weights/train"):          
        os.makedirs("./weights/train", exist_ok=True)  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(get_model_module(model).state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    if is_master():
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
    else:
        raise Exception("should choose a dataset")
    
    # Khi dùng DDP, sử dụng DistributedSampler để chia đều dataset cho các GPU
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_set, shuffle=opt.shuffle)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, sampler=train_sampler, shuffle=False)
    else:
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return training_data_loader, testing_data_loader

def build_model(local_rank):
    if is_master():
        print('===> Building model ')
    model = CIDNet().cuda()
    
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        state_dict = torch.load(pth, map_location=lambda storage, loc: storage)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        if is_master():
            print(f"Loaded weights from {pth}")

    # Cấu hình Dark Focus cho các module hỗ trợ (IG_Mamba)
    ig_mamba_count = 0
    for m in model.modules():
        if hasattr(m, 'dark_focus'):
            m.dark_focus = opt.dark_focus
            ig_mamba_count += 1
    if is_master():
        focus_target = "DARK regions (vùng tối)" if opt.dark_focus else "BRIGHT regions (vùng sáng)"
        print(f"===> Dark Focus: {opt.dark_focus} -> Tập trung ưu tiên: {focus_target} (đã đồng bộ {ig_mamba_count} modules)")
        
    # Bọc model bằng DistributedDataParallel
    if dist.is_initialized():
        if is_master():
            print(f"===> Sử dụng DistributedDataParallel (DDP) trên {get_world_size()} GPUs!")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
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
    local_rank = train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model(local_rank)
    optimizer, scheduler = make_scheduler()
    L1_loss, L2_loss, P_loss, E_loss, D_loss, LSGD_loss, EXP_loss = init_loss()
    
    psnr = []
    ssim = []
    lpips = []
    start_epoch = 0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
        
    if is_master():
        if not os.path.exists(opt.val_folder):          
            os.makedirs(opt.val_folder, exist_ok=True) 
            
        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        os.makedirs("./results/training", exist_ok=True)
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
        epoch_loss, pic_num = train(epoch, local_rank)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            
            # Chỉ rank 0 thực hiện đánh giá (eval) và lưu kết quả
            if is_master():
                norm_size = True
                if opt.dataset == 'lol_v1':
                    output_folder = 'LOLv1/'
                    label_dir = opt.data_valgt_lol_v1
                if opt.dataset == 'lolv2_real':
                    output_folder = 'LOLv2_real/'
                    label_dir = opt.data_valgt_lolv2_real
                if opt.dataset == 'lolv2_syn':
                    output_folder = 'LOLv2_syn/'
                    label_dir = opt.data_valgt_lolv2_syn
                if opt.dataset == 'lol_blur':
                    output_folder = 'LOL_blur/'
                    label_dir = opt.data_valgt_lol_blur
                if opt.dataset == 'SID':
                    output_folder = 'SID/'
                    label_dir = opt.data_valgt_SID
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

                im_dir = opt.val_folder + output_folder + '*.png'
                is_lol_v1 = (opt.dataset == 'lol_v1')
                is_lolv2_real = (opt.dataset == 'lolv2_real')
                
                eval(get_model_module(model), testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                     norm_size=norm_size, LOL=is_lol_v1, v2=is_lolv2_real, alpha=0.8)
                
                # import shutil
                # target_img_path = os.path.join(opt.val_folder+output_folder, '00787.png')
                # if os.path.exists(target_img_path):
                #     debug_dir = './results/debug'
                #     os.makedirs(debug_dir, exist_ok=True)
                #     shutil.copy(target_img_path, os.path.join(debug_dir, f'00787_epoch_{epoch}.png'))
                
                avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
                print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
                print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
                print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
                psnr.append(avg_psnr)
                ssim.append(avg_ssim)
                lpips.append(avg_lpips)

                with open(f"./results/training/metrics{now}.md", "a") as f:
                    f.write(f"| {epoch} | { avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n") 
                
                avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=True)
                print("===> Avg.PSNR (GT): {:.4f} dB ".format(avg_psnr))
                print("===> Avg.SSIM (GT): {:.4f} ".format(avg_ssim))
                print("===> Avg.LPIPS (GT): {:.4f} ".format(avg_lpips))
                with open(f"./results/training/metrics{now}.md", "a") as f:
                    f.write(f"| {epoch} | { avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} | GT Mean |\n") 

        # Đảm bảo các GPU đợi GPU 0 hoàn thành eval xong rồi mới chuyển epoch tiếp theo
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
