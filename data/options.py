import argparse

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def option():
    # Training settings
    parser = argparse.ArgumentParser(description='CIDNet')
    parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=5, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=_str2bool, default=True)
    parser.add_argument('--shuffle', type=_str2bool, default=True)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for dataloader to use')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=_str2bool, default=False)
    parser.add_argument('--cos_restart', type=_str2bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=_str2bool, default=True, help='turn False to train without warmup') 

    # train datasets
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/LOLv1/train')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='/kaggle/input/datasets/vinhnub/lolv2-real/LOLv2-real/Train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')
    parser.add_argument('--data_train_fivek'        , type=str, default='./datasets/FiveK/train')
    parser.add_argument('--data_train_small'        , type=str, default='./dataset/SMALL/train')

    # validation input
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/LOLv1/test/low')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='/kaggle/input/datasets/vinhnub/lolv2-real/LOLv2-real/Test/Input')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='./datasets/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'            , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_test_fivek'         , type=str, default='./datasets/FiveK/test/input')
    parser.add_argument('--data_val_small'          , type=str, default='./dataset/SMALL/val/low')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/LOLv1/test/high/')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='/kaggle/input/datasets/vinhnub/lolv2-real/LOLv2-real/Test/GT/')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'          , type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_fivek'        , type=str, default='./datasets/FiveK/test/target/')
    parser.add_argument('--data_valgt_small'        , type=str, default='./dataset/SMALL/val/high/')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--L2_weight', type=float, default=0.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=0.01)
    parser.add_argument('--LSGD_weight', type=float, default=0.0)
    parser.add_argument('--dark_focus', type=_str2bool, default=True, help='Enable dark focus for both IG_Mamba and LSGD loss (False focuses on bright regions)')
    parser.add_argument('--number_warmup_epoch', '--number_warmup_epochs', type=int, default=0, help='Number of warmup epochs for loss weights where weights are 0')
    parser.add_argument('--number_transition_epoch', '--number_transition_epochs', type=int, default=1, help='Number of transition epochs for loss weights to linearly scale from 0 to 1')
    
    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--gamma', type=_str2bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=_str2bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=_str2bool, default=True, help='if gradient fluctuates too much, turn-on it')
    
    
    # choose which dataset you want to train
    parser.add_argument('--dataset', type=str, default='lol_v1',
    choices=['lol_v1',
             'lolv2_real',
             'lolv2_syn',
             'lol_blur', 
             'SID',
             'SICE_mix',
             'SICE_grad',
             'fivek',
             'small'],
    help='Select the dataset to train on (default: %(default)s)')

    return parser
