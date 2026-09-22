import sys
import os
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data.options import option as get_option
from data.options_2stage import option as get_option_2stage
from loss.losses import RegionLSGDLoss, DarkFocusedGradientLoss
from net.IG_Mamba import IG_Mamba
from net.CIDNet_Mamba_separable_learning import CIDNet as CIDNet_Separable
from net.CIDNet_Mamba_separable_learning_edge import CIDNet as CIDNet_Edge

def test_options():
    print("=" * 60)
    print("1. KIEM TRA PARSING TUY CHON --dark_focus TRONG OPTIONS")
    print("=" * 60)
    
    # 1a. Test default options.py
    opt_default = get_option().parse_args([])
    assert hasattr(opt_default, 'dark_focus'), "options.py thieu argument dark_focus"
    assert opt_default.dark_focus is True, f"Mac dinh dark_focus trong options.py phai la True, nhan: {opt_default.dark_focus}"
    print("  -> [PASS] options.py mac dinh: dark_focus = True")

    # 1b. Test CLI flag True / False trong options.py
    opt_true = get_option().parse_args(['--dark_focus', 'True'])
    assert opt_true.dark_focus is True
    opt_false = get_option().parse_args(['--dark_focus', 'False'])
    assert opt_false.dark_focus is False
    print("  -> [PASS] options.py nhan dung co CLI: --dark_focus True / False")

    # 1c. Test options_2stage.py
    opt_2s_default = get_option_2stage().parse_args([])
    assert hasattr(opt_2s_default, 'dark_focus'), "options_2stage.py thieu argument dark_focus"
    assert opt_2s_default.dark_focus is True
    opt_2s_false = get_option_2stage().parse_args(['--dark_focus', 'False'])
    assert opt_2s_false.dark_focus is False
    print("  -> [PASS] options_2stage.py nhan dung co CLI: --dark_focus True / False")

def test_model_sync():
    print("\n" + "=" * 60)
    print("2. KIEM TRA DONG BO DARK_FOCUS CHO CAC MODULE IG_MAMBA")
    print("=" * 60)

    # 2a. Standalone IG_Mamba
    m = IG_Mamba(dim=36)
    assert m.dark_focus is True, "IG_Mamba mac dinh phai co dark_focus = True"
    m.dark_focus = False
    assert m.dark_focus is False and m.attn.dark_focus is False
    print("  -> [PASS] IG_Mamba doc lap chuyen doi dark_focus chinh xac.")

    # 2b. CIDNet Separable Learning
    model = CIDNet_Separable(channels=[16, 16, 32, 64], heads=[1, 2, 4, 8])
    assert model.dark_focus is True, "CIDNet Separable mac dinh phai co dark_focus = True"
    
    # Mo phong build_model khi tat dark_focus: opt.dark_focus = False
    for mod in model.modules():
        if hasattr(mod, 'dark_focus'):
            mod.dark_focus = False
    
    assert model.dark_focus is False
    for name, mod in model.named_modules():
        if isinstance(mod, IG_Mamba):
            assert mod.dark_focus is False, f"Module {name} chua duoc cap nhat sang False"
    print("  -> [PASS] CIDNet Separable dong bo tat dark_focus cho toan bo 6 khoi IG_Mamba.")

    # Bat lai dark_focus: opt.dark_focus = True
    for mod in model.modules():
        if hasattr(mod, 'dark_focus'):
            mod.dark_focus = True
    assert model.dark_focus is True
    for name, mod in model.named_modules():
        if isinstance(mod, IG_Mamba):
            assert mod.dark_focus is True, f"Module {name} chua duoc cap nhat sang True"
    print("  -> [PASS] CIDNet Separable dong bo bat dark_focus cho toan bo 6 khoi IG_Mamba.")

    # 2c. CIDNet Edge
    model_edge = CIDNet_Edge(channels=[16, 16, 32, 64], heads=[1, 2, 4, 8])
    assert model_edge.dark_focus is True
    for mod in model_edge.modules():
        if hasattr(mod, 'dark_focus'):
            mod.dark_focus = False
    assert model_edge.dark_focus is False
    print("  -> [PASS] CIDNet Separable + Edge dong bo dark_focus chinh xac.")

def test_loss_sync():
    print("\n" + "=" * 60)
    print("3. KIEM TRA DONG BO VA HOAT DONG CUA LOSS LSGD THEO DARK_FOCUS")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tao anh ground-truth: nua trai Toi (0.05), nua phai Sang (0.95)
    gt = torch.zeros(1, 3, 32, 32, device=device)
    gt[:, :, :, :16] = 0.05   # Vung toi
    gt[:, :, :, 16:] = 0.95   # Vung sang

    # Case A: Sai sot o vung toi (pred sai tai nua trai)
    pred_dark_err = gt.clone()
    pred_dark_err[:, :, :, :16] = gt[:, :, :, :16] + 0.3 * torch.randn_like(gt[:, :, :, :16])

    # Case B: Sai sot o vung sang (pred sai tai nua phai voi cung bien do)
    pred_bright_err = gt.clone()
    pred_bright_err[:, :, :, 16:] = gt[:, :, :, 16:] + 0.3 * torch.randn_like(gt[:, :, :, 16:])

    # 3a. BAT Dark Focus (dark_focus=True)
    loss_fn_dark = RegionLSGDLoss(loss_weight=1.0, dark_focus=True).to(device)
    loss_dark_err = loss_fn_dark(pred_dark_err, gt).item()
    loss_bright_err = loss_fn_dark(pred_bright_err, gt).item()
    print(f"  [dark_focus = True]")
    print(f"    -> Loss khi sai o vung toi:  {loss_dark_err:.6f}")
    print(f"    -> Loss khi sai o vung sang: {loss_bright_err:.6f}")
    assert loss_dark_err > loss_bright_err, "Loi: Khi dark_focus=True, loss vung toi phai lon hon vung sang!"
    print("    -> [PASS] Phat nang hon o vung toi!")

    # 3b. TAT Dark Focus (dark_focus=False) -> chuyen sang Bright Focus
    loss_fn_bright = RegionLSGDLoss(loss_weight=1.0, dark_focus=False).to(device)
    loss_dark_err_b = loss_fn_bright(pred_dark_err, gt).item()
    loss_bright_err_b = loss_fn_bright(pred_bright_err, gt).item()
    print(f"  [dark_focus = False]")
    print(f"    -> Loss khi sai o vung toi:  {loss_dark_err_b:.6f}")
    print(f"    -> Loss khi sai o vung sang: {loss_bright_err_b:.6f}")
    assert loss_bright_err_b > loss_dark_err_b, "Loi: Khi dark_focus=False, loss vung sang phai lon hon vung toi!"
    print("    -> [PASS] Phat nang hon o vung sang!")

def test_full_pipeline():
    print("\n" + "=" * 60)
    print("4. KIEM TRA TOAN DIEN KICH BAN BAT / TAT QUA CLI OPTION")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for flag in [True, False]:
        print(f"\n--- Thu nghiem voi CLI option: --dark_focus {flag} ---")
        opt = get_option().parse_args(['--dark_focus', str(flag)])
        
        # 1. Model init & module sync (nhu trong build_model)
        model = CIDNet_Separable(channels=[16, 16, 32, 64], heads=[1, 2, 4, 8]).to(device)
        for m in model.modules():
            if hasattr(m, 'dark_focus'):
                m.dark_focus = opt.dark_focus
        
        # 2. Loss init (nhu trong init_loss)
        loss_fn = RegionLSGDLoss(loss_weight=1.0, dark_focus=opt.dark_focus).to(device)

        # Kiem tra trang thai
        assert model.dark_focus == flag, f"Model dark_focus ({model.dark_focus}) khong khop voi opt.dark_focus ({flag})"
        assert loss_fn.dark_focus == flag, f"Loss dark_focus ({loss_fn.dark_focus}) khong khop voi opt.dark_focus ({flag})"
        print(f"  -> Model dark_focus = {model.dark_focus}")
        print(f"  -> Loss LSGD dark_focus = {loss_fn.dark_focus}")
        print(f"  -> [SUCCESS] Ca IG_Mamba va Loss LSGD deu dong bo trang thai: {flag}")

if __name__ == '__main__':
    test_options()
    test_model_sync()
    test_loss_sync()
    test_full_pipeline()
    print("\n" + "=" * 60)
    print("TAT CA CAC BAI KIEM TRA DONG BO DA THANH CONG 100%!")
    print("=" * 60)
