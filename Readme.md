# Low-Light-Image-Enhancement
+ Install Conda enviroment for CUDA >= 13:
    - conda create -n mamba python=3.10 -y
    - conda activate mamba
    - pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
    - pip install mamba_ssm-2.3.2.post1+cu13torch2.9cxx11abiTRUE-cp310-cp310-linux_x86_64.whl (install that wheel first and run this command)
    - pip install opencv-python lpips thop

+ Install Conda enviroment for CUDA < 13:
    - conda create -n mamba23 python=3.10 -y
    - conda activate mamba23
    - pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
    - pip install ./mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-cp310-cp310-linux_x86_64.whl (install that wheel first and run this command)
    - pip install opencv-python lpips thop

+ Training command:
    - Single GPU: python train.py (have to cd to folder which contain train.py)
    - Mutiple GPU: python -m torch.distributed.run --nproc_per_node=2 train_ddp.py (have to cd to folder which contain train.py)