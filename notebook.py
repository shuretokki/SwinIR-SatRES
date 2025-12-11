import os

repo_name = "vdronerez-swinir"
repo_url = "https://github.com/shuretokki/vdronerez-swinir.git"

if os.path.exists(repo_name):
    %cd {repo_name}
    !git reset --hard
    !git pull
    %cd ..
else:
    !git clone --filter=blob:none --no-checkout {repo_url} {repo_name}
    %cd {repo_name}
    !git sparse-checkout init --cone
    !git sparse-checkout set --no-cone "/*" "!/web/" "!/api/"
    !git checkout
    %cd ..

!pip install -q -r {repo_name}/requirements.txt

ds = "/kaggle/input/visdrone-dataset/VisDrone_Dataset/VisDrone2019-DET-train/images"
%cd {repo_name}

if os.path.exists(ds):
    if not os.path.exists("data/train_hr"):
        print("Prepping dataset...")
        !python prep.py --source "$ds" --limit 5000

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
!python train.py --epochs 1 --batch_size 32 --patch_size 96
# !python train.py --batch_size 32 --patch_size 96