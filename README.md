> **Hierarchical decision transformer**\
> André Correia, Luís A. Alexandre\
> Paper: https://ieeexplore.ieee.org/abstract/document/10342230/


# Install Anaconda
```
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

bash Anaconda3-2024.02-1-Linux-x86_64.sh

conda init
```
# Create and activate environment
```
conda create --name HDT

conda activate HDT
```
# Install Dependencies
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip3 install numpy

pip3 install transformers

pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```
# Train Decision Transformer
```
python3 train_dt.py --env_name 'hopper' --dataset medium-replay --batch_size 64 --d_model 128 --n_layer 3 --n_head 1 --K 20 --iterations 10000 --eval_every 1000
```
# Train Hierarchical Decision Transformer
```
python3 train_hdt.py --env_name 'hopper' --dataset medium-replay --batch_size 64 --high_d_model 128 --high_n_layer 3 --high_n_head 1 --low_d_model 128 --low_n_layer 3 --low_n_head 1 --K 20 --iterations 10000 --eval_every 1000
```

## Citation

If you use this codebase, or otherwise found our work valuable, please cite HDT:
```
@inproceedings{correia2023hierarchical,
  title={Hierarchical decision transformer},
  author={Correia, Andr{\'e} and Alexandre, Lu{\'\i}s A},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1661--1666},
  year={2023},
  organization={IEEE}
}
```
