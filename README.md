# visual_pomdp_smm

Uses Gym-Minigrid (https://github.com/maximecb/gym-minigrid)

# Requirements with ray framework.

```
git clone https://github.com/BurakDmb/pomdp_tmaze_baselines.git
cd pomdp_tmaze_baselines

conda create -n pomdp python=3.10 -y
conda activate pomdp
mamba install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# Note: Currently, ray framework is installed from the daily releases. In the future, it is recommended to download from the stable releases.
pip install "ray[all] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"

pip install scikit-learn profilehooks progressbar matplotlib tensorboard plotly flake8 tqdm minigrid tensorboard-reducer torchinfo pyarrow
python setup.py develop

cd ..
git clone https://github.com/BurakDmb/visual_pomdp_smm.git
cd visual_pomdp_smm
python setup.py develop
```