# visual_pomdp_smm

Uses Gym-Minigrid (https://github.com/maximecb/gym-minigrid)

# Requirements with ray framework.

```

git clone https://github.com/BurakDmb/visual_pomdp_smm.git
cd visual_pomdp_smm

conda create -n pomdp python=3.10 -y
conda activate pomdp
mamba install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# Note: Currently, ray framework is installed from the daily releases. In the future, it is recommended to download from the stable releases.
pip install ray[all]==2.4.0 scikit-learn profilehooks progressbar matplotlib tensorboard plotly flake8 tqdm minigrid tensorboard-reducer torchinfo pyarrow natsort gymnasium[other]

python setup.py develop
```