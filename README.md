# visual_pomdp_smm

Uses Gym-Minigrid (https://github.com/maximecb/gym-minigrid)
## Requirements

```
git clone https://github.com/BurakDmb/pomdp_tmaze_baselines.git
cd pomdp_tmaze_baselines

conda create -n pomdp python=3.8 -y
conda activate pomdp
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge gym scikit-learn profilehooks progressbar matplotlib tensorboard numpy pandas cloudpickle optuna mysqlclient mysql-client plotly flake8 -y
conda install pip -y
pip install tensorboard-reducer --no-dependencies --trusted-host pypi.org --trusted-host files.pythonhosted.org
pip install git+https://github.com/DLR-RM/stable-baselines3 --no-dependencies --trusted-host pypi.org --trusted-host files.pythonhosted.org
pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib --no-dependencies --trusted-host pypi.org --trusted-host files.pythonhosted.org
pip install gym-minigrid torchsummary --no-dependencies  --trusted-host pypi.org --trusted-host files.pythonhosted.org
python setup.py develop

cd ..
git clone https://github.com/BurakDmb/visual_pomdp_smm.git
cd visual_pomdp_smm
python setup.py develop
```