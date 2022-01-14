# xREgoPose
Pytorch Implementation of xREgoPose

How to setup:

1. Setup Virtualenv on your home directory.
```
cd ~/
module load python/3.9
virtualenv --no-download torch
source torch/bin/activate
pip install --no-index --upgrade pip
```
2. Install required dependencies for the repo 
```
# First cd into the xREgoPose directory 
# For example, I will have to " cd ~/projects/def-pfieguth/j97park/xREgoPose/xREgoPose/
pip install -r requirements.txt --no-dependencies
```


Additional Notes:
1. You will need to update line 92 of utils/config.py so that the directory of config.yml fits to your local repository. For example, for me I'll need to change it to '/home/j97park/projects/def-pfieguth/j97park/xREgoPose/xREgoPose/data/config.yml'.
