<h1> Setup </h1>

Pytorch Lightning Implementation of xREgoPose

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
# Like " cd ~/path/to/xREgoPose/
pip install -r requirements.txt
```
<h1> Navigation </h1>

* train.py
    * Main wrapper for training
    * Options for training can be found inside the script
    * Testing can be done after training is finished. Check the arguments. 
* eval.py
    * Given a trained checkpoint, run the evaluation only.
* net/
    * Folder that contains all the different Pytorch Lightning classes of different methods. 
* visualizations/
    * Folder that contains the visualizations included in the main paper. 
* results/
    * Folder that contains the csv files of results.
* dataset/
    * Folder that contains the codes for processing data. 
* data/
    * Folder that contains information about mo2cap2 (in progress) and a config file for xREgoPose. 
* base/
    * Folder that contains base class for dataset, eval and transform. 
* utils/
    * Folder that contains codes for utility. (Logging, metrics, configuration of joints etc.)
* sbash_scripts/
    * Example scripts to run on compute canada.


     
