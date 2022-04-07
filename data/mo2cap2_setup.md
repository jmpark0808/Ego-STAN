# Downloading and Extracting Mo2Cap2

Start off by downloading the [mo2cap2](http://vcai.mpi-inf.mpg.de/projects/wxu/Mo2Cap2/) dataset and unzip the training and test sets. <br>
Once the training and testing zip files are unzipped, change the names to `TrainSet` and `TestSet`. <br>
Make a directory for `ValSet` as well and copy some of the chunks from TrainSet to it (if needed). <br>
Use the `utils/extract_mo2cap2.py` script to write png and json files. -> Compress them individually -> Download the rest later. <br>
Start the raining and testing pipeline. <br>

# Using Mo2Cap2

The use mo2cap2 would be almost identical to xREgoPose. The correct directory needs to be specified in `train.py`. <br>
The correct relevant argument is the `--dataloader` argument where `mo2cap2` should be passed. <br>
The folders where TrainSet, ValSet and TestSet were zipped/compressed to need to be unzipped to `$SLURM_TMPDIR`. <br>
(TO DO: upload the bash scripts to do the appropriate operations)
