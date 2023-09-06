conda create -n mem_loc python=3.10
conda activate mem_loc

# install pytorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install ipdb matplotlib lightning-bolts
pip uninstall pytorch-lightning-bolts pytorch_lightning==1.4.9