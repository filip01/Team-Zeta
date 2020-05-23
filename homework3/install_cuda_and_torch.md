# Pytorch setup

Install nvidia driver.
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

sudo apt-get install --no-install-recommends nvidia-driver-440
```

Now you need to restart your system.

Install pytorch.
```
virtualenv --no-site-packages -p python3 ros3env
. ./ros3env/bin/actiave
pip install --extra-index-url https://rospypi.github.io/simple/ rospy
pip install torch torchvision
```

Test installation in REPL.

```python
import torch
torch.cuda.is_available()
```
```
True
```