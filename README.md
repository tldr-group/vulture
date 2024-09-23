# threetures


## Datasets:

[MOSE](https://henghuiding.github.io/MOSE/): 'MOSE focuses on complex scenes, including the disappearance-reappearance of objects, inconspicuous small objects, heavy occlusions, crowded environments, etc.'

[LVOS](https://lingyihongfd.github.io/lvos.github.io/): 'LVOS focuses on long-term videos, with complex object motion and long-term reappearance.'

from [LSVOS](https://lsvos.github.io/)

## Installation:

```
export CUDA_HOME=/home/ronan/miniconda3/envs/dv2/bin/nvcc
~/miniconda3/envs/dv2/bin/python -m pip install
pip install wheel setuptools
git clone https://github.com/mhamilton723/FeatUp.git
cd FeatUp
pip install -e .
```

# NB: need hydra-core, tensorboard, xformers, triton
# NB: dv2 needs xformers needs triton needs cuda12 - maybe try install that in cuda environment
# try install dv2 locally then featup
# 2:51 min on 3000x3000 img for implict for 1200
# not much difference for 1000x100 
# Try test acc vs N_iter to see if can decrease it
# -> 240 hours for 5000 embeddings?

# seems to take 6GB of memory per featup instance: i have two ~64GB graphics cards here.
# can run 5 * 2 = 10 threads of featup at once. 
# run on dataset of 2/10k 1024x1024 natural images
# https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images
# seems promising - most images seem to be 2040x

# around 200/300/400 upsampler seems to overfit on micrographs and have periodic effects - might not be a problem if we're only training on natural images

# featup-I resizes to 224,224 - is it worth running at higher resolution or not?