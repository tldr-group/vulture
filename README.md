# threetures

## Model:

1) learned conv downsampler that takes image and iteratively downsamples, increasing channel depth. 
    - (H, W, 3)
    - (H / 2, W / 2, 8)
    - (H / 4, W / 4, 8)
    - (H / 8, W / 8, 16)
    - (H / 16, W / 16, 16)
    - Resize to (H / 14, W / 14, 16)
2) learned conv upsampler that goes from lr features -> hr features with image guidance from each stage of the downsampler:
    - (H / 14, W / 14, 128) + (H / 14, W / 14, 16)
    - (H / 8, W/ 8, 128) + (H / 8, W / 8, 16)
    - (H / 4, W/ 4, 128) + (H / 4, W / 4, 8)
    - (H / 2, W/ 2, 128) + (H / 2, W / 2, 8)
    - (H, W, 128) + (H, W, 8)


## Dataset:

Featup implict can generate high-res features, but increasing resolution above (224, 224) is slow - may as well use a (224, 224) image dataset. [ImageNet reduced](https://huggingface.co/datasets/richwardle/reduced-imagenet) consists of 26,000 (224, 224) images of 26 examples of the 1,000 classes in ImageNet. I've taken 2 from each class for a total of 2,000 images, saved in 10 different splits in [`splits/`](data/imagenet_reduced/splits/).

By default FeatUp compresses to 128 channels - we'll keep that as it a) speeds things up and b) reduces storage space taken up by features. We just use their default config, except changing the input dir and adding the option in [`featup/train_implict_upsampler.py`](FeatUp/featup/train_implicit_upsampler.py). NB: if we want to generate embeddings for higher image size (square) data (i.e, > 224^2), need to adjust `final_size` to be larger.






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