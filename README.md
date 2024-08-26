# threetures


## Datasets:

[MOSE](https://henghuiding.github.io/MOSE/): 'MOSE focuses on complex scenes, including the disappearance-reappearance of objects, inconspicuous small objects, heavy occlusions, crowded environments, etc.'

[LVOS](https://lingyihongfd.github.io/lvos.github.io/): 'LVOS focuses on long-term videos, with complex object motion and long-term reappearance.'

from [LSVOS](https://lsvos.github.io/)

## Installation:

```
python3.12 -m venv .venv
source .venv/bin/activate
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
