# threetures

## Notes 26/09/24;
- most models bottom out around 15-17k loss (L2 or smooth L1)
- can get high res but poor semantics or better semantics and blurring
- simple strategy of resize lr feats -> img size then pass through many convs w/ implict feats not working
- could try adding guidance from lr feats at each upsample (multiply by resized lr feats) to retain semantics?
    - theory is this stops the network needing to learn it
- could try larger kernels (11, 21 etc)
    - theory is that semantics not being properly transmitted due to small receptive field
- could try simple strategy:
    - cat (lr feats from prev layer * resized lr feats, impl feats of resized img) -> 128 dims 
    - i.e no learned downsampling
    - reintroduces semantic info
- what network size would help?
    - would a small network force it to learn general features?
    - would a large network actually be able to learn the correct output?
- should you normalize output as well?
- above simple strategy seems to be working well but:
    - get nasty patching artefacts
        - could try having more convs after upsampling (w/out lr feat guidance) N
        - could try weighting the features less as res intreaces X
        - could try doing (weighted) mean of feats not the product X
        - could try weihted sum of feats X
        - could try using non conv upsampling (and use bilinear instead)
    - sums/ means not working as well as multiply for some reason
    - adding weights to either product or means worse (initially, though evens out)
    - adding in the lr feats does mess it up early: large blobs in centre of image
    - adding small pixel shifts to HR feats and image seems to be working okay in ameliorating artefacts
        - logic is that it puts lr feats and hr out of step, forces model not to over-rely on LR feats
        - still not perfect
    - an even more simple approach (i.e single convs, not implict feats) is desirable: is it better?
    - multi-GPU training to speed stuff up?

## Notes 27/09/24:
- model struggles if not adding semantics
- what I want is to break the 10k loss for smooth L1 with 20 validation images
- learned downsample has some nice properties: will pxiel shift trs help it?
    - the loss does drop very quickly between epochs
    - sharp resolution in some areas, but lots of blurring otherwise
    - will adding feature skips help or hindr?
    
## Notes 29/09/24:
- bumped nch for learned downsampler
- do I need to add feature guidance in the upsampler - or will that cause the patching problem?
- definite problems with smooth background images: seeing the DINOv2 anomalous bg token effect. Fix in future by moving to DINOv2_reg for featurizer and recompute the features (expensive). Maybe starting with implict model pretrained on old features will help?
- promising: learned down + img guidance + feature guidance (first two upsamples, weight=0.25) + 32 batch + k=3 (might increase in future) + lr=1e-3
- worry: small k could work for small input image sizes as global information can be passed but at large image size (i.e materials images) this could not be communicated. 
    - Larger k + multi-resolution training should be considered in future.
    - FeatUP seems to think you can get features out at any resolution from a trained implict upsampler by 'querying the pixel field' - this could come in handy
- maybe the pixel shifting transforms aren't good and encourage blurring
- is sharpness the last step i.e it only offers incremental improvements in the loss so won't happen till the end
    - will it happen at all if the model gets stuck in discrepenacy between the featup feats and the dv2 lr feats
    - or the featup feats and the conv features it's able to reproduce?
- seems like this approach is working!
    - parameters:
    - learned downsampler, n_ch=65 (+3 for img)
    - kernel size=3
    - resize image guidance in downsampler, no implict features
    - (multiplicative) feature guidance in first half of the layers in the upsampler, with feature weight=0.25
    - LR=1e-3, batch size 32
    - N_epochs=5000
    - good results after 230 ish epochs!!
- problem - the lr feats I use as my training input (from featup) is actually a pca over a series of features of 50 transforms of the image (jitters/zooms/etc)
    - model now trained to use those (and not the dv2 features of a single image)
    - can hopefully be trained to just use one set of dv2 features - might improve perf, but might not as those transforms contain some info
    - will have to go through all splits and compute proper dv2 features for them
    - opens up possiblity to have contractive model (i.e one that goes from 384 -> 128 dims over the layers) but that means can't do feature guidance in second layer as easily
        - does that mean get rid of feature guidance? idk how much it does as only in first two layers and in the
- tomorrow:
    - get vanilla dv2 embeddings for all train & val data, dim 384 (and 128?)
    - parametrise everything with a json file
    - torch seed stuff
    - get stuff working for images that aren't multiples of 224 - test w/ rectangles as well
    - try train a contractive net:
        - w/out feature guidance
        - w/out pixel shifts in dataset (cause of blurring?)

## Notes 30/09/24:
- got  vanilla dv2 embeddings, fixed image for non 224 multiples
- seems tough to do a contractive model with dv2 features w/out feature guidance in early layers
    - val loss seems flat after after 120 epochs, train loss decreasing smoothly
        - hoping for a step change after 200 epochs
    - unsure if this is the contractiveness, or using the proper dv2 features or the lack of early guidance
        - contractivenss = harder problem
        - using dv2 lr features (and not featup lr features computed over batch of jitters) = harder problem - now a distribution drift as well
        - lack of feature guidance: harder problem, now also needs to focus more on semantics
        - could also maybe be a lack of pixel shifts? see that as unlikely though
    - might be worth trying with lower lr?
    - theoretically the resolution comes later in training once it's sorted the main bits, though worry is it's not really sorting the main bits
    - could try not doing contractive and just going from dv2 128-dim pca
    - could add another loss term that compares downsample to dv2 feats - although wrong n dim
    - if it still sucks after 300 try with lower lr
    - it seems like the previous approach w/ feat guidance and wrong feats learned semantics first then resolution
    - what i see with no feat guidance and dv2 feats is kind of the other way round: get good res but features 'wrong colour' and am now seeing a gradual phase shift in feature color in my val set
    - this mostly wasn't needed in the original bc the dist of the lr feats and hr feats seemed similar
    - just had big spike in loss oh dear
    - @epoch 200 seems like the val loss just fluctuates between 40-50k, despite the train loss decreasing. 
        - might try the non-contractive model on pca 128 dinov2 data
    - @ epoch 221: massive loss spike - lol, lmao even
        - [220/5000]: train=938440.421875, val=51108.9609375
        - [221/5000]: train=2697649.017578125, val=5022666.0
        - [222/5000]: train=2961435.69921875, val=46635.546875
        - ????????
        - [223/5000]: train=2099878.984375, val=41993.75390625
- is the fact that the input and output are now drawn from different dists actually quite a big problem? now needs to learn whatever mapping was done between the lr dv2 feats and the lr featup feats which are then upsampled. note that the featup feats involve different view of the features so has more info
    - could try training a style transfer net from lr dv2 feats to lr featup feats: very small - two/three double convs?
    - it really does seem like it's struggling between learning generic feature upsampling and learning to remap the feature spaces (or at least struggling to balance them)
    - now means downsampler will have to learn some remapping I suppose (as well as upsampler ofc) - which isn't what we want
- literally all of them have a nasty peak around 200 when trained with lr=1e-3. best run so far could recover, other didn't

## Notes 1/10/24:
- today:
    - test training a feature transfer & contraction net (384 lr dv2 -> 128 featup lr). Does it need image info or will a double conv stack work?
    - if feature trasnfer works, add it as (frozen) component in training and test previous approach (non-contractive, no pixel shuffle, no feature guidance, image guidance) 
- feature transfer seems tougher than expected:
    -  maybe l1 loss can help for sharpness
    - maybe some form of image guidance is required (annoying as expensive)
    - still seems tough to learn the semantic mapping (spatial is fine though)
    - large kernel size + ciruclar padding might work well as can reproduce the padding effect of the jittering
    - NB: should probs still norm image for transfer regardless of if not norming features
    - img guidance in the transfer seems to cause a thinning of features - this might even be desirable were it not for the fact that the upsampling will probably work best with the lr features as they are (i.e slighly blurred) - may have to drop it if want 
    - may need to include all that zoom/pad/flip info into the image, pass it through a learned downsampler and use that to guide the feature transfer
        - in future this could all be part of the combined network? maybe with a loss term between the first levels of the input features and 
    - seems like I can get away with just using 50 images in the lr feature generation - this is acceptable but for large images (2200, 2200) this takes 15s
        - this is because featup learns to upsample via the 3000 images, and takes the first 50 to compute the lr feats for visualisation. why does the pca work then - features should be misaligned - i guess this is why the features are so different
        - does this also point to a fundamental limit to your approach in that you don't have the info


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