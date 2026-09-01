from torch import no_grad
from PIL import Image

from vulture import CompleteUpsampler
from vulture.utils import vis

# Data source:
# 'Cast iron with magnesium induced spheroidised graphite', Cambridge DoITPoMS, Dr R F Cochrane
# https://www.doitpoms.ac.uk/miclib/micrograph_record.php?id=394
path = "data/apply/default_image.jpg"
img = Image.open(path).convert("RGB")

# upsampler = CompleteUpsampler("FEATUP", "trained_models/fit_reg_f32.pth", device="cuda:0", to_half=True, to_eval=True)
upsampler = CompleteUpsampler(
    "ALIBI_COMPRESSED",
    "trained_models/alibi_f24_slow_e5000.pth",
    autoencoder_chk_or_cfg="trained_models/dac_alibi_dv2_e500_c24.pth",
    dino_chk="trained_models/alibi_dv2_vits14_reg.pth",
    device="cuda:0",
    to_half=False,
    to_eval=True,
)
with no_grad():
    lr_feats = upsampler.get_lr_feats(img)
    hr_feats = upsampler.forward(img)
print(hr_feats.shape)

vis("tmp/test.png", img, lr_feats, hr_feats, None, False)
