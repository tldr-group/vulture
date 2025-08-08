from torch import no_grad
from PIL import Image

from vulture import CompleteUpsampler
from vulture.utils import vis

# Data source:
# 'Cast iron with magnesium induced spheroidised graphite', Cambridge DoITPoMS, Dr R F Cochrane
# https://www.doitpoms.ac.uk/miclib/micrograph_record.php?id=394
path = "examples/example_data/cast_iron_alloy.jpg"
img = Image.open(path).convert("RGB")

upsampler = CompleteUpsampler("FEATUP", "trained_models/fit_reg_f32.pth", device="cuda:0", to_half=True, to_eval=True)
with no_grad():
    lr_feats = upsampler.get_lr_feats(img)
    hr_feats = upsampler.forward(img)
print(hr_feats.shape)

vis("tmp/test.png", img, lr_feats, hr_feats, None, True)
