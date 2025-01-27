#!/usr/bin/env bash
mkdir data
gdown https://drive.google.com/file/d/1-ehpl5s0Fd14WwtT-GmWtIWa_BxZl9D6/view?usp=share_link -O data/lvos.zip --fuzzy
unzip data/lvos.zip -d data/lvos # unzips to data/lvos/train
rm -rf data/lvos/train/Annotations
rm -f data/lvos/train/train_meta.json data/lvos/train/train_expression_meta.json
rm -f data/lvos.zip

gdown https://drive.google.com/file/d/17Hwc__6i2rpF5e2s5OPqoywNxG5bzlcO/view?usp=share_link -O data/lvos_val.zip --fuzzy
unzip data/lvos_val.zip -d data/lvos # unzips to data/lvos/val
mv data/lvos/valid data/lvos/val
rm -rf data/lvos/val/Annotations
rm -rf data/lvos/val/Scribbles
rm -f data/lvos/val/train_meta.json data/lvos/val/train_expression_meta.json
rm -f data/lvos_val.zip