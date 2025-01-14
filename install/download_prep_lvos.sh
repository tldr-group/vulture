#!/usr/bin/env bash
mkdir data
python -m gdown https://drive.google.com/file/d/1-ehpl5s0Fd14WwtT-GmWtIWa_BxZl9D6/view?usp=share_link -o data/lvos.zip
unzip data/lvos.zip -d data/lvos # unzips to data/lvos/train
rm -rf data/lvos/train/Annotations
rm -f data/lvos/train/train_meta.json data/lvos/train/train_expression_meta.json
rm -f data/lvos.zip