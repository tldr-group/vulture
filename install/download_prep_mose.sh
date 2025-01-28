#!/usr/bin/env bash
mkdir data
mkdir data/mose
# to get link, navigate to file on drive folder & get sharing link
gdown https://drive.google.com/file/d/16Ns7a_frLaCo2ug18UIUkzVYFQqyd4N0/view?usp=drive_link -O data/mose.tar.gz --fuzzy
tar -xzf data/mose.tar.gz -C data/mose
rm -rf data/mose/train/Annotations

rm -f data/mose.tar.gz

gdown https://drive.google.com/file/d/1yFoacQ0i3J5q6LmnTVVNTTgGocuPB_hR/view?usp=drive_link -O data/mose_val.tar.gz --fuzzy
tar -xzf data/mose_val.tar.gz -C data/mose
mv data/mose/valid data/mose/val
rm -rf data/lvos/val/Annotations
rm -rf data/lvos/val/Scribbles
rm -f data/lvos/val/train_meta.json data/lvos/val/train_expression_meta.json
rm -f data/lvos_val.zip