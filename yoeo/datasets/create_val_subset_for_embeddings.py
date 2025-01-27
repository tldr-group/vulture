from random import shuffle
from os import listdir
from shutil import move

N_VAL = 150
input_dir = "data/imagenet_reduced/data_reg"
target_dir = "data/imagenet_reduced/val_reg"
if __name__ == "__main__":
    all_data = listdir(input_dir)
    # sample without replacement
    inds = [i for i in range(len(all_data))]
    shuffle(inds)
    file_inds_to_move = inds[:N_VAL]

    for ind in file_inds_to_move:
        src = f"{input_dir}/{all_data[ind]}"
        dst = f"{target_dir}/{all_data[ind]}"
        move(src, dst)
