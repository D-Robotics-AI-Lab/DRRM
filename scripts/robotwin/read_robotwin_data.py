import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import einops
import cv2

def read_robotwin_data(load_dir, current_ep, file_num):
    with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, default="data/dual_bottles_pick_easy_D435_pkl")
    parser.add_argument("--current_ep", type=int, default=0)
    parser.add_argument("--file_num", type=int, default=0)
    args = parser.parse_args()

    data = read_robotwin_data(args.load_dir, args.current_ep, args.file_num)
    print(data['observation']['head_camera']['rgb'].shape)
    # print(data)