'''
Example script for converting .mat files to .npz files
Illustrate on the bilinear parabolic problem dataset
'''

import scipy.io as sio
import numpy as np
from utils.utils_fno import MatReader

TRAIN_PATH = ('./data/bilinparab/bp_cts_gradadj_train_u.mat', 
              './data/bilinparab/bp_cts_gradadj_train_f.mat', 
              './data/bilinparab/bp_cts_gradadj_train_y.mat')
TEST_PATH = ('./data/bilinparab/bp_cts_gradadj_test_u.mat', 
             './data/bilinparab/bp_cts_gradadj_test_f.mat', 
             './data/bilinparab/bp_cts_gradadj_test_y.mat')

reader = MatReader(TRAIN_PATH[0])
train_u = reader.read_field('u')
reader = MatReader(TRAIN_PATH[1])
train_f = reader.read_field('f')
reader = MatReader(TRAIN_PATH[2])
train_s = reader.read_field('s')
np.savez('bp_cts_gradadj_train.npz', u=train_u, f=train_f, s=train_s)

reader = MatReader(TEST_PATH[0])
test_u = reader.read_field('u')
reader = MatReader(TEST_PATH[1])
test_f = reader.read_field('f')
reader = MatReader(TEST_PATH[2])
test_s = reader.read_field('s')
np.savez('bp_cts_gradadj_test.npz', u=test_u, f=test_f, s=test_s)