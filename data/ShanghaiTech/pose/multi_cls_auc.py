import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset import shanghaitech_hr_skip
import joblib
import json
import cv2
import copy
###############################################################################################################################################
##################################### Part 1: combine human with other classes for state machine ##############################################
###############################################################################################################################################

