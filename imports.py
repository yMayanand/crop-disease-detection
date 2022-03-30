#=============================#
#common imports               #
#=============================#
import zipfile
import requests
import cv2
import matplotlib.pyplot as plt
import re
import numpy as np
import math
import gradio as gr
# use this only on google colab
#from google.colab import files 
from glob import glob
import os

#=============================#
#pytorch imports              #
#=============================#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, datasets, transforms

#=============================#
#pytorch-lightning imports    #
#=============================#

import pytorch_lightning as pl
import torchmetrics as tm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback