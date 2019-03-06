from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import time
from datetime import date
import _pickle as cPickle
from six.moves import urllib
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def _norm(img):
    return (img - np.mean(img)) / np.std(img)

def main():
    

