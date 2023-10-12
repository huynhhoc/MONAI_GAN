import pandas as pd
import numpy as np
import os
import cv2
if __name__ == '__main__':
    path = os.getcwd()
    savepath = './check_datagen'
    if not os.path.exists(savepath):
        os.mkdir(savepath)    
    datatest_path = path
    df = pd.read_csv(datatest_path)
