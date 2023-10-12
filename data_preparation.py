import os
from utils.population_dataset import generate_to_csvfiles
if __name__ =='__main__':
    root = "E:/git_tor/skin_cancer_x400"
    copy_from = root + '/Images'
    print("copy_ from: ", copy_from)
    generate_to_csvfiles(copy_from)