import os
import random
import pandas as pd
import shutil
from utils.parameters import load_class_weights, load_sub_class_weights, GROUPS_MALIGNANT_SKIN_CANCER, GROUPS_CLASS_LABEL
#------------------------------------------------------------------------------------------
# remove/clean all special characters, punctuation, ASCII characters and spaces from the string
def removeSpecialChacters(line):
  item = ''.join(e for e in line if (e.isalnum() or e.isspace()))
  item = item.replace(' ','_')
  return item
#------------------------------------------------------------------------------------------
def get_all_files_subfolders(folder: str) -> list:
    listOfFiles = list()
    print("folder: ", folder)
    for (dirpath, _, filenames) in os.walk(folder):
        nclass = dirpath.split('/')[-1]
        for file in filenames:
                listOfFiles.append([nclass,os.path.join(dirpath, file)])
    return listOfFiles
#--------------------------------------------------------------------------------------------
def set_label_based_on_subclass(sclass):
    return sclass
#--------------------------------------------------------------------------------------------
def set_label_based_on_groups(sclass, subclasses):
    #malignant, benign
    label = "benign"
    if sclass.upper() in subclasses:
        label = "malignant"
    return label
#-------------------------------------------------------------------------------------------
def number_of_images_per_5fold(number_of_images):
    lst_images_per_fold = {}
    twelve_point_five = int(number_of_images*0.10)
    for fold in [2,3,4,5]:
        lst_images_per_fold[fold] = twelve_point_five
    lst_images_per_fold[1] = number_of_images - 4*twelve_point_five
    return lst_images_per_fold
#-------------------------------------------------------------------------------------------
def generate_data_with_kfold_for_subclass(copy_from):

    listOfFiles = get_all_files_subfolders(copy_from)
    random.shuffle(listOfFiles)  #shuffles a list in place

    number_images_per_class = len(listOfFiles)
    print("listOfFiles: ", number_images_per_class)
    lst_images_per_fold = number_of_images_per_5fold(number_images_per_class)
    kfold = 1
    ncount = 1
    items = []
    no =-1
    for sclass, pathfile in listOfFiles:
        try:
            no = no + 1
            if ncount > lst_images_per_fold[kfold]: #number_images_per_kfold:
                kfold +=1
                ncount = 1
            filename = pathfile.split("/")[-1]
            
            index_last_slash = pathfile.rfind("/")
            sub_folder = pathfile[: index_last_slash]
            new_filename = removeSpecialChacters(filename[:-4]) + ".jpg"
            new_path = os.path.join(sub_folder,new_filename)
            os.rename(pathfile,new_path)
            label = set_label_based_on_groups(sclass, GROUPS_MALIGNANT_SKIN_CANCER)
            #if not os.path.exists(paste_to + "/" + label):
            #    os.mkdir(paste_to + "/" + label)
            items.append([new_filename,pathfile,sub_folder, label, sclass, kfold])
            print("item: ", new_filename, pathfile)
            ncount +=1
        except Exception as ex:
            print ("error: ", ex)
            pass
    return items
#--------------------------------------------------------------------------------------------
def generate_to_csvfiles(copy_from):
    items_train = generate_data_with_kfold_for_subclass(copy_from + "/train")
    df = pd.DataFrame(data= items_train, columns=['id', 'image', 'path', 'label', 'slabel', 'kfold'])
    df.to_csv("dataset/dataset_train.csv")
    # test
    items_test = generate_data_with_kfold_for_subclass(copy_from + "/test")
    df = pd.DataFrame(data= items_test, columns=['id', 'image', 'path', 'label', 'slabel', 'kfold'])
    df.to_csv("dataset/dataset_test.csv")
    nclass = len(next(os.walk(copy_from + "/train"))[1])
    return nclass
#--------------------------------------------------------------------------------------------
def balance_data_slabel(df, slabel, classweight):
    temp_df = []
    for row in df.itertuples(index=False):
        if row.slabel == slabel:
            temp_df.extend([list(row)]*classweight)
        else:
            temp_df.append(list(row))

    df = pd.DataFrame(temp_df, columns=df.columns)
    return df
#--------------------------------------------------------------------------------------------
def balance_data_label(df, label, classweight):
    temp_df = []
    for row in df.itertuples(index=False):
        if row.label == label:
            temp_df.extend([list(row)]*classweight)
        else:
            temp_df.append(list(row))

    df = pd.DataFrame(temp_df, columns=df.columns)
    return df
#--------------------------------------------------------------------------------------------
def balance_data_based_class_weight(df, c_label, sc_label):

    sub_class_weights =load_sub_class_weights()
    sub_class_labels = list(set(df[sc_label]))
    df2 = df.groupby([sc_label])[sc_label].count()
    print("Level1: before balancing: ", df2)
    
    for sub_class_label in sub_class_labels:
        df = balance_data_slabel(df, sub_class_label,sub_class_weights[sub_class_label])
    
    df2 = df.groupby([sc_label])[sc_label].count()
    print("Level 1: after balancing: ", df2)

    # balance for two class:
    class_weights = load_class_weights()
    class_labels = list(set(df[c_label]))

    df2 = df.groupby([c_label])[c_label].count()
    print("Level 2: before balancing: ", df2)

    for class_label in class_labels:
        df = balance_data_label(df, class_label, class_weights[class_label])
    
    df2 = df.groupby([c_label])[c_label].count()
    print("Level 2: after balancing: ", df2)
    
    return df
