import matplotlib.pyplot as plt
import numpy as np
import itertools
import re
from matplotlib.pyplot import *

def plot_classification_report(runfolders, classificationReport,name_report, title='Classification report',cmap='RdBu'):
    import matplotlib.pyplot as plt
    classificationReport = classificationReport.replace('nn', 'n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')
    classes, plotMat, support, class_names = [], [], [], []
    f1=[]
    for line in lines[2:-4]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        line = re.sub(' +', ' ', line)
        t = line.strip().split(' ')
        try:
            f1.append(t[3])
        except:
            pass
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()
    plt.savefig(runfolders + '/' + name_report+'.png')
    return f1
#-------------------------------------------------------------------------
def get_classification_report(step, report):
    import re
    global runfolders
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        try:
            row_data = re.sub(' +', ' ', line)
            row_data = row_data.split(' ')
            #print ("row_data: ", row_data)
            if len(row_data) == 6:
                row['name'] = row_data[1]
                row['precision'] = float(row_data[2])
                row['recall'] = float(row_data[3])
                row['f1_score'] = float(row_data[4])
                row['support'] = row_data[5]
                report_data.append([step, row['name'], row['precision'], row['recall'], row['f1_score'], row['support']])
        except:
            pass
    #------------------------
    nline = len(lines)
    for line in lines[nline-4:]:
        #print ("line: ", line)
        row = {}
        try:
            row_data = re.sub(' +', ' ', line).strip()
            row_data = row_data.split(' ')
            #print ("row_data: ", row_data, len(row_data))
            row['name'] = row_data[0] + ' ' + row_data[1]
            row['precision'] = float(row_data[2])
            row['recall'] = float(row_data[3])
            row['f1_score'] = float(row_data[4])
            row['support'] = float(row_data[5])
            report_data.append([step, row['name'], row['precision'], row['recall'], row['f1_score'], row['support']])
        except:
            pass
    return report_data
#------------------------------------------------------
def exportDic2CSV(runfolders, dic):
    with open(runfolders +  '/evaluation.csv', 'w') as f:
        f.write(dic)
#-------------------------------------------------------
def plot_classification(runfolders, df_reports, name = 'cancerx400'):
    e=0
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    fig.set_figheight(8)
    fig.set_figwidth(12)
    gca().cla()
    precisions = df_reports['precision']
    recalls = df_reports['recall']
    f1_scores = df_reports['f1_score']
    ax1.plot(precisions,label='precision')
    ax2.plot(recalls,label='recalls')
    ax3.plot(f1_scores, label='f1_score')
    ax1.tick_params(labelright=True)
    ax2.tick_params(labelright=True)
    ax3.tick_params(labelright=True)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax2.set_xlabel('number of epochs '+ name)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    fig.savefig(runfolders + '/classification_'+ name +'.png')
#-------------------------------------------------------
def plot_report_by_name(runfolders, df_report, yname):
    classnames = df_report['class']
    classnames = np.unique(classnames)
    print("class names: ", type(classnames), classnames)
    for nclass in classnames:
        df_class = df_report[df_report['class']==nclass]
        plot_classification(runfolders, df_class, nclass)
#------------------------------------------------------------------------------------------------
def plothistory(d, runfolder, name = 'cancerx400'):
    e=0
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_figheight(8)
    fig.set_figwidth(12)
    gca().cla()
    ax1.plot(d[1][e:],label='train')
    ax2.plot(d[0][e:],label='val')
    ax1.tick_params(labelright=True)
    ax2.tick_params(labelright=True)
    ax1.legend()
    ax2.legend()
    ax2.set_xlabel('number of epochs '+ name)
    ax1.grid(True)
    ax2.grid(True)
    fig.savefig(runfolder + '/history_'+ name +'.png')
#--------------------------------------------------------------------------------------------
def show_figure_4(d, runfolder,kfold,hist_names, filename= 'hist'):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_figheight(8)
    fig.set_figwidth(12)
    plt.gca().cla()
    
    ax1.plot(d[0],label=hist_names[0] )
    ax1.plot(d[1],label=hist_names[1])
    ax1.tick_params(labelright=True)
    ax1.legend()
    ax1.set_xlabel('loss')
    ax1.grid(True)
    
    ax2.plot(d[2],label=hist_names[2])
    ax2.plot(d[3],label=hist_names[3])
    ax2.tick_params(labelright=True)
    ax2.legend()
    ax2.set_xlabel('metric')
    ax2.grid(True)
    fig.savefig(runfolder + '/' + filename + str(kfold) + '.png')
#--------------------------------------------------------------------------------------------
def show_figure(d, runfolder,kfold, metric_name):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_figheight(8)
    fig.set_figwidth(12)
    plt.gca().cla()
    
    ax1.plot(d[0],label='train_loss' )
    ax1.plot(d[1],label='val_loss')
    ax1.tick_params(labelright=True)
    ax1.legend()
    ax1.set_xlabel('loss')
    ax1.grid(True)
    
    ax2.plot(d[2],label='train_' + metric_name[0])
    ax2.plot(d[3],label='val_' + metric_name[0])
    ax2.plot(d[4],label='train_' + metric_name[1])
    ax2.plot(d[5],label='val_' + metric_name[1])
    ax2.tick_params(labelright=True)
    ax2.legend()
    ax2.set_xlabel('metric')
    ax2.grid(True)
    fig.savefig(runfolder + '/history_' +str(kfold) + '.png')
