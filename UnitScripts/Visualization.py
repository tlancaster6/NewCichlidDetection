import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train_log = pd.read_csv('train.log', sep='\t')
print("\n",train_log.columns, "\n")
val=pd.read_csv('val.log', sep='\t')
print(val.columns,"\n")
print(val.head())

fig = plt.figure()
fig2 = plt.figure()
# plt.rcParams.update({'font.size': 7})

def total_loss_vs_epoch():
    """plot the training loss vs epoch
    """
    ax = fig.add_subplot(321)
    ax.set(xlabel='epoch', ylabel='total loss', title='Training Loss vs. Epoch')
    sns.lineplot(x=train_log['epoch'], y= train_log['loss_total'] , ax=ax)

def class_loss_vs_epoch():
    """plot the classifier loss vs epoch
    """
    ax = fig.add_subplot(322)
    ax.set(xlabel='epoch', ylabel='classifier loss', title='Classifier Loss vs. Epoch')
    sns.lineplot(x=train_log['epoch'], y= train_log['loss_classifier'] , ax=ax)

def boxreg_loss_vs_epoch():
    """plot the  loss vs epoch
    """
    ax = fig.add_subplot(325)
    ax.set(xlabel='epoch', ylabel='box reg loss', title='Box Reg Loss vs. Epoch')
    sns.lineplot(x=train_log['epoch'], y= train_log['loss_box_reg'] , ax=ax)
    
def loss_vs_lr():
    """plot the objectness loss vs epoch
    """
    ax = fig.add_subplot(326)
    ax.set(xlabel='LR', ylabel='total loss', title='Training Loss vs. Learning Rate')
    sns.lineplot(x=train_log['lr'], y= train_log['loss_total'] , ax=ax)

    
    
def iou_vs_epoch():
  """plot the iou vs epoch
  """
  ax = fig2.add_subplot(131)
  ax.set(xlabel='epoch', ylabel='avg IOU', title='Avg IOU vs. Epoch')
  sns.lineplot(x=val['epoch'], y= val['IOU'] , ax=ax)

def accuracy_vs_epoch():
    """plot the accurary of predicted sex vs epoch
    """
    acc = pd.DataFrame()
    acc['Label_Accuracy'] = val['Label_Accuracy'].str.split()
    for  i in range(0,acc.shape[0]):
        acc['Label_Accuracy'][i] = int(acc.iloc[i,0][0]) / ( int(acc.iloc[i,0][0]) + int(acc.iloc[i,0][3]) )
    ax = fig2.add_subplot(133)
    ax.set(xlabel='epoch', ylabel='Accuracy of predicted sex', title='Accuracy vs. Epoch')
    sns.lineplot(x=val['epoch'], y= acc['Label_Accuracy'] , ax=ax)



total_loss_vs_epoch()
class_loss_vs_epoch()
boxreg_loss_vs_epoch()
loss_vs_lr()
iou_vs_epoch()
accuracy_vs_epoch()