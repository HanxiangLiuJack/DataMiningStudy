import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sb
import re
import json
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def replaceNull(columns):
    value=columns[0]
    if pd.isnull(value):
        return 0
    else:
        return value


def combineData(youtubeTrendingData,youtubeNonTrendingData):
    youtubeTrendingData = youtubeTrendingData.dropna()
    youtubeTrendingData = youtubeTrendingData.drop(['trending_date', 'channel_title', 'thumbnail_link', 'description', 'comments_disabled', 'ratings_disabled', 'video_error_or_removed', 'tags', 'title','video_id','publish_time'], axis = 1)
    youtubeTrendingData['trending'] = 1

    youtubeNonTrendingData = youtubeNonTrendingData.dropna()
    youtubeNonTrendingData = youtubeNonTrendingData.drop(['channelId','thumbnail','description', 'tags', 'title', 'video_id','publish_time'], axis = 1)
    youtubeNonTrendingData['comment_count']=youtubeNonTrendingData[['comment_count']].apply(replaceNull,axis=1)
    youtubeNonTrendingData['dislikes']=youtubeNonTrendingData[['dislikes']].apply(replaceNull,axis=1)
    youtubeNonTrendingData['likes']=youtubeNonTrendingData[['likes']].apply(replaceNull,axis=1)
    youtubeNonTrendingData['views']=youtubeNonTrendingData[['views']].apply(replaceNull,axis=1)
    youtubeNonTrendingData['trending'] = 0
                                                            
    youtubeData=pd.concat([youtubeTrendingData,youtubeNonTrendingData],ignore_index=True, sort = True)
    return youtubeData

youtubeTrendingData = pd.DataFrame(pd.read_csv('./data/CAvideos.csv', encoding = 'ISO-8859-1'))
youtubeNonTrendingData = pd.DataFrame(pd.read_csv('./data/NonTrendingVideos.csv', encoding = 'ISO-8859-1'))

youtubeData = combineData(youtubeTrendingData, youtubeNonTrendingData)

X = youtubeData.loc[:, ['category_id','comment_count','dislikes','likes','views']].values
Y = youtubeData.loc[:, ['trending']].values

os = SMOTE(random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10)
os_data_X, os_data_Y = os.fit_sample(X_train,Y_train.ravel())

logReg = LogisticRegression(solver='lbfgs')
logReg.fit(os_data_X,os_data_Y.ravel())
Y_pred = logReg.predict(X_test)

confusion_matrix = confusion_matrix(Y_test,Y_pred)
accuracy=((confusion_matrix[0][0]+confusion_matrix[1][1])/len(Y_test)*100)
print('\nAccuracy of logistic regression classifier on test set:',accuracy,'\n')
print('Classfication Report\n',classification_report(Y_test, Y_pred))

logit_roc_auc = roc_auc_score(Y_test, Y_pred.ravel())
fpr, tpr, thresholds = roc_curve(Y_test, logReg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()











