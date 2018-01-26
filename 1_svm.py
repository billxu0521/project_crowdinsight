#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:58:36 2018

@author: billxu
"""
# -*- coding: utf-8 -*-
import csv
import jieba
#jieba.load_userdict('wordDict.txt')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


# gettraindata
def readtrain():
    with open('Ch_trainfile_Sentiment_3000.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row for row in reader]
    content_train = [i[0] for i in column1[1:]] #第一列为文本内容，并去除列名
    opinion_train = [i[1] for i in column1[1:]] #第二列为类别，并去除列名
    print ('%s sentence' % len(content_train))
    train = [content_train, opinion_train]
    return train


# utf8 -> unicode
def changeListCode(b):
    a = []
    for i in b:
        a.append(i.decode('utf8'))
    return a


# 对列表进行分词并用空格连接
def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c


# corpus = ["我 来到 北京 清华大学", "他 来到 了 网易 杭研 大厦", "小明 硕士 毕业 与 中国 科学院"]
train = readtrain()
content = segmentWord(train[0])
opinion = train[1]


# 划分
train_content = content[:2100]
test_content = content[2100:]
train_opinion = opinion[:2100]
test_opinion = opinion[2100:]

# 计算权重
vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))  # 先转换成词频矩阵，再计算TFIDF值
all_tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(content))  # 先转换成词频矩阵，再计算TFIDF值

print (tfidf.shape)
print (all_tfidf.shape)

'''
# 单独预测

word = vectorizer.get_feature_names()
weight = tfidf.toarray()

# 分类器
clf = MultinomialNB().fit(all_tfidf, opinion)
docs = ["总体感觉蛮好的", 1]

new_tfidf = tfidftransformer.transform(vectorizer.transform(docs))
predicted = clf.predict(new_tfidf)
print (predicted)
'''


# 训练和预测一体
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=0.99, kernel = 'linear'))])
text_clf = text_clf.fit(train_content, train_opinion)
predicted = text_clf.predict(test_content)
accuracy = metrics.accuracy_score(test_opinion, predicted)
print ('SVC',np.mean(predicted == test_opinion))
print(accuracy)
print (set(predicted))

#print metrics.confusion_matrix(test_opinion,predicted) # 混淆矩阵


# 循环调参
parameters = {'vect__max_df': (0.4, 0.5, 0.6, 0.7),'vect__max_features': (None, 5000, 10000, 15000),
              'tfidf__use_idf': (True, False)}
grid_search = GridSearchCV(text_clf, parameters, n_jobs=1, verbose=1)
grid_search.fit(content, opinion)
best_parameters = dict()
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

