#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:58:36 2018

@author: billxu
"""
# -*- coding: utf-8 -*-
import csv
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import cross_validation

# 取得資料
def readtrain():
    with open('Ch_trainfile_Sentiment_3000.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row for row in reader]
    content_train = [i[0] for i in column1[1:]] 
    opinion_train = [i[1] for i in column1[1:]] 
    print ('%s sentence' % len(content_train))
    train = [content_train, opinion_train]
    return train

#斷詞處理
def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c

def main():
    train = readtrain()
    content = segmentWord(train[0])
    opinion = train[1]
    
    #切割訓練測試資料段
    train_content = content[:2100]
    test_content = content[2100:]
    train_opinion = opinion[:2100]
    test_opinion = opinion[2100:]
    
    #計算權重
    vectorizer = CountVectorizer()
    tfidftransformer = TfidfTransformer()
    tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))  
    all_tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(content))  
    
    print (tfidf.shape)
    print (all_tfidf.shape)
    
    # 訓練
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=0.99, kernel = 'linear'))])
    text_clf = text_clf.fit(train_content, train_opinion)
    scores = cross_validation.cross_val_score(text_clf, train_content, train_opinion, cv=5)#5-fold cv
    predicted = text_clf.predict(test_content)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
   
if __name__ == "__main__":
    main()
    # test()