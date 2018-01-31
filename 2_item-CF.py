#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:19:50 2018

@author: billxu
"""

import math  
  
class ItemBasedCF:  
    def __init__(self,train_file):  
        self.train_file = train_file  
        self.readData()  
    def readData(self):  
        self.train = dict()     #將用戶購買數量當作分數
        for line in open(self.train_file):  
            lines = line.strip().split(",")  
            user = lines[1]
            score = lines[3]
            item = lines[2]
            self.train.setdefault(user,{})  
            self.train[user][item] = score  
  
    def ItemSimilarity(self):  
        C = dict()  #j物品共現度  
        N = dict()  #物品被用戶購買 
        for user,items in self.train.items():  
            for i in items.keys():  
                N.setdefault(i,0)  
                N[i] += 1  
                C.setdefault(i,{})  
                for j in items.keys():  
                    if i == j : continue  
                    C[i].setdefault(j,0)  
                    C[i][j] += 1  
        #計算相似度
        self.W = dict()  
        for i,related_items in C.items():  
            self.W.setdefault(i,{})  
            for j,cij in related_items.items():  
                self.W[i][j] = cij / (math.sqrt(N[i] * N[j]))  
        return self.W  
  
    #推薦前N個物品
    def Recommend(self,user,K=3,N=5):  
        rank = dict()  
        action_item = self.train[user]     #該顧客過去的分數  
        for item,score in action_item.items():  
            for j,wj in sorted(self.W[item].items(),key=lambda x:x[1],reverse=True)[0:K]:  
                if j in action_item.keys():  
                    continue  
                rank.setdefault(j,0)  
                rank[j] += int(score) * wj  
        return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:N])  
      
  
Item = ItemBasedCF("rs.csv")  
Item.ItemSimilarity()  
recommedDic = Item.Recommend("32013007606310")  
for k,v in recommedDic.items():  
    print ("item:",k,"\t",v) 