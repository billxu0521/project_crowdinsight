
=====================
在進行這個作業前重新定義資料標籤
將[-2,-1,0,1,2]轉換成[1,2,3,4,5]

## 1-1 SVM處理
檔名:1_svm.py
第一次摸sklearn這個套件
作法為：
1.先取得資料
2.斷詞處理
3.切割訓練段落
4.處理權重
5.訓練
6.驗證

-結果為Accuracy: 0.72 (+/- 0.03)

## 1-2 LSTM處理
檔名:1_lstm.py
使用先前做過的LSTM架構來跑這筆資料
作法為：
1.先取得資料
2.斷詞處理
3.處理權重
4.切割訓練段落
5.訓練
6.驗證

-結果為Score=0.8603678925778954

====================
## 2-1 推薦系統
檔名:2_item-CF.[y]
先前沒做過推薦系統相關的主題
這題基本上沒達到需求的條件

而看完1,2篇文獻後查了一下
用基於物品的協同過濾做了一個樣板
可以針對不同ID產生推薦item(物品)

-為32013007606310顧客的推薦物品(5個) ===>  {'19360': 2.964997266644405, '16450': 3.1214135783855297, '45887': 3.3920026264998406, '10751': 1.212480002547752, '8987': 1.1633833262844284}



