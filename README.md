# Navie Bayes Classfiers report

## 1. Code

```python
def load_data(csv_file):#載入資料(特徵與標籤)
    data = pd.read_csv(csv_file)

    features = data[['f2', 'f3', 'f4', 'f5']]    #features載入4個特徵列
    features = features.iloc[:,:].values    #提取特徵值部分並轉為ndarray

    label = data['label']    #label載入label列
    label = label.iloc[:].values    #提取label值部分並轉為ndarray

    return features, label
```

```python
def counting(label):    #計算每個label種類的個數
    count = np.zeros(3, dtype=int)    #count = [0, 0, 0]

    for i in range(3):
        count[i] = Counter(label)[i+1]

    return count
```


```python
def caculate_mean(features, count):#計算每種標籤每個特徵的平均值
    mean = np.zeros((3, 4))    #[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    for i in range(4):    #第i個 feature
        for j in range(3):    #第j個 label
            for k in range(count[j]*j,count[j]*(j+1)):#1(0~49),2(50,99),3(100~149)
                mean[j][i] += features[k][i]    #xi累加
            mean[j][i] /= count[j]    #除以總數

    return mean
```
Mean formula：
![](https://i.imgur.com/SULDjAq.png)
```python
def caculate_varience(features, mean, count):#計算每種標籤每個特徵的變異數
    var = np.zeros((3, 4))    #[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    for i in range(4):    #第i個 feature
        for j in range(3):    #第j個 label
            for k in range(count[j]*j,count[j]*(j+1)):#1(0~49),2(50,99),3(100~149)
                var[j][i] += (features[k][i] - mean[j][i])**2    #(xi-u)^2 累加
            var[j][i] /= (count[j] - 1)    #除以(n-1)

    return var
```
Varience formula：
![](https://i.imgur.com/cKftayI.jpg)


```python
def normal_dis(x, mean, var):#常態分佈公式

    val = 1/((2*m.pi*var)**0.5)*m.exp(-((x-mean)**2)/(2*var))    

    return val
```
Normal distribution formula：
![](https://i.imgur.com/AV6EF4P.png)

```python
def predict(features, count, mean, var):#預測
    pre = np.zeros(len(features))    #存放預測的標籤值
    
    for i in range(len(features)):    #第i筆資料
        pro = np.zeros(3)    #暫存要相乘的項
        for j in range(3):    #第j個 label
            for k in range(4):    #第k個 feature
                pro[j] += m.log(normal_dis(features[i][k], mean[j][k], var[j][k]))
#4個特徵的值、平均數、變異數帶入公式後相乘，但因為相乘後導致數值太小會underflow，所以取log變相加
            pro[j] += m.log((count[j]/len(features)))    #乗以樣本出現機率(log變相加)

        pre[i] = np.argmax(pro) + 1
        
    return pre.astype('int')
```
Navie Bayes Classfiers formula:
![](https://i.imgur.com/hfXqUCC.png)

```python
def output(filename, pre):#預測結果存入csv檔
    with open(filename, 'w') as file:    #讀檔
        file.write('serial id' + "," + 'label' + "\n")    #標題
        for i in range(len(pre)):    #逐行寫入預測值
            #          第幾筆data   ,      預測值
            file.write(str(i+1) + "," + str(pre[i]) + "\n")
            
    file.close()    #關檔
```
### Main
```python
features, label = load_data('train.csv')
features_,label_ = load_data('test.csv')

count = counting(label)
mean = caculate_mean(features, count)
var = caculate_varience(features, mean, count)

pre = predict(features_, count, mean, var)
print("acc:", score(pre, label_)*100, "%")

print(pre)
output("./output.csv", pre)
```
predict：
![](https://i.imgur.com/rVXXj75.png)

![](https://i.imgur.com/xrgxjRX.png)

## 2. Summary
透過這次的實作，讓我更了解貝氏分類器的整個過程，例如一開始我沒有將相乘那取log，導致了underflow預測準確率很低，透過trace發現原來值越乘越小，之後取log就解決此問題。不過比較可惜的是我沒有去做其他類型的貝氏分類器，看可不可以改善模型的預測準確度。我認為這種實作非常有趣，能與同學一起競爭並學習到相關的知識點，希望能有多一點這樣的作業。
