# DSAI HW1

## Run Code

```
python app.py --training "data/power_information_detail.csv" --output submission.csv
```

---

## EDA
* 下圖為2021/1到2022/3的備轉容量，從此圖可以看出備轉容量與季節有很大的關聯性，通常春天備轉容量通常不多都在4000以下，但是一進入5月，備轉容量開始急速升高，呈現與季節高度相關的週期性。

![](https://i.imgur.com/gmGYr53.png)

* 下圖為近三年的備轉容量分布，可以大略看出，每年都遵從上述的周期性變化，且當天的備轉容量一定跟近一個月的高度相關且很少急遽攀升的狀況出現，因此可以使用前30天的備轉容量資料來去預測當天的備轉容量。

![](https://i.imgur.com/UPegiwq.png)

## Dataset

* 使用[台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)
* 因為每年週期變化性不大，所以只使用前一年(2021)的data即可。

## Method

* 用LSTM使用前30天的data去預測當天的備轉容量。





