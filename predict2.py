import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels as smt
import matplotlib.pyplot as plt
import statsmodels.graphics.gofplots as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")
import traceback
import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AR
import time
import pyramid as pm
from pyramid.utils import c, diff

train = pd.read_csv("data/train.csv", header=0, parse_dates=["Date"], low_memory=True)
test = pd.read_csv("data/test.csv", header=0,parse_dates=["Date"], low_memory=True)



def mypredict(train, test, new_test, t):
    store= sorted(test.Store.unique())
    dept = sorted(test.Dept.unique())
    try:
        for s in store:
            if t <=6:
                break
            print("Store: ", s)
            for d in dept:
                if len(train[(train.Store==s) & (train.Dept==d)])!=0: 
                    train_end_date= datetime.date(2011,2,28) + relativedelta(months=2*t)
                    if t in [1,4,5,7,9]:
                        train_end_date+=relativedelta(weeks=1)
                    idx = pd.date_range('2010-02-05', train_end_date, freq="W")
                    idx=idx-pd.DateOffset(days=2) 
                    
                    train_temp = train[(train.Store==s) & (train.Dept==d)]
                    train_temp.set_index('Date', inplace=True)
                    

                    series=np.log1p(np.abs(train_temp['Weekly_Sales']))
                    series=series.reindex(idx, fill_value=0)

                    
                    if len(train[(train.Store==s) & (train.Dept==d)])==1:
                            series[:]=(np.random.randint(1, 6, len(series)))
                            
                    pred_start=series.index.max() + np.timedelta64(1, 'W')
                    pred_stop=series.index.max() + np.timedelta64(9, 'W')
                    interpolated= series
                    
                    if t==8:
                        pred_ix = pd.date_range(pred_start, pred_stop +np.timedelta64(3, 'D')+np.timedelta64(1, 'W'), freq="W")
                    else:
                        pred_ix = pd.date_range(pred_start, pred_stop +np.timedelta64(3, 'D'), freq="W")

                    if t==0:
                        model3 = ARMA(interpolated,order=(0, 1))
                        model_fit3 = model3.fit()
                        pred=model_fit3.predict(start=pred_start, end=pred_stop)
                        pred3=pd.DataFrame(pred, columns=["predection3"])
                        pred2=pd.DataFrame(pred, columns=["predection2"])
                        pred1=pd.DataFrame(pred, columns=["predection1"])
                    elif t in [1,2,3,4,5,6]:
                        pred1= pd.DataFrame(index= pred_ix, columns=["predection1"])
                        pred1.index= pred1.index-np.timedelta64(2, 'D')
                        pred2= pd.DataFrame(index= pred_ix, columns=["predection2"])
                        pred2.index= pred2.index-np.timedelta64(2, 'D')
                        pred3= pd.DataFrame(index= pred_ix, columns=["predection3"])
                        pred3.index= pred3.index-np.timedelta64(2, 'D')
                        for index, row in pred1.iterrows():
                            date=(index+np.timedelta64(2, 'D')-np.timedelta64(1, 'Y'))
                            date_week_before=date-np.timedelta64(1, 'W')
                            date_after_week=date+np.timedelta64(1, 'W')
                            sale=interpolated.loc[date.date()]
                            sale_week_before=interpolated.loc[date_week_before.date()]
                            sale_week_after=interpolated.loc[date_after_week.date()]
                            pred1.ix[index, "predection1"]=sale
                            pred2.ix[index, "predection2"]=(np.add(np.add(sale, sale_week_after),sale_week_before))/3
                            pred3.ix[index, "predection3"]=(np.add(np.add(sale, sale_week_after),sale_week_before))/3
                        pred1=pred1.astype("float32")
                        pred2=pred2.astype("float32")
                        pred3=pred3.astype("float32")
                    else:
                        #   pm.plot_acf(diff(interpolated, lag=52, differences=1))
                        
                        #diff = difference(series)
                        #model1=ExponentialSmoothing(series ,seasonal_periods=52 ,trend=None, seasonal='add')
                        #model_fit1= model1.fit()
                        #pred1=pd.DataFrame(model_fit1.predict(start=pred_start, end=pred_stop), columns=["predection1"])

                        #model2 = AR(diff)
                        model2=smt.tsa.statespace.sarimax.SARIMAX(interpolated,order=(10, 0, 0), seasonal_order=(0, 1, 0, 52))
                        model_fit2 = model2.fit()
                        pred=model_fit2.predict(n_periods=9)
                    
                        pred2=pd.DataFrame(pred, columns=["predection2"])
                        pred3=pd.DataFrame(pred, columns=["predection3"])
                        pred1=pd.DataFrame(pred, columns=["predection1"])
                        
                        #model3 = ARMA(interpolated,order=(0, 1))
                        #model_fit3 = model3.fit()
                        #pred=model_fit3.predict(start=pred_start, end=pred_stop)
                        #pred3=pd.DataFrame(pred, columns=["predection3"])
                        
                        
                    #################################################################################
                    
                    
                    indexes1=test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred1.index)), "Date"]
                    indexes2=test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred2.index)), "Date"]
                    indexes3=test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred3.index)), "Date"]
                    test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred1.index)), "Weekly_Pred1"]= np.expm1(pred1.loc[indexes1,'predection1'].values)
                    test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred2.index)), "Weekly_Pred2"]= np.expm1(pred2.loc[indexes2,'predection2'].values)
                    test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred3.index)), "Weekly_Pred3"]= np.expm1(pred3.loc[indexes3,'predection3'].values)
            train = pd.concat([train, new_test], axis=0)
            train.sort_values(by=["Store", "Dept", "Date"], inplace=True)
    except Exception as ex:
        traceback.print_exc()
        print("Error occured while prediction for department {0} and store {1} due to".format(d, s), ex)
        

try:
    model_pred_columns = ['Weekly_Pred{0}'.format(i) for i in range(1, 4)]
    n_folds = 10
    new_test = None
    wae = pd.DataFrame(np.zeros((n_folds, 3)),
                       columns=['model_one', 'model_two', 'model_three'])
    for t in range(0, n_folds):
        mypredict(train, test, new_test, t)
        fold_file = 'data/fold_{t}.csv'.format(t=(t+1))
        new_test = pd.read_csv(fold_file, parse_dates=["Date"])
        scoring_df = new_test.merge(test, on=['Date', 'Store', 'Dept'], how='left')
        weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday:
                                                    5 if is_holiday else 1)
        weights = weights.values.reshape(-1, 1)
        actuals = scoring_df['Weekly_Sales'].values.reshape(-1, 1)
        preds = scoring_df[model_pred_columns].values
        wae.iloc[t, :] = (
            np.sum(weights * np.abs(actuals - preds), axis=0) / np.sum(weights))
        print("Fold", t ,"completed")
    wae.to_csv('Error.csv', index=False)
except Exception as ex:
    traceback.print_exc()
    print("Encountered an exception in running different folds at fold = {}, due to {}".format(t, ex))



"""
model_one	model_two	model_three
3641.750273758943	2460241.0838486515	1888.9395296419218
3524.1206538409674	inf	2411.19773411414
3371.077658624075	8.211090187323651e+55	2329.1210560028726
2761.67027607466	1.782516747915124e+22	2105.813365015425
7425.5715073389465	inf	9040.917468193828
2846.6523208634953	220838387694.62045	2618.2005329489875
3464.8674022621226	1.7399590967585304e+55	2167.084159428575
4918.573312296537	inf	4037.526108212932
4840.949181943381	3439.695710153339	3967.7960540219547
2824.5608776572785	1690.48161394282	2233.5873995891898
"""
