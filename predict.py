import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import traceback
import datetime
from dateutil.relativedelta import relativedelta
import time
import xgboost as xgb
from sklearn.linear_model import LinearRegression

train = pd.read_csv("data/train.csv", header=0, parse_dates=["Date"], low_memory=True)
test = pd.read_csv("data/test.csv", header=0,parse_dates=["Date"], low_memory=True)




def do_xgboost(train_temp, test_temp):
    target= train_temp.logSales.values
    train_temp.drop(["logSales"], axis=1, inplace=True)
    reg =  xgb.XGBRegressor(
                     learning_rate=0.02,
                     min_child_weight=1.5,
                     n_estimators=4000,                                                                  
                     silent=1, nthreads=2).fit(train_temp, target)
    return (np.expm1(reg.predict(test_temp)))


def linearRegression(train_temp, test_temp):
    target= train_temp.logSales.values
    train_temp.drop(["logSales"], axis=1, inplace=True)
    reg = LinearRegression().fit(train_temp, target)
    predictions=(np.expm1(reg.predict(test_temp)))
    predictions[predictions==np.inf]=0
    predictions[predictions>400000]=100
    return predictions


def prosData(dfTrain, dfTest):
    dfFeature = pd.read_csv('data/features.csv', parse_dates=["Date"])
    dfStores = pd.read_csv('data/stores.csv')
    dfTrainTmp           = pd.merge(dfTrain, dfStores)
    dfTestTmp            = pd.merge(dfTest, dfStores)   
    train                = pd.merge(dfTrainTmp, dfFeature)
    test                 = pd.merge(dfTestTmp, dfFeature)
    train['Year']        = pd.to_datetime(train['Date']).dt.year
    train['Month']       = pd.to_datetime(train['Date']).dt.month
    train['Day']         = pd.to_datetime(train['Date']).dt.day
    train['Days']        = train['Month']*30+train['Day'] 
    train['Type']        = train['Type'].replace('A',1)
    train['Type']        = train['Type'].replace('B',2)
    train['Type']        = train['Type'].replace('C',3)
    train['daysHoliday'] = train['IsHoliday']*train['Days']
    train['logSales']    = np.log1p(np.abs(train['Weekly_Sales']))
    test['Year']         = pd.to_datetime(test['Date']).dt.year
    test['Month']        = pd.to_datetime(test['Date']).dt.month
    test['Day']          = pd.to_datetime(test['Date']).dt.day
    test['Days']         = test['Month']*30+test['Day']
    test['Type']         = test['Type'].replace('A',1)
    test['Type']         = test['Type'].replace('B',2)
    test['Type']         = test['Type'].replace('C',3)
    test['daysHoliday']  = test['IsHoliday']*test['Days']
    train                = train.drop(['CPI','Unemployment','Date',
                                       'MarkDown1','MarkDown2','MarkDown3', 
                                       'MarkDown4','MarkDown5','Weekly_Sales'],axis=1)                                      
    test                 = test.drop(['CPI','Unemployment','Date',
                                      'MarkDown1','MarkDown2','MarkDown3',
                                      'MarkDown4','MarkDown5'],axis=1)
    return (train,test)


def mypredict(train, test, new_test, t):
    store= sorted(test.Store.unique())
    dept = sorted(test.Dept.unique())
    train = pd.concat([train, new_test], axis=0)
    train.sort_values(by=["Store", "Dept", "Date"], inplace=True)
    try:
        for s in store:
            print("Store: ", s)
            start = time.time()
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
                        pred_ix = pd.date_range(pred_start, pred_stop+np.timedelta64(2, 'W'), freq="W")
                        pred_ix-=np.timedelta64(2, 'D')
                    else:
                        pred_ix = pd.date_range(pred_start, pred_stop +np.timedelta64(3, 'D'), freq="W")
                    
                    if t in [12]:
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
                            pred3.ix[index, "predection3"]=np.mean([sale, sale_week_after,sale_week_before])
                        pred1=pred1.astype("float32")
                        pred2=pred2.astype("float32")
                        pred3=pred3.astype("float32")
                        indexes1=test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred1.index)), "Date"]
                        indexes2=test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred2.index)), "Date"]
                        indexes3=test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred3.index)), "Date"]
                        test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred1.index)), "Weekly_Pred1"]= np.expm1(pred1.loc[indexes1,'predection1'].values)
                        test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred2.index)), "Weekly_Pred2"]= np.expm1(pred2.loc[indexes2,'predection2'].values)
                        test.loc[(test.Store==s) & (test.Dept==d) & (test.Date.isin(pred3.index)), "Weekly_Pred3"]= np.expm1(pred3.loc[indexes3,'predection3'].values)
                    elif t in [0, 1, 6,7,8,9,2,3,4,5]:
                        time_period= np.datetime64("2011-03-04") + np.timedelta64(61*t,'D')
                        test_lower=time_period
                        test_upper=np.datetime64(time_period) + np.timedelta64(8, 'W')
                        if t ==8:
                            test_upper+=+np.timedelta64(2, 'D')
                        train_temp = train[(train.Dept==d) & (train.Store==s)]
                        test_temp=test[(test.Dept==d) & (test.Store==s)]
                        test_temp = test_temp[(test_temp.Date>=test_lower) & (test_temp.Date<=test_upper)]
                        test_temp.drop(["Weekly_Pred1", "Weekly_Pred2", "Weekly_Pred3"], axis=1, inplace=True)
                        train_t, test_t=prosData(train_temp.copy(), test_temp.copy())
                        if len(test_temp)==0:
                            continue
                        results = linearRegression(train_t.copy(), test_t.copy())
                        test.loc[test_temp.index, "Weekly_Pred1"]=results
                        test.loc[test_temp.index, "Weekly_Pred2"]=results
                        test.loc[test_temp.index, "Weekly_Pred3"]=results

            print(time.time()-start)
        return train, test
    except Exception as ex:
        traceback.print_exc()
        print("Error occured while prediction for department {0} and store {1} due to".format(d, s), ex)
        


model_pred_columns = ['Weekly_Pred{0}'.format(i) for i in range(1, 4)]
n_folds = 10
new_test = None
wae = pd.DataFrame(np.zeros((n_folds, 3)),
                   columns=['model_one', 'model_two', 'model_three'])
for t in range(0, n_folds):
    print("Processing fold ", t)
    train, test=mypredict(train, test, new_test, t)
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
wae.to_csv('Error.csv', index=False)