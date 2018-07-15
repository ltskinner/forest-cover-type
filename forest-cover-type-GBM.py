# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn import model_selection, metrics 
#from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt

def distEngineering(train)
    # Thank you @codename007 for this feature engineering
    ####################### Train data #############################################
    train['HF1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
    train['HF2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
    train['HR1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
    train['HR2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
    train['FR1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
    train['FR2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
    train['ele_vert'] = train.Elevation-train.Vertical_Distance_To_Hydrology

    train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
    train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    #Mean distance to Amenities 
    train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3 
    #Mean Distance to Fire and Water 
    train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2 


def main():

    train = pd.read_csv("../train.csv")
    #test = pd.read_csv("../test.csv")

    # Feature Engineering - I did significant Soil_Type engineering, which did not improve results. Please see soilEngineering.py
    distEngineering(train)
    #distEngineering(test)


    target = "Cover_Type"
    IDcol = "Id"

    """
    ------------------- Grid Searching -------------------
    predictors = [x for x in train.columns if x not in [target, IDcol]]
    
    
    # n_estimators
    param_test1 = {'n_estimators':range(300, 401, 10)} # 20, 40, 60, 80 # using 81 to get inclusive 80
    gsearch1 = model_selection.GridSearchCV(estimator=GradientBoostingClassifier(
                                                                    learning_rate=0.5,
                                                                    min_samples_split=250, # 87,000/5000 = 174, 11,000/174 = 62
                                                                    min_samples_leaf=50,
                                                                    max_depth=8, 
                                                                    max_features='sqrt',
                                                                    subsample=0.8,
                                                                    random_state=10),
                                            param_grid=param_test1,
                                            #scoring='roc_auc',
                                            n_jobs=4,
                                            iid=False,
                                            cv=5)
    
    #param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(50,250,50)}
    param_test2 = {'min_samples_leaf':range(30, 71, 10)}
    gsearch1 = model_selection.GridSearchCV(estimator=GradientBoostingClassifier(
                                                                    learning_rate=0.5, # [OK]
                                                                    n_estimators=380, # [OK]
                                                                    min_samples_split=100, # [OK]
                                                                    min_samples_leaf=40, # [OK]
                                                                    max_depth=11, # [OK]
                                                                    max_features='sqrt',
                                                                    subsample=0.8,
                                                                    random_state=10), # [OK]
                                            param_grid=param_test2,
                                            #scoring='roc_auc',
                                            n_jobs=4,
                                            iid=False,
                                            cv=5)
    
    param_test3 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
    gsearch1 = model_selection.GridSearchCV(estimator=GradientBoostingClassifier(
                                                                    learning_rate=0.1, # Dropping learning rate JUST to test subsample
                                                                    n_estimators=380, # [OK]
                                                                    min_samples_split=100, # [OK]
                                                                    min_samples_leaf=40, # [OK]
                                                                    max_depth=11, # [OK]
                                                                    max_features='sqrt',
                                                                    #subsample=0.8, # .8 [OK]
                                                                    random_state=10), # [OK]
                                            param_grid=param_test3,
                                            #scoring='roc_auc',
                                            n_jobs=4,
                                            iid=False,
                                            cv=5)

    gsearch1.fit(train[predictors], train[target])
    print(gsearch1.grid_scores_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)

    """
    
    # Final Best Hyper Parameters
    gbm = GradientBoostingClassifier( learning_rate=0.0625, # Tweaking here # .125 orig
                                        n_estimators=3200, # Tweaking here # 1600 orig
                                        min_samples_split=100, # [OK]
                                        min_samples_leaf=40, # [OK]
                                        max_depth=11, # [OK]
                                        max_features='sqrt',
                                        subsample=0.8, # .8 [OK]
                                        random_state=10) # [OK]

    gbm.fit(train[predictors], train['Cover_Type'])

    
    print("\n[+] Fitted [+]")
    """
    # Predict training set:
    preds = gbm.predict(test[predictors])
    submission = pd.DataFrame({'Id':test['Id'], 'Cover_Type':preds})

    submission[['Id', 'Cover_Type']].to_csv("GBM-submission.csv", index=False)
    """



if __name__ == "__main__":
    main()