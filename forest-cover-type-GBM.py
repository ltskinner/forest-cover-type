# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn import model_selection, metrics 

import matplotlib.pyplot as plt


"""
Because there was so much overlap in the components of each soil classification, I decided to convert the
one hot markings into marking each individual component
"""
def soilFixer(data):
    originals = {
        'Soil_Type1': '1 cathedral family - rock outcrop complex, extremely stony.', 
        'Soil_Type2': '2 vanet - ratake families complex, very stony.', 
        'Soil_Type3': '3 haploborolis - rock outcrop complex, rubbly.', 
        'Soil_Type4': '4 ratake family - rock outcrop complex, rubbly.', 
        'Soil_Type5': '5 vanet family - rock outcrop complex complex, rubbly.', 
        'Soil_Type6': '6 vanet - wetmore families - rock outcrop complex, stony.', 
        'Soil_Type7': '7 gothic family.', 
        'Soil_Type8': '8 supervisor - limber families complex.', 
        'Soil_Type9': '9 troutville family, very stony.', 
        'Soil_Type10': '10 bullwark - catamount families - rock outcrop complex, rubbly.', 
        'Soil_Type11': '11 bullwark - catamount families - rock land complex, rubbly.', 
        'Soil_Type12': '12 legault family - rock land complex, stony.', 
        'Soil_Type13': '13 catamount family - rock land - bullwark family complex, rubbly.', 
        'Soil_Type14': '14 pachic argiborolis - aquolis complex.', 
        'Soil_Type15': '15 unspecified in the usfs soil and elu survey.', 
        'Soil_Type16': '16 cryaquolis - cryoborolis complex.', 
        'Soil_Type17': '17 gateview family - cryaquolis complex.', 
        'Soil_Type18': '18 rogert family, very stony.', 
        'Soil_Type19': '19 typic cryaquolis - borohemists complex.', 
        'Soil_Type20': '20 typic cryaquepts - typic cryaquolls complex.', 
        'Soil_Type21': '21 typic cryaquolls - leighcan family, till substratum complex.',
        'Soil_Type22': '22 leighcan family, till substratum, extremely bouldery.', 
        'Soil_Type23': '23 leighcan family, till substratum - typic cryaquolls complex.', 
        'Soil_Type24': '24 leighcan family, extremely stony.', 
        'Soil_Type25': '25 leighcan family, warm, extremely stony.', 
        'Soil_Type26': '26 granile - catamount families complex, very stony.', 
        'Soil_Type27': '27 leighcan family, warm - rock outcrop complex, extremely stony.', 
        'Soil_Type28': '28 leighcan family - rock outcrop complex, extremely stony.', 
        'Soil_Type29': '29 como - legault families complex, extremely stony.', 
        'Soil_Type30': '30 como family - rock land - legault family complex, extremely stony.', 
        'Soil_Type31': '31 leighcan - catamount families complex, extremely stony.', 
        'Soil_Type32': '32 catamount family - rock outcrop - leighcan family complex, extremely stony.', 
        'Soil_Type33': '33 leighcan - catamount families - rock outcrop complex, extremely stony.', 
        'Soil_Type34': '34 cryorthents - rock land complex, extremely stony.', 
        'Soil_Type35': '35 cryumbrepts - rock outcrop - cryaquepts complex.', 
        'Soil_Type36': '36 bross family - rock land - cryumbrepts complex, extremely stony.', 
        'Soil_Type37': '37 rock outcrop - cryumbrepts - cryorthents complex, extremely stony.', 
        'Soil_Type38': '38 leighcan - moran families - cryaquolls complex, extremely stony.', 
        'Soil_Type39': '39 moran family - cryorthents - leighcan family complex, extremely stony.', 
        'Soil_Type40': '40 moran family - cryorthents - rock land complex, extremely stony.'
    }

    # New columns
    soils = ['cathedral', 'haplobor', 'ratake', 'vanet', 'wetmore', 
            'gothic', 'supervisor', 'limber', 'troutville', 'bullwark', 
            'catamount', 'legault', 'pachic', 'gateview', 'boro', 'leighcan', 
            'moran', 'granile', 'como', 'bross',
            'cryumbrepts', 'cryaquepts', 'cryorthents', 'cryaquolls', 'cryaquolis']
    
    converted = {k:[] for k in soils}

    for index, row in data.iterrows():
        for c, s in enumerate(originals.keys()):
            if row[s] == 1:
                for nw in soils:
                    if nw in originals[s]:
                        converted[nw].append(1)
                    else:
                        converted[nw].append(0)
                break


"""
Was looking for features of similar units to play with
morn_sun ended up being in the top 8 most highly weighted features
"""
def sunFixer(data):
    data['total_sun'] = data['Hillshade_9am'] + data['Hillshade_Noon'] + data['Hillshade_3pm']
    data['morn_sun'] = data['Hillshade_9am'] + data['Hillshade_Noon']
    data['after_sun'] = data['Hillshade_Noon'] + data['Hillshade_3pm']
    data['diff_sun'] = data['Hillshade_9am'] - data['Hillshade_3pm']
    data['Hillshade_3pm'] += 1 # incrementing to root zeros
    data['ratio_sun'] = data['Hillshade_9am'] / data['Hillshade_3pm']

    data['Aspect'] += 1
    data['Slope'] += 1
    data['aspct_slp'] = data['Aspect'] / data['Slope']


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

    # Feature Engineering
    distEngineering(train)
    #distEngineering(test)
    soilFixer(train)
    #soilFixer(test)
    sunFixer(train)
    #sunFixer(test)


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
    gbm = GradientBoostingClassifier( learning_rate=0.01, # Tweaking here # .125 orig
                                        n_estimators=3600, # Tweaking here # 1600 orig
                                        min_samples_split=10, # [OK]
                                        min_samples_leaf=50, # [OK]
                                        max_depth=14, # [OK]
                                        max_features='sqrt',
                                        subsample=0.9, # .8 [OK]
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