import os
import joblib
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from quantile_forest import RandomForestQuantileRegressor

HWTYPE = ['HW17', 'HWsum']
INDUSTRY = [['3130', '3340', '3360', '3982', 'combined'],
            ['2614', '2631', '2669', '3130', '3340', '3360', '3982', '4417', '4419', 'combined']]

class HWPrediction:
    """
    Construct a quantile forest to predict the quantile range of HW generation

    Args:
        train(pd.DataFrame): Train data set
        test(pd.DataFrame): Test data set
        X(list): Input features
        y(str): Target variable
        sector(str): Industrial sector
        train_class(pd.DataFrame): Train data set for classification
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame,
                 X:list, y:str, sector:str, train_class:pd.DataFrame=pd.DataFrame([])):
        if y not in ['HWsum', 'HW17']:
            raise ValueError("HWtype must be HWsum or HW17!")
        self.X_train = train.loc[:, train.columns.isin(X)]
        self.y_train = np.array(train.loc[:, train.columns == y]).reshape(-1,)
        self.X_test = test.loc[:, test.columns.isin(X)]
        self.y_test = np.array(test.loc[:, test.columns == y]).reshape(-1,)
        self.sector = sector
        self.y = y
        self.scores = {'HWtype':y, 'sector':sector}

        ## Processing the dataset for classfication (if exist)
        ## 如果传入了分类数据集的话，进行处理
        if len(train_class) > 0:
            self.X_train_class = train_class.loc[:, train_class.columns.isin(X)]
            self.y_train_bool = np.array(train_class.loc[:, train_class.columns == y]).reshape(-1,)
            self.y_test_bool = self.y_test.copy()
            self.y_test_bool[self.y_test_bool > 0] = 1

    def run_regress(self, params:pd.DataFrame):
        '''
        Fit the quantile regression model

        Args:
            params(pd.DataFrame): Best hyperparameters of cross validation
        Returns:
            regress_y(np.ndarray): Predicted mean value of each test data
            model(RandomForestQuantileRegressor): Fitted quantile regressor

        '''
        os.makedirs('log', exist_ok=True)
        save_fp = f'log/{self.y}_{self.sector}_regress.pkl'
        if not os.path.exists(save_fp):
            param = params.loc[
                (params.HWtype == self.y) & (params.sector == self.sector) & (params.task == 'regress')
            ]
            model = RandomForestQuantileRegressor(n_estimators=int(param['n_estimators']), 
                                                  min_samples_leaf = int(param['min_samples_leaf']),
                                                  max_features= int(param['max_features']),
                                                  random_state=42)
            
            # model = RandomForestRegressor(n_estimators = int(param['n_estimators']),
            #                               min_samples_leaf = int(param['min_samples_leaf']),
            #                               max_features = int(param['max_features']),
            #                               random_state = 42)
            model.fit(self.X_train, self.y_train)
            joblib.dump(model, save_fp)
        else:
            model = joblib.load(save_fp)

        regress_y = model.predict(self.X_test, 'mean')
        quantiles = model.predict(self.X_test, quantiles=[0.025, 0.5, 0.975])
        return regress_y, model

    def run_class_regress(self, params:pd.DataFrame):
        '''
        Fit the classification model and quantile regression model

        Args:
            params(pd.DataFrame): Best hyperparameters of cross validation
        Returns:
            class_y(np.ndarray): Predicted class of each test data
            regress_y(np.ndarray): Predicted mean value of each test data
        '''
        os.makedirs('log', exist_ok=True)
        save_fp = f'log/{self.y}_{self.sector}_class.pkl'
        if not os.path.exists(save_fp):
            param = params.loc[
                (params.HWtype == self.y) & (params.sector == self.sector) & (params.task == 'class')
            ]
            model = RandomForestClassifier(n_estimators = int(param['n_estimators']),
                                           min_samples_leaf = int(param['min_samples_leaf']),
                                           max_features = int(param['max_features']),
                                           random_state = 42)
            model.fit(self.X_train_class, self.y_train_bool)
            joblib.dump(model, save_fp)
        else:
            model = joblib.load(save_fp)
        class_y = model.predict(self.X_test)
        regress_y, _ = self.run_regress(params)
        return class_y, regress_y
    
    def run_quantitles(self, params:pd.DataFrame):
        '''
        Run quantile prediction of HW

        Args:
            params(pd.DataFrame): Best hyperparameters of cross validation
        Returns:
            None
        '''
        os.makedirs('quantiles', exist_ok=True)

        print (self.y, self.sector)

        file_name = f'{self.y}_{self.sector}_quantile.csv'
        _, model = self.run_regress(params)
        if self.y == 'HWsum':
            quantiles = model.predict(self.X_test, quantiles=[0.025, 0.5, 0.975])
            df = pd.DataFrame(quantiles, columns = ['0.025', '0.5', '0.975'])
            df['y_true'] = self.y_test

            df.to_csv(os.path.join('quantiles', file_name), index = False)
        else:
            class_y, regress_y = self.run_class_regress(params)
            self.y_pred = class_y * regress_y   ## class_y means whether generate HW17, like [0, 1, 1, 0...]
            
            quantiles = model.predict(self.X_test, quantiles=[0.025, 0.5, 0.975])
            df = pd.DataFrame(quantiles, columns = ['0.025', '0.5', '0.975'])
            df['y_true'] = self.y_test
            df['y_label'] = self.y_pred

            df = df[df['y_label'] != 0]
            df.drop(labels='y_label', axis = 1, inplace = True)

            print (df.head())
            df.to_csv(os.path.join('quantiles', file_name), index = False)


if __name__ == '__main__':
    os.chdir('./')
    summary = pd.DataFrame(columns=[
        'HWtype', 'sector', 'rsq', 'rmse', 'mae',
        'sse', 'acc', 'recall', 'precision', 'f1'])
    params = pd.read_csv('Params.csv')  ## Best hyperparameters for cross validation
    features = pd.read_csv('Features.csv')  ## Input features for each industrial sector

    ## Uncertainty analysis for all sectors and waste types
    for i, hwtype in enumerate(HWTYPE):
        print(f'Begin predict {hwtype} for each industrial sector...')
        for sector in tqdm(INDUSTRY[i]):
            train = pd.read_csv(f'Train/{hwtype}_{sector}_smogn.csv')
            test = pd.read_csv(f'Test/{hwtype}_{sector}_test.csv')
            if hwtype == 'HW17':
                train_class = pd.read_csv(f'Train/{hwtype}_{sector}_class.csv')
                train_class.rename(columns={'HW17logic':'HW17'}, inplace=True)
            feature = features.loc[(features.HWtype == hwtype) &
                                   (features.sector == sector)]
            feature = list(feature.iloc[0, 2:].dropna())
            if hwtype == 'HW17':
                HWpre = HWPrediction(train=train, test=test, X=feature, y=hwtype,
                                     sector=sector, train_class=train_class)
            else:
                HWpre = HWPrediction(train=train, test=test, X=feature, y=hwtype,
                                     sector=sector)
            HWpre.run_quantitles(params)
