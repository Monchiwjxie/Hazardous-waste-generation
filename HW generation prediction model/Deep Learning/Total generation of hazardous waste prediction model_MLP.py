import os
import joblib
import math
import pandas as pd
import time
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV


class MLPEnsemble:
    """
    An ensemble framework for multi-layer perceptron. In the training phase, several MLPs
    are trained independently under the control of different random seeds. In the inference
    phase, the results of all estimators are integrated as output using the bagging method.

    Args:
        ensemble_num(int): Number of estimators
        train_set(pd.DataFrame): Train data set for classification
        train_reg(pd.DataFrame): Train data set for regression
        test_set(pd.DataFrame): Test data set
        feature_target(list): list[0]: features(list), list[1]: target(str)
        params(dict): Optional hyperparameters list
    """

    def __init__(self, ensemble_num:int, train_set:pd.DataFrame, test_set:pd.DataFrame,
                 feature_target:list, params:dict)->None:
        self.ensemble_num = ensemble_num
        self.train_set = train_set
        self.test_set = test_set
        self.feature = feature_target[0]
        self.target = feature_target[1]
        self.params = params
        self.model = []
        self.run()

    def preprocess(self)->None:
        """
        Data sets split and normalization

        Returns:
            None
        """
        self.y_train = self.train_set.loc[:, self.target]
        self.y_test = self.test_set.loc[:, self.target]
        self.X_train = self.train_set.loc[:, self.train_set.columns.isin(self.feature)]
        self.X_test = self.test_set.loc[:, self.test_set.columns.isin(self.feature)]
        X_scaler = StandardScaler()
        self.X_train = X_scaler.fit_transform(self.X_train)
        self.X_test = X_scaler.transform(self.X_test)

        #y_scaler = MinMaxScaler()
        #self.y_train = y_scaler.fit_transform(self.y_train.reshape(-1, 1))
        #self.y_test = y_scaler.transform(self.y_test.reshape(-1, 1))
        #self.y_scaler = y_scaler


    def gsCV(self)->None:
        """
        Grid search cross validation to find the best hyperparameters

        Returns:
            None
        """
        model = MLPRegressor()
        model_gs = GridSearchCV(
            estimator=model,
            param_grid=self.params,
            cv=10,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )
        model_gs.fit(self.X_train, self.y_train)
        print('Best: %f using %s' % (model_gs.best_score_, model_gs.best_params_))
        self.best_params = model_gs.best_params_

    def train(self)->None:
        """
        Train n MLPs respectively. If there is already a model file locally,
        training will be skipped and the model will be loaded directly.

        Returns:
            None
        """
        for i in range(self.ensemble_num):
            print(f'{i+1}th model is training...')
            seed = self.params.get('random_state')[0]+i
            save_fp = os.path.join('log', f'{self.target}_seed{seed}_MLP.pkl')
            os.makedirs('log', exist_ok=True)
            if not os.path.exists(save_fp):
                mlp = MLPRegressor(hidden_layer_sizes=self.best_params.get('hidden_layer_sizes'),
                                   activation=self.best_params.get('activation'),
                                   solver=self.best_params.get('solver'),
                                   learning_rate_init=self.best_params.get('learning_rate_init'),
                                   max_iter=self.best_params.get('max_iter'),
                                   random_state=seed,
                                   verbose=1)
                mlp.fit(self.X_train, self.y_train)
                joblib.dump(mlp, save_fp)
            else:
                print(f'{i+1}th model has been trained before')
                mlp = joblib.load(save_fp)
            self.model.append(mlp)

    def test(self):
        """
        Inferring on the test data set and evaluate the model performance

        Returns:
            None
        """
        print("model is testing...")

        self.y_pred = self.model[0].predict(self.X_test)
        for i in range(1, self.ensemble_num):
            self.y_pred += self.model[i].predict(self.X_test)
        self.y_pred /= self.ensemble_num

        self.train_pred = self.model[0].predict(self.X_train)
        for i in range(1, self.ensemble_num):
            self.train_pred += self.model[i].predict(self.X_train)
        self.train_pred /= self.ensemble_num

        result = pd.DataFrame({'truth': self.y_train, 'response': self.train_pred})
        result.to_csv(f"output/{self.target}_{self.ensemble_num}_train.csv")

        result = pd.DataFrame({'truth':self.y_test, 'response':self.y_pred})
        result.to_csv(f"output/{self.target}_{self.ensemble_num}_models.csv")

    def run(self):
        """
        Run the framework

        Returns:
            None
        """
        self.preprocess()
        print("begin gsCV")
        self.gsCV()
        self.train()
        self.test()

if __name__ == "__main__":
    ## Optional hyperparameters
    params = {
        'hidden_layer_sizes':[(512,512,256), (512,256), (256,256,128)],
        'activation':['relu', 'tanh'],
        'solver':['adam'],
        'learning_rate_init':[.001,.005,.01],
        'max_iter':[150],
        'random_state':[43]
    }

    ## input features
    feature = ['industry', 'staff', 'process_1', 'process_2', 'process_3', 'process_4',
               'process_5', 'process_6', 'process_7', 'process_8','process_9',
               'process_10', 'process_11', 'COD', 'pH', 'water', 'N', 'P', 'NH3N',
               'Cr6', 'Cr', 'Fe', 'Cu', 'Ni', 'Zn']

    HWtype = 'HWsum'
    train = pd.read_csv(f"data/{HWtype}_combined_train.csv")
    test = pd.read_csv(f"data/{HWtype}_combined_test.csv")
    MLPEnsemble(5, train, test, [feature, HWtype], params)