import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
tf.config.run_functions_eagerly(True) # run in eager mode
tf.data.experimental.enable_debug_mode()
from typing import Union, Optional, Tuple, Any
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, concatenate, Lambda, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import pandas as pd

def create_Classifer(input_dim: int, mlp_list: list, output_dim: int,
              cat_idxs: list = None, cat_dims: list = None, cat_emb_dim: int = None, dense_idxs: list = None,
              dropout_list: list = None, lr: float = None):
    all_inputs = Input(shape=(input_dim,))
    # Process dense features
    if dense_idxs is not None and len(dense_idxs) != 0:
        dense_features = tf.gather(all_inputs, dense_idxs, axis=1)
    else:
        dense_features = all_inputs
    # Process categorical features with embedding layers
    cat_features = []
    if cat_idxs is not None and cat_dims is not None and cat_emb_dim is not None:
        for i, idx in enumerate(cat_idxs):
            cat_input = tf.gather(all_inputs, [idx], axis=1)
            cat_emb = Embedding(cat_dims[i], cat_emb_dim)(cat_input)
            cat_emb = Flatten()(cat_emb)
            cat_features.append(cat_emb)

    # Concatenate categorical and dense features
    if cat_features:
        combined_features = concatenate([dense_features] + cat_features)
    else:
        combined_features = dense_features

    x = combined_features
    for units, dropout in zip(mlp_list, dropout_list):
        x = Dense(units, activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dropout(dropout)(x)

    output = Dense(output_dim, activation='softmax')(x)
    #output=Dense(output_dim, activation='relu')(x)
    model = Model(inputs=all_inputs, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                 loss=tf.losses.categorical_crossentropy,
                 metrics=['accuracy'])
                #loss=tf.losses.mean_squared_error,
                #metrics='mean_squared_error')
    return model


def create_NN(input_dim: int, mlp_list: list, output_dim: int,
              cat_idxs: list = None, cat_dims: list = None, cat_emb_dim: int = None, dense_idxs: list = None,
              dropout_list: list = None, lr: float = None):
    all_inputs = Input(shape=(input_dim,))
    # Process dense features
    if dense_idxs is not None and len(dense_idxs) != 0:
        dense_features = tf.gather(all_inputs, dense_idxs, axis=1)
    else:
        dense_features = all_inputs
    # Process categorical features with embedding layers
    cat_features = []
    if cat_idxs is not None and cat_dims is not None and cat_emb_dim is not None:
        for i, idx in enumerate(cat_idxs):
            cat_input = tf.gather(all_inputs, [idx], axis=1)
            cat_emb = Embedding(cat_dims[i], cat_emb_dim)(cat_input)
            cat_emb = Flatten()(cat_emb)
            cat_features.append(cat_emb)

    # Concatenate categorical and dense features
    if cat_features:
        combined_features = concatenate([dense_features] + cat_features)
    else:
        combined_features = dense_features

    x = combined_features
    for units, dropout in zip(mlp_list, dropout_list):
        x = Dense(units, activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dropout(dropout)(x)

    #output = Dense(output_dim, activation='softmax')(x)
    output=Dense(output_dim, activation='relu')(x)
    model = Model(inputs=all_inputs, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                 #loss=tf.losses.categorical_crossentropy,
                 #metrics=['accuracy'])
                loss=tf.losses.mean_squared_error,
                metrics='mean_squared_error')
    return model

def simple_train(train_data: Tuple[np.array, np.array], validation_data: Tuple[np.array, np.array],
                 input_dim: int, config: Any, cat_idxs: list = None, cat_dims: list = None,
                 cat_emb_dim: int = 1, dense_idxs: list = None, option_idx:list = None, type:str = "regr"):
    if cat_idxs is not None and len(cat_idxs) != 0:
        assert len(cat_idxs) == len(cat_dims)
        assert min(cat_idxs) >= 0 and max(cat_idxs) < train_data[0].shape[1]
        #assert max(dense_idxs) < train_data[0].shape[1]
    seed = config.seed
    X_train, y_train = train_data
    X_valid, y_valid = validation_data
    tf.random.set_seed(seed)
    if type == "class":
        model = create_Classifer(input_dim=input_dim, mlp_list=config.mlp_list[option_idx[2]], output_dim=config.num_classes,
                          cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
                          dense_idxs=dense_idxs, dropout_list=config.dropout, lr=config.lr[option_idx[0]])
    else:
        model = create_NN(input_dim=input_dim, mlp_list=config.mlp_list[option_idx[2]], output_dim=config.num_classes,
                          cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
                          dense_idxs=dense_idxs, dropout_list=config.dropout, lr=config.lr[option_idx[0]])
    print(model.summary())

    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=config.epochs,
                        batch_size=config.batch_size[option_idx[1]], verbose=1)
    return history, model

class GSCV:

    def __init__(self, train_data,params_list,cat_idxs, cat_dims, dense_idxs , task='regr'):
        self.params_list = params_list
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.dense_idxs = dense_idxs

        assert task in ('class', 'regr')
        self.task = task
        self.X_train, self.y_train = train_data
        self.opt_params = []
        self.performance = []

    def option_params(self):
        for i in range(len(self.params_list.lr)):
            for j in range(len(self.params_list.batch_size)):
                for k in range(len(self.params_list.mlp_list)):
                    self.opt_params.append([i,j,k])

    def CV(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=self.params_list.seed)
        for i in range(len(self.opt_params)):
            perfermance_per_fold = []
            for train_idx, test_idx in kf.split(self.X_train):
                the_X_train = self.X_train[train_idx]
                the_X_test = self.X_train[test_idx]
                the_y_train = self.y_train[train_idx]
                the_y_test = self.y_train[test_idx]
                history, model = simple_train(train_data = (the_X_train, the_y_train), validation_data = (the_X_test, the_y_test),
                                     input_dim = the_X_train.shape[1], config = self.params_list,cat_idxs = self.cat_idxs,
                                     cat_dims = self.cat_dims, dense_idxs = self.dense_idxs,option_idx=self.opt_params[i], type = self.task)
                perfermance_per_fold.append(history.history.get('val_loss')[-1])
            self.performance.append(np.mean(perfermance_per_fold))
            print(f'Params is lr={self.params_list.lr[self.opt_params[i][0]]}')
            print(f'batch_size={self.params_list.batch_size[self.opt_params[i][1]]}')
            print(f'mlp={self.params_list.mlp_list[self.opt_params[i][2]]}')
            print(f'Val mean loss is {self.performance[-1]}')
        self.best_params = self.opt_params[np.argmin(self.performance)]
        print(f'Best model is {self.best_params}')

    def run(self)->list:
        self.option_params()
        self.CV()
        return self.best_params


if __name__ == "__main__":
    class Config_class():
        seed = 42  # not like seed_everthing in torch
        num_classes = 2
        epochs = 150
        lr = [1e-3, 5e-3]
        batch_size = [4096, 2048]
        patience = 20
        mlp_list = [[512,512,128] , [256, 256, 128]]
        dropout = [0.2, 0.2]

    class Config_regr():
        seed = 42  # not like seed_everthing in torch
        num_classes = 1
        epochs =200
        lr = [1e-3, 5e-3]
        batch_size = [4096, 2048]
        patience = 20
        mlp_list = [[512,512,128] ,[256,  256, 128]]
        dropout = [0.2, 0.2]


    config_class = Config_class()
    config_regr = Config_regr()


    feature_cat = ['industry', 'staff', 'process_1', 'process_2', 'process_3',
                   'process_4','process_5','process_6', 'process_7',
                   'process_8','process_9','process_10', 'process_11']
    #feature_cat = ['industry', 'staff', 'process_1', 'process_2', 'process_3', 'process_4',
    #               'process_8', 'process_9', 'process_10', 'process_11']
    #feature_cat = ['industry', 'staff', 'process_1', 'process_2', 'process_4', 'process_6',
    #               'process_8', 'process_9', 'process_10', 'process_11']
    feature_dense = ['COD', 'pH', 'water', 'N', 'P', 'NH3N',
                     'Cr6', 'Cr', 'Fe', 'Cu', 'Ni', 'Zn']
    #feature_dense = ['COD', 'pH', 'water', 'Cu', 'Fe', 'Ni', 'Zn']
    #feature_dense = ['COD', 'pH', 'water', 'N', 'P', 'NH3N',
    #                 'Cr6', 'Cr', 'Fe', 'Cu', 'Ni', 'Zn']

    HWtype = 'HW17'
    train = pd.read_csv(f"data/{HWtype}_combined_smogn.csv")
    train_regr = pd.read_csv(f"data/{HWtype}_combined_regr.csv")
    test = pd.read_csv(f"data/{HWtype}_combined_test.csv")
    #X_train, X_test, y_train, y_test = preprocess(train, test, feature, HWtype)
    y_train = train.loc[:, HWtype].apply(lambda x: 1 if x > 0 else 0).values
    y_target = test.loc[:, HWtype].apply(lambda x: 1 if x > 0 else 0).values
    y_regr = train_regr.loc[:, HWtype].values
    y_test = test.loc[:, HWtype].values

    X_scaler = MinMaxScaler()
    X_train_cat = train.loc[:, train.columns.isin(feature_cat)].values
    X_test_cat = test.loc[:, test.columns.isin(feature_cat)].values
    X_regr_cat = train_regr.loc[:, train_regr.columns.isin(feature_cat)].values

    if feature_dense != []:
        X_train_dense = train.loc[:, train.columns.isin(feature_dense)]
        X_test_dense = test.loc[:, test.columns.isin(feature_dense)]
        X_regr_dense = train_regr.loc[:, train_regr.columns.isin(feature_dense)]

        X_train_dense = X_scaler.fit_transform(X_train_dense)
        X_test_dense = X_scaler.transform(X_test_dense)
        X_regr_dense = X_scaler.transform(X_regr_dense)

        X_train = np.hstack([X_train_cat, X_train_dense])
        X_test = np.hstack([X_test_cat, X_test_dense])
        X_regr = np.hstack([X_regr_cat, X_regr_dense])
    else:
        X_train = X_train_cat
        X_test = X_test_cat
        X_regr = X_regr_cat


    cat_idxs, cat_dims, dense_idxs = [i for i in range(13)], [4420, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [13+i for i in range(12)]
    #cat_idxs, cat_dims, dense_idxs = [i for i in range(10)], [4420, 6, 2, 2, 2, 2, 2, 2, 2, 2], [10+i for i in range(7)]
    y_train_cp = tf.keras.utils.to_categorical(y_train, 2)
    y_target_cp = tf.keras.utils.to_categorical(y_target, 2)

    cv_class = GSCV(train_data=(X_train, y_train_cp), params_list=config_class,
                    cat_idxs=cat_idxs, cat_dims=cat_dims, dense_idxs=dense_idxs,task='class')
    class_idx = cv_class.run()


    cv_regr = GSCV(train_data=(X_regr, y_regr), params_list=config_regr,
                    cat_idxs=cat_idxs, cat_dims=cat_dims, dense_idxs=dense_idxs)
    regr_idx = cv_regr.run()


    _,model_class = simple_train(train_data=(X_train, y_train_cp), validation_data=(X_test, y_target_cp),
                         input_dim=X_train.shape[1], config=config_class,
                         cat_idxs=cat_idxs, cat_dims=cat_dims, dense_idxs=dense_idxs,option_idx= class_idx, type="class")

    _,model_regr = simple_train(train_data=(X_regr, y_regr), validation_data=(X_test, y_test),
                         input_dim=X_train.shape[1], config=config_regr,
                         cat_idxs=cat_idxs, cat_dims=cat_dims,option_idx= regr_idx, dense_idxs=dense_idxs)

    class_pred_train = np.argmax(model_class.predict(X_train),axis=1).reshape(-1,)
    class_pred_test = np.argmax(model_class.predict(X_test),axis=1).reshape(-1,)
    regr_pred_train = model_regr.predict(X_regr).reshape(-1,)
    regr_pred_test = model_regr.predict(X_test).reshape(-1,)
    regr_pred_test *= class_pred_test



    print("Train Acc is {:.3f}".format(metrics.accuracy_score(y_train, class_pred_train)))
    print("Test Acc is {:.3f}".format(metrics.accuracy_score(y_target, class_pred_test)))

    print("Train R2 is {:.3f}".format(metrics.r2_score(y_regr, regr_pred_train)))
    print("Test R2 is {:.3f}".format(metrics.r2_score(y_test, regr_pred_test)))

    result_train = pd.DataFrame({'truth': y_train, 'response': class_pred_train})
    result_train.to_csv(f"output/{HWtype}_tabular_nn_train_class.csv")

    result_test = pd.DataFrame({'truth': y_target, 'response': class_pred_test})
    result_test.to_csv(f"output/{HWtype}_tabular_nn_test_class.csv")

    result_train = pd.DataFrame({'truth': y_regr, 'response': regr_pred_train})
    result_train.to_csv(f"output/{HWtype}_tabular_nn_train_regr.csv")

    result_test = pd.DataFrame({'truth': y_test, 'response': regr_pred_test})
    result_test.to_csv(f"output/{HWtype}_tabular_nn_test_regr.csv")