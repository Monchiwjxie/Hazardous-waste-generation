import os
import time
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
tf.config.run_functions_eagerly(True) # run in eager mode
tf.data.experimental.enable_debug_mode()
from TabularNN17 import *

if __name__ == "__main__":
    start = time.time()
    class Config():
        seed = 42  # not like seed_everthing in torch
        num_classes = 1
        epochs = 200
        lr = [1e-3, 5e-3]
        batch_size = [4096, 2048]
        patience = 20
        mlp_list = [[512, 512, 128], [256, 256, 128]]
        dropout = [0.2, 0.2]
    config = Config()
    feature_cat = ['industry', 'staff', 'process_1', 'process_2', 'process_3',
                   'process_4','process_5',
                   'process_6', 'process_7',
                   'process_8','process_9','process_10', 'process_11']
    feature_dense = ['COD','pH','water', 'N', 'P', 'NH3N',
        'Cr6', 'Cr',
        'Fe', 'Cu',
        'Ni', 'Zn'
        ]
    HWtype = 'HWsum'
    train = pd.read_csv(f"data/{HWtype}_combined_smogn.csv")
    test = pd.read_csv(f"data/{HWtype}_combined_test.csv")
    #X_train, X_test, y_train, y_test = preprocess(train, test, feature, HWtype)
    y_train = train.loc[:, HWtype].values
    y_test = test.loc[:, HWtype].values

    X_scaler = MinMaxScaler()
    X_train_cat = train.loc[:, train.columns.isin(feature_cat)].values
    X_test_cat = test.loc[:, test.columns.isin(feature_cat)].values
    X_train_dense = train.loc[:, train.columns.isin(feature_dense)]
    X_test_dense = test.loc[:, test.columns.isin(feature_dense)]
    X_train_dense = X_scaler.fit_transform(X_train_dense)
    X_test_dense = X_scaler.transform(X_test_dense)

    X_train = np.hstack([X_train_cat, X_train_dense])
    X_test = np.hstack([X_test_cat, X_test_dense])

    X_valid, y_valid = X_test, y_test
    cat_idxs, cat_dims, dense_idxs = [i for i in range(13)], [4420, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [13+i for i in range(12)]
    #cat_idxs, cat_dims, dense_idxs = [i for i in range(10)], [4420, 6, 2, 2, 2, 2, 2, 2, 2, 2], [10+i for i in range(5)]
    cv_regr = GSCV(train_data=(X_train, y_train), params_list=config,
                   cat_idxs=cat_idxs, cat_dims=cat_dims, dense_idxs=dense_idxs)
    regr_idx = cv_regr.run()
    #X_train, y_train, X_valid, y_valid, X_test, y_test, cat_idxs, cat_dims, dense_idxs = get_fake_dataset(seed=config.seed)
        # get_cifar10_dataset(seed=config.seed, one_hot=True, select_class=config.num_classes)
    _, model = simple_train(train_data=(X_train, y_train), validation_data=(X_test, y_test),
                            input_dim=X_train.shape[1], config=config,
                            cat_idxs=cat_idxs, cat_dims=cat_dims, option_idx=regr_idx, dense_idxs=dense_idxs)
    y_pred_train = model.predict(X_train).reshape(-1, )
    y_pred = model.predict(X_test).reshape(-1,)
    print("Train R2 is {:.3f}".format(metrics.r2_score(y_train, y_pred_train)))
    print("Test R2 is {:.3f}".format(metrics.r2_score(y_test, y_pred)))
    result_train = pd.DataFrame({'truth': y_train, 'response': y_pred_train})
    result_train.to_csv(f"{HWtype}_tabular_nn_train.csv")

    result_test = pd.DataFrame({'truth': y_test, 'response': y_pred})
    result_test.to_csv(f"{HWtype}_tabular_nn_test.csv")
    total = time.time()-start
    print(f'total time:{total}s')