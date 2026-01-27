import pandas as pd
#import torch
#from dagma import utils
from dagma.linear import DagmaLinear
#from dagma.nonlinear import DagmaMLP, DagmaNonlinear
#from sklearn.preprocessing import MinMaxScaler
#from castle.algorithms import PC, Notears, NotearsNonlinear
#from castle.common import GraphDAG
#import pygraphviz as pgv
from DAG import DAGNodesClassification
from MarkovBlanket import  MarkovBlanket
import numpy as np
import os

#industry = 'combined'
HWtype = 'HWsum'
for industry in ['combined', 2614, 2631, 2669, 3130, 3340, 3360, 3982, 4417, 4419]:
    """ Load data """

    df_train_data = pd.read_csv(f'data/{industry}_train_smogn.csv')
    df_train_data = df_train_data.iloc[:,1:]
    all_zero_columns = df_train_data.apply(lambda x:all(x == 0))
    df_train_data = df_train_data.drop(df_train_data.columns[all_zero_columns], axis = 1)
    # df_train_data.drop(labels=['process_4', 'process_6', 'process_8', 'process_9',
    #                            'process_16','process_17','process_18'], axis = 1, inplace=True)
    #df_train_data['HW17'] = df_train_data['HW17'].apply(lambda x: 1 if x > 0 else 0)
    print(df_train_data.shape)
    X = df_train_data.to_numpy()

    ## nonlinear
    # d = df_train_data.shape[1]
    # eq_model = DagmaMLP(dims=[d, 20, 1], bias=True, dtype=torch.double) # create the model for the structural equations, in this case MLPs
    # model = DagmaNonlinear(eq_model, dtype=torch.double) # create the model for DAG learning
    # W_est = model.fit(X, lambda1=0.02, lambda2=0.005) # fit the model with L1 reg. (coeff. 0.02) and L2 reg. (coeff. 0.005)
    # np.save('W_est_nonlinear', W_est)
    # print (W_est)

    ## PC algorithm
    # pc = PC()
    # pc.learn(X)
    # print (pc.causal_matrix)


    #include_edges = ((0, 32), (22, 32))

    ## Linear
    node_labels = list(df_train_data.columns)
    #print(node_labels)

    """ Fit a linear model for DAG """

    if not os.path.exists(f'npynew/{HWtype}_{industry}.npy'):
        model = DagmaLinear('l2') # create a linear model with least squares loss
        W_est = model.fit(X, lambda1=0.02, w_threshold=.5)# fit the model with L1 reg. (coeff. 0.02)
        np.save(f'npynew/{HWtype}_{industry}', W_est)
    else:
        W_est = np.load(f'npynew/{HWtype}_{industry}.npy')

    np.fill_diagonal(W_est, 0)

    """ Visualize the graph """
    G = DAGNodesClassification(W_est, node_labels=node_labels)
    G.visualize_graph(f'DAGnew/{HWtype}_{industry}_DAG')

    """ Solve the Markov Blanket for target variable """
    mb = MarkovBlanket(node_labels, W_est)
    mb.dfmtx.to_csv(f'npynew/{HWtype}_{industry}_mtx.csv')
    mbset = mb.getMB(f'{HWtype}')
    with open(f'MBnew/{industry}_{HWtype}_MB.txt','w') as f:
       for item in mbset:
           f.write(str(item) +'\n')
