__author__ = 'viktor'
import pandas as pd
import numpy as np
from SAMKNN import SAMKNN
from sklearn.metrics import accuracy_score
from ClassifierVisualizer import ClassifierVisualizer
from ClassifierListener import DummyClassifierListener
import logging
import sys
from data_loader import DataLoader

def run(X, y, hyperParams, visualize=False):
    """
    Test function for SAMKNN
    """
    if visualize:
        visualizer = ClassifierVisualizer(X, y, drawInterval=200, datasetName='Moving Squares')
    else:
        visualizer = DummyClassifierListener()
    classifier = SAMKNN(n_neighbors=hyperParams['nNeighbours'], maxSize=hyperParams['maxSize'],
                        knnWeights=hyperParams['knnWeights'], recalculateSTMError=hyperParams['recalculateSTMError'],
                        useLTM=hyperParams['useLTM'], listener=[visualizer], metric=hyperParams['metric'],
                        metric_step=hyperParams['metric_step'])

    logging.info('applying model on dataset')
    predictedLabels, complexity, complexityNumParameterMetric = classifier.trainOnline(X, y, np.unique(y))
    accuracy = accuracy_score(y, predictedLabels)
    logging.info('error rate %.2f%%' % (100-100*accuracy))

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    hyperParams ={'maxSize': 1000, 'nNeighbours': 3, 'knnWeights': 'distance', 'recalculateSTMError': False,
                  'useLTM': True, 'metric': 'LMNN', 'metric_step': 100}
    #hyperParams = {'windowSize': 5000, 'nNeighbours': 5, 'knnWeights': 'distance', 'STMSizeAdaption': None,
    #               'useLTM': False}


    logging.info('loading dataset')
    #X=pd.read_csv('../data/driftDatasets/artificial/chess/transientChessboard.data', sep=',', header=None).values
    #y=pd.read_csv('../datasets/NEweather_class.csv', sep=',', header=None, dtype=np.int8).values.ravel()
    #X = np.loadtxt('../datasets/movingSquares.data')
    #y = np.loadtxt('../datasets/movingSquares.labels', dtype=np.uint8)

    data = sys.argv[1]
    loader = DataLoader("../../")
    print("Data Set: " + data)
    if data == "electricity":
        X, y = loader.load_electricity()
    elif data == "outdoor":
        X, y = loader.load_outdoor()
    elif data == "poker":
        X, y = loader.load_poker()
    elif data == "rialto":
        X, y = loader.load_rialto()
    elif data == "chess":
        X, y = loader.load_chess()
    elif data == "hyperplane":
        X, y = loader.load_hyperplane()
    elif data == "mixed_drift":
        X, y = loader.load_mixed_drift()
    elif data == "moving_squares":
        X, y = loader.load_moving_squares()
    elif data == "interchanging_rbf":
        X, y = loader.load_interchanging_rbf()
    elif data == "moving_rbf":
        X, y = loader.load_moving_rbf()

    logging.info('%d samples' % X.shape[0])
    logging.info('%d dimensions' % X.shape[1])
    run(X, y, hyperParams, visualize=False)

