'''
The Convolutional Support Estimator Network (CSEN) and
Compressive Learning CSEN (CL-CSEN) implementations.

The competing regrossor implementation: Support Vector Regressor (SVR).

Author: Mete Ahishali,
Tampere University, Tampere, Finland.
'''
import os
import numpy as np
import matplotlib as plt
import argparse
import scipy.io

os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.use('agg')
np.random.seed(10)

from  metrics import *

# INITIALIZATION
# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('--method', default='CSEN', 
                help="Method for the regression: CL-CSEN, CSEN, CL-CSEN-1D, CSEN-1D, SVR.")
ap.add_argument('--feature_type', help="Features extracted by the network (DenseNet121, VGG19, ResNet50).")
ap.add_argument('--isTrain', action='store_true', help="Train the model. ") 
ap.add_argument('--datadir', default='./data/CSENdata-2D/demo/', help="directory for input data")
ap.add_argument('--savedir', default='./result/demo/', help="directory for input data")
args = vars(ap.parse_args())
modelType = args['method'] # CL-CSEN, CSEN, and SVR.
feature_type = args['feature_type']
isTrain = args['isTrain']
dataDir = args['datadir']
savedir = args['savedir']

nuR = 1 # number of run times
MR = '0.5' # Measurement rate for CL-CSEN and CSEN approaches.


# For the results.
if not os.path.exists(savedir + modelType): os.makedirs(savedir + modelType)
if modelType == 'CL-CSEN':
    from cl_csen_regressor import model
    outName  = savedir + 'CL-CSEN/' + feature_type + '_mr_' + MR
elif modelType == 'CL-CSEN-1D':
    from cl_csen_1d_regressor import model
    outName  = savedir + 'CL-CSEN-1D/' + feature_type + '_mr_' + MR    
elif modelType == 'CSEN':
    from csen_regressor import model
    outName  = savedir + 'CSEN/' + feature_type + '_mr_' + MR    
elif modelType == 'CSEN-1D':
    from csen_1d_regressor import model
    outName  = savedir + 'CSEN-1D/' + feature_type + '_mr_' + MR    
elif modelType == 'SVR':
    from competing_regressor import svr as model
    outName  = savedir + modelType + '/' + feature_type + '_mr_' + MR    
 
 # Record weights.
weightsDir = 'weights/' + modelType + '/'
if not os.path.exists(weightsDir): os.makedirs(weightsDir)

for set in range(1, 1+nuR): # 5 different runs.
    
    # Init the model
    modelFold = model.model()
    
    modelFold.loadData(dataDir, feature_type, set, MR, isTrain)
    
    weightPath = weightsDir + feature_type + '_' + MR + '_' + str(set) + '.h5'
    
    if isTrain is True: modelFold.train(weightPath, epochs = 100, batch_size = 16)

    # Testing and performance evaluations.
    modelFold.load_weights(weightPath)
    
    y_pred = modelFold.predict()

    if 'metric' not in globals():
        metric = metrics(sets = np.array(list(range(nuR))),
                        test_size = len(modelFold.x_test))

    metric.compute(set, y_pred, modelFold.y_test)
    
    metric.display(set) # Print performances for this set.

    del modelFold

# Saving the results.
scipy.io.savemat(outName + '_res.mat',
    {'th':metric.th, 'ard':metric.ard, 
    'srd':metric.srd, 'rmse':metric.rmse, 'rmseLog':metric.rmseLog})

scipy.io.savemat(outName + '_predictions.mat',
    {'y_preds':metric.y_preds, 'y_trues':metric.y_tests})

