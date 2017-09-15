import sys
import numpy as np
import regLOO as cv
from OPS import OPS
from OPS import OPS_auto
from OPS import correlationCut as corrCut
from sklearn import preprocessing


if len(sys.argv) < 8 or len(sys.argv) > 9:
    print('Invalid number of arguments!')
    sys.exit()

datasetPath = sys.argv[1]
responsePath = sys.argv[2]
outputName = sys.argv[3]

try:
    corrTh = float(sys.argv[4])
except ValueError:
    print('Correlation threshold must be a float!')
    sys.exit()
try:
    maxVL_OPS = int(sys.argv[5])
except ValueError:
    print('maxVL_OPS must be an int!')
    sys.exit()
try:
    maxVL_Model = int(sys.argv[6])
except ValueError:
    print('maxVL_Model must be an int!')
    sys.exit()
try:
    maxVariables = int(sys.argv[7])
except ValueError:
    print('maxVariables must be an int!')
    sys.exit()

if len(sys.argv) > 8:
    nameInfoVec = sys.argv[8]
    if not (nameInfoVec == 'prod' or nameInfoVec == 'reg' or nameInfoVec == 'corr'):
        print('Invalid informative vector!')
        sys.exit()

try:
    X = np.loadtxt(datasetPath, delimiter='\t')
    Y = np.loadtxt(responsePath, delimiter='\t')
except Exception:
    print("Error loading dataset files!")
    sys.exit()

X = preprocessing.scale(X)

subset = corrCut(X, Y, corrTh)

if len(subset) == 0:
    print('No features with enough correlation!')
    sys.exit()
elif len(subset) < maxVariables:
    maxVariables = len(subset)
elif len(subset) < maxVL_OPS:
    maxVL_OPS = len(subset)

X = X[:, subset]
print('Processed dataset shape:', X.shape)
if len(sys.argv) > 8:
    sel, q2, NV, NVL_Model, NVL_OPS = OPS(X, Y, maxVL_OPS, maxVL_Model, maxVariables, infoVec=nameInfoVec, verbose=1)
    q2, r2, rmsecv, rmse, corrcv, corrmdl = cv.plsLOO(X, Y, int(NVL_Model))
    output = np.array([q2, r2, rmsecv, rmse, corrcv, corrmdl,int(NV), int(NVL_Model), int(NVL_OPS)]) 
else:
    sel, q2, NV, NVL_Model, NVL_OPS, infoVec = OPS_auto(X, Y, maxVL_OPS, maxVL_Model, maxVariables, verbose=1)
    q2, r2, rmsecv, rmse, corrcv, corrmdl = cv.plsLOO(X, Y, int(NVL_Model))
    output = np.array([q2, r2, rmsecv, rmse, corrcv, corrmdl, int(NV), int(NVL_Model), int(NVL_OPS), int(infoVec)])

output = np.reshape(output, (1, -1))
np.savetxt(outputName + '_metrics.txt',output, delimiter=',', fmt='%f')

subset = np.array(subset)
subset = subset[sel]
subset = np.reshape(subset, (1, -1))

np.savetxt(outputName + '_selectedFeatures.txt',subset, delimiter=',', fmt='%f')