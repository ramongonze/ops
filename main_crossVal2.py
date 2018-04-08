import sys
import numpy as np
import regLOO as cv
from OPS import correlationCut as corrCut
from sklearn import preprocessing


if len(sys.argv) < 5 or len(sys.argv) > 6:
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

if len(sys.argv) > 5:
    nameInfoVec = sys.argv[5]
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

X = X[:, subset]
print('\nProcessed dataset shape:', X.shape)

# Get the results from OPS or OPS_auto
data = open('transition.txt', 'r')
results = data.readlines()
data.close()
values = [] # Contains all the "q2's"
for i in range(len(results)):
    results[i] = results[i][:-1]
    results[i] = results[i].split(',')
    values.append(float(results[i][0])) 

bmi = values.index(max(values)) # Best Model Index

q2, NV, NVL_Model, NVL_OPS = float(results[bmi][0]), int(results[bmi][1]), int(results[bmi][2]), int(results[bmi][3])
if len(sys.argv) > 5:
    start = 4
else:
    infoVec = int(results[bmi][4])
    start = 5

sel = []
for i in range(start, len(results[bmi])):
    sel.append(int(results[bmi][i]))
sel = np.array(sel)

q2, r2, rmsecv, rmse, corrcv, corrmdl = cv.plsLOO(X[:,sel], Y, int(NVL_Model))

if len(sys.argv) > 5:
    output = np.array([q2, r2, rmsecv, rmse, corrcv, corrmdl,int(NV), int(NVL_Model), int(NVL_OPS)]) 
    print('Best model found:', 'Q2:', '{0:.4f}'.format(q2), 'NV:', int(NV), 'NVL_Model:', int(NVL_Model), 'NVL_OPS:', int(NVL_OPS))
else:
    output = np.array([q2, r2, rmsecv, rmse, corrcv, corrmdl, int(NV), int(NVL_Model), int(NVL_OPS), int(infoVec)])
    print('Best model found:', 'Q2:', '{0:.4f}'.format(q2), 'NV:', int(NV), 'NVL_Model:', int(NVL_Model), 'NVL_OPS:', int(NVL_OPS), 'InfoVec', int(infoVec))
    
output = np.reshape(output, (1, -1))
np.savetxt(outputName + '_metrics.txt',output, delimiter=',', fmt='%f')


subset = np.array(subset)
subset = subset[sel]
subset = np.reshape(subset, (1, -1))

np.savetxt(outputName + '_selectedFeatures.txt',subset, delimiter=',', fmt='%f')