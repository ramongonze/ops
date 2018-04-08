import sys
import numpy as np
import regLOO as cv
from OPS import OPS
from OPS import OPS_auto
from OPS import correlationCut as corrCut
from sklearn import preprocessing

if len(sys.argv) < 7 or len(sys.argv) > 8:
    print('Invalid number of arguments!')
    sys.exit()

datasetPath = sys.argv[1]
responsePath = sys.argv[2]

try:
    corrTh = float(sys.argv[3])
except ValueError:
    print('Correlation threshold must be a float!')
    sys.exit()
try:
    NVL_OPS = int(sys.argv[4])
except ValueError:
    print('NVL_OPS must be an int!')
    sys.exit()
try:
    maxVL_Model = int(sys.argv[5])
except ValueError:
    print('maxVL_Model must be an int!')
    sys.exit()
try:
    maxVariables = int(sys.argv[6])
except ValueError:
    print('maxVariables must be an int!')
    sys.exit()

if len(sys.argv) > 7:
    nameInfoVec = sys.argv[7]
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
elif len(subset) < NVL_OPS:
    NVL_OPS = len(subset)

X = X[:, subset]

if len(sys.argv) > 7:
    sel, q2, NV, NVL_Model, NVL_OPS = OPS(X, Y, NVL_OPS, maxVL_Model, maxVariables, infoVec=nameInfoVec, verbose=0)
    text = '{:.64f},{:d},{:d},{:d}'.format(float(q2), int(NV), int(NVL_Model), int(NVL_OPS))
else:
    sel, q2, NV, NVL_Model, NVL_OPS, infoVec = OPS_auto(X, Y, NVL_OPS, maxVL_Model, maxVariables, verbose=0)
    text = '{:.64f},{:d},{:d},{:d},{:d}'.format(float(q2), int(NV), int(NVL_Model), int(NVL_OPS), int(infoVec))

for i in sel:
    text = text + ',{:d}'.format(i)
text = text + '\n'
output = open('transition.txt', 'a')
output.write(text)
output.close()
