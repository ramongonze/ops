import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
import multiprocessing


def correlation(x, y):

    my = np.mean(y)
    mx = np.mean(x)
    c = (y - my).T.dot(x - mx)
    sigy = np.sqrt(np.sum((y - my)**2))
    sigx = np.sqrt(np.sum((x - mx)**2))
    if sigy == 0 or sigx == 0:
        return 0.0
    p = c / (sigy * sigx)

    return p

X = []
Y = []
numVL = 0

def cvStep(cvIndex):
    
    global X, Y, numVL
    train_index = cvIndex[0]
    test_index = cvIndex[1]
    
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    mdl = PLSRegression(n_components=numVL).fit(X_train, Y_train)
    coefs = mdl.coef_
    intercept = mdl.y_mean_ - np.dot(mdl.x_mean_, mdl.coef_)
    trainFit = X_train.dot(coefs) + intercept
    testFit = X_test.dot(coefs) + intercept

    return (testFit, mean_squared_error(Y_train, trainFit), mean_squared_error(Y_test, testFit), r2_score(Y_train, trainFit), correlation(Y_train, trainFit))

def plsLOO(X_in, Y_in, nVL, n_jobs=multiprocessing.cpu_count()):

    global X, Y, numVL
    X = X_in
    Y = Y_in
    numVL = nVL
    """
    Parallel Leave-one-out cross-validation with Partial Least Squares(PLS)

    Parameters
    ----------
    X_in : numpy array of shape [n_samples,n_features]
        Samples.
    Y_in : numpy array of shape [n_samples]
        Target values.
    nVL: int
        Number of latent variables for PLS.
    n_jobs: int
        Number of jobs

    Returns
    -------
    q2 : float
        Q^2 external validation metric. For details see reference [1]
    r2 : float
        mean of R-squared obtained in each trained model.
    rmsecv : float
        mean of root mean squared error (RSME) obtained in each testing phase.
    rmse : float
        mean of root mean squared error (RSME) obtained in each trained model.
    corrcv : float
        Pearson correlation between target values and fitted values with cross-validation.
    corrmdl : float
        mean of pearson correlation between target values and fitted values obtained in each trained model.       
    
    References
    ----------
        [1] Martins, J. P. A.; Barbosa, E. G.; Pasqualoto, K. F. M. & Ferreira, M. M. C. (2009).
        LQTA-QSAR: a new 4D-QSAR methodology. 
        Journal of Chemical Information and Modeling, 49(6):1428--1436. ISSN 1549-9596

    """

    nSamples, nFeatures = np.shape(X)
    yFitCV = np.zeros((nSamples, 1))
    rmseModel = np.zeros((nSamples, 1))
    rmseCV = np.zeros((nSamples, 1))
    r2Model = np.zeros((nSamples, 1))
    corrModel = np.zeros((nSamples, 1))
    cv = LeaveOneOut()
   

    if numVL > nFeatures:
        print("Number of latent variables must be less than or equal to the number of features!")
        numVL = nFeatures
    else:
        numVL = nVL

    metrics = []
    with multiprocessing.Pool(processes=n_jobs) as pool:
        metrics = pool.map(cvStep, cv.split(X))
        pool.close()
        pool.join()

    cursor = 0
    for metric in metrics:
        yFitCV[cursor] = metric[0]
        rmseModel[cursor] = metric[1]
        rmseCV[cursor] = metric[2]
        r2Model[cursor] = metric[3]
        corrModel[cursor] = metric[4]
        cursor += 1

    q2 = r2_score(Y, yFitCV)
    r2 = np.mean(r2Model)
    rmsecv = np.mean(rmseCV)
    rmse = np.mean(rmseModel)
    corrcv = correlation(Y, np.reshape(yFitCV, yFitCV.size))
    corrmdl = np.mean(corrModel)
    return q2, r2, rmsecv, rmse, corrcv, corrmdl
