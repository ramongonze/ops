import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression


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


def plsLOO(X, Y, numVL):
    """
    Leave-one-out cross-validation with Partial Least Squares(PLS)

    Parameters
    ----------
    X : numpy array of shape [n_samples,n_features]
        Samples.
    Y : numpy array of shape [n_samples]
        Target values.
    numVL: int
        Number of latent variables for PLS.

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

    for train_index, test_index in cv.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        mdl = PLSRegression(n_components=numVL).fit(X_train, Y_train)
        coefs = mdl.coef_
        intercept = mdl.y_mean_ - np.dot(mdl.x_mean_, mdl.coef_)
        trainFit = X_train.dot(coefs) + intercept
        testFit = X_test.dot(coefs) + intercept

        yFitCV[test_index] = testFit
        rmseModel[test_index] = mean_squared_error(Y_train, trainFit)
        rmseCV[test_index] = mean_squared_error(Y_test, testFit)
        r2Model[test_index] = r2_score(Y_train, trainFit)
        corrModel[test_index] = correlation(Y_train, trainFit)

    q2 = r2_score(Y, yFitCV)
    r2 = np.mean(r2Model)
    rmsecv = np.mean(rmseCV)
    rmse = np.mean(rmseModel)
    corrcv = correlation(Y, np.reshape(yFitCV, yFitCV.size))
    corrmdl = np.mean(corrModel)
    return q2, r2, rmsecv, rmse, corrcv, corrmdl
