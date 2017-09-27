import numpy as np
from sklearn.cross_decomposition import PLSRegression
import regLOO_par as cv


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


def correlationCut(X, Y, threshold):

    subset = []
    for i in range(X.shape[1]):
        corr = correlation(X[:, i], Y)
        if abs(corr) > threshold:
            subset.append(i)

    return subset


def correlationVector(X, Y):

    correlations = []
    for i in range(X.shape[1]):
        correlations.append(abs(correlation(X[:, i], Y)))

    return np.array(correlations)


def regressionVector(X, Y, numLV):

    mdl = PLSRegression(n_components=numLV).fit(X, Y)
    coefs = abs(mdl.coef_)
    return coefs[:, 0]


def formatInfoVec(informativeVector, nFeatures):

    informativeVector = np.reshape(informativeVector, (-1, 1))
    ind = np.array([x for x in range(nFeatures)])
    ind = np.reshape(ind, (-1, 1))
    informativeVector = np.concatenate((informativeVector, ind), axis=1)

    return informativeVector


def sortVariables(X, Y, informativeVector):

    X = np.concatenate((informativeVector, X.T), axis=1)
    mapping = np.argsort(X[:, 0])
    mapping = mapping[::-1]
    X = X[mapping, :]
    X = np.delete(X, [0, 1], axis=1)
    X = X.T

    return X, mapping


def OPS(X, Y, maxVL_OPS, maxVL_Model, maxVariables, infoVec='prod', verbose=0):

    """
    Ordered Predictors Selection (OPS): feature selection algorithm for multivariate regression.
    For details about OPS, please see reference [1].

    Parameters
    ----------
    X : numpy array of shape [n_samples,n_features]
        Samples.
    Y : numpy array of shape [n_samples]
        Target values.
    maxVL_OPS: integer
        Maximum number of latent variables tested in OPS for sorting varables (when using 'prod' or 'reg' infoVec).
    maxVL_Model: integer
        Maximum number of latent variables tested in PLS models with each feature subset.
    maxVariables: integer
        Maximum number of features tested.
    infoVec : {'prod', 'reg', or 'corr'}
        The informative vector used to sort variables. 
        'corr' is the correlation vector between features and target values, 
        'reg' is the regression vector obtained with a PLS model, and
        'prod' is the combination of correlation vector and regression vector (usually is best)
    verbose : {0,1}
        0 Verbosity mode off.
        1 Verbosity mode on.


    Returns
    -------
    subset : integer list
        Index columns of selected features.
    q2 : float
        Q^2 external validation metric. For details see reference [2].
    NV : int
        Number of selected features.
    NVL_Model: float
        Number of PLS latent variables.
    NVL_OPS : int
        Number of OPS latent variables used for sorting variables.
       
    
    References
    ----------
        [1] Teófilo, R. F.; Martins, J. P. A. & Ferreira, M. M. C. (2009).
        Sorting variables by using informative vectors as a strategy for feature selection in multivariate regression. 
        Journal of Chemometrics, 23(1):32--48. ISSN 0886-9383

        [2] Martins, J. P. A.; Barbosa, E. G.; Pasqualoto, K. F. M. & Ferreira, M. M. C. (2009).
        LQTA-QSAR: a new 4D-QSAR methodology. 
        Journal of Chemical Information and Modeling, 49(6):1428--1436. ISSN 1549-9596
    """
    nSamples, nFeatures = X.shape
    if maxVariables > nFeatures:
        maxVariables = nFeatures

    # Best models are saved in this matrix, each line is formatted as 
    # [Q^2, NumVariables, numLatentVariables (Model), NumLatentVariables (OPS), Key of features subset]
    results = np.empty([0, 5])

    selFeats = {}
    key_features = 1

    if infoVec == 'corr':
        maxVL_OPS = 1

    for NVL_OPS in range(1, maxVL_OPS + 1):

        if verbose == 1:
            stdout = 'Running OPS ' + str(int((NVL_OPS / maxVL_OPS) * 100)) + '%/' + str(int((maxVL_OPS / maxVL_OPS) * 100)) + '%'
            print(stdout)

        iterResults = np.empty([0, 5])
        iterSelFeats = {}


        if infoVec == 'corr':
            corrVec = correlationVector(X, Y)
            informativeVector = formatInfoVec(corrVec, nFeatures)
        elif infoVec == 'reg':
            regVec = regressionVector(X, Y, NVL_OPS)
            informativeVector = formatInfoVec(regVec, nFeatures)
        else:
            corrVec = correlationVector(X, Y)
            regVec = regressionVector(X, Y, NVL_OPS)
            prodVec = regVec * corrVec
            informativeVector = formatInfoVec(prodVec, nFeatures)

        _, mapping = sortVariables(X, Y, informativeVector)

        for NV in range(1, maxVariables + 1):
            subset = mapping[0:NV]
            for NVL_Model in range(1, maxVL_Model + 1):

                if NV < NVL_Model:
                    continue
                q2, _, _, _, _, _ = cv.plsLOO(X[:, subset], Y, NVL_Model)

                metrics = np.array([q2, NV, NVL_Model, NVL_OPS, key_features])
                iterSelFeats[key_features] = subset
                key_features = key_features + 1
                iterResults = np.vstack((iterResults, metrics))

        bestModelIndex = np.argmax(iterResults[:, 0])
        bestModel = iterResults[bestModelIndex, :]
        if verbose == 1:
            print('Iteration\'s best model:', 'Q2:', '{0:.4f}'.format(bestModel[0]), 'NV:', int(bestModel[1]), 'NVL_Model:', int(bestModel[2]), 'NVL_OPS:', int(bestModel[3]))
        results = np.vstack((results, bestModel))
        selFeats[bestModel[4]] = iterSelFeats[bestModel[4]]

    maxPerf = np.argmax(results[:, 0])
    q2 = results[maxPerf, 0]
    NV = results[maxPerf, 1]
    NVL_Model = results[maxPerf, 2]
    NVL_OPS = results[maxPerf, 3]
    subset = selFeats[results[maxPerf, 4]]
    if verbose == 1:
        print('Best model found:', 'Q2:', '{0:.4f}'.format(q2), 'NV:', int(NV), 'NVL_Model:', int(NVL_Model), 'NVL_OPS:', int(NVL_OPS))
    return subset, q2, NV, NVL_Model, NVL_OPS


def OPS_auto(X, Y, maxVL_OPS, maxVL_Model, maxVariables, verbose=0):
    """
    Automated version of Ordered Predictors Selection (OPS): 
    feature selection algorithm for multivariate regression. For details about OPS, please see reference [1].

    This implemetation automats the process testing all available informative vectors for sorting variables.
    It's runs slowly than the previous implementation.

    Parameters
    ----------
    X : numpy array of shape [n_samples,n_features]
        Samples.
    Y : numpy array of shape [n_samples]
        Target values.
    maxVL_OPS: integer
        Maximum number of latent variables tested in OPS for sorting varables (when using 'prod' or 'reg' infoVec).
    maxVL_Model: integer
        Maximum number of latent variables tested in PLS models with each feature subset.
    maxVariables: integer
        Maximum number of features tested.
    verbose : {0,1}
        0 Verbosity mode off.
        1 Verbosity mode on.


    Returns
    -------
    subset : integer list
        Index columns of selected features.
    q2 : float
        Q^2 external validation metric. For details see reference [2].
    NV : int
        Number of selected features.
    NVL_Model: float
        Number of PLS latent variables.
    NVL_OPS : int
        Number of OPS latent variables used for sorting variables.
    infoVec : int
        Returns the informative vector used for sorting variables in the best model found.
        1 if it's correlation vector.
        2 if it's regression vector.
        3 if it's combination of correlation and regression vectors
       
    
    References
    ----------
        [1] Teófilo, R. F.; Martins, J. P. A. & Ferreira, M. M. C. (2009).
        Sorting variables by using informative vectors as a strategy for feature selection in multivariate regression. 
        Journal of Chemometrics, 23(1):32--48. ISSN 0886-9383

        [2] Martins, J. P. A.; Barbosa, E. G.; Pasqualoto, K. F. M. & Ferreira, M. M. C. (2009).
        LQTA-QSAR: a new 4D-QSAR methodology. 
        Journal of Chemical Information and Modeling, 49(6):1428--1436. ISSN 1549-9596
    """


    nSamples, nFeatures = X.shape
    if maxVariables > nFeatures:
        maxVariables = nFeatures


    # Best models are saved in this matrix, each line is formatted as 
    # [Q^2, NumVariables, numLatentVariables (Model), NumLatentVariables (OPS), Key of features subset, Informative Vector]
    results = np.empty([0, 6])

    selFeats = {}
    key_features = 1

    for NVL_OPS in range(1, maxVL_OPS + 1):

        if verbose == 1:
            stdout = 'Running OPS ' + str(int((NVL_OPS / maxVL_OPS) * 100)) + '%/' + str(int((maxVL_OPS / maxVL_OPS) * 100)) + '%'
            print(stdout)

        iterResults = np.empty([0, 6])
        iterSelFeats = {}

        regVec = regressionVector(X, Y, NVL_OPS)
        corrVec = correlationVector(X, Y)
        prodVec = regVec * corrVec

        # Begin testing with correlation vetor
        informativeVector = formatInfoVec(corrVec, nFeatures)
        _, mapping = sortVariables(X, Y, informativeVector)
        for NV in range(1, maxVariables + 1):
            subset = mapping[0:NV]
            for NVL_Model in range(1, maxVL_Model + 1):

                if NV < NVL_Model:
                    continue
                q2, _, _, _, _, _ = cv.plsLOO(X[:, subset], Y, NVL_Model)

                metrics = np.array([q2, NV, NVL_Model, NVL_OPS, key_features, 1])
                iterSelFeats[key_features] = subset
                key_features = key_features + 1
                iterResults = np.vstack((iterResults, metrics))
        # End testing with correlation vetor

        # Begin testing with regression vetor
        informativeVector = formatInfoVec(regVec, nFeatures)
        _, mapping = sortVariables(X, Y, informativeVector)
        for NV in range(1, maxVariables + 1):
            subset = mapping[0:NV]
            for NVL_Model in range(1, maxVL_Model + 1):

                if NV < NVL_Model:
                    continue
                q2, _, _, _, _, _ = cv.plsLOO(X[:, subset], Y, NVL_Model)

                metrics = np.array([q2, NV, NVL_Model, NVL_OPS, key_features, 2])
                iterSelFeats[key_features] = subset
                key_features = key_features + 1
                iterResults = np.vstack((iterResults, metrics))
        # End testing with regression vetor

        # Begin testing with product vetor
        informativeVector = formatInfoVec(prodVec, nFeatures)
        _, mapping = sortVariables(X, Y, informativeVector)
        for NV in range(1, maxVariables + 1):
            subset = mapping[0:NV]
            for NVL_Model in range(1, maxVL_Model + 1):

                if NV < NVL_Model:
                    continue
                q2, _, _, _, _, _ = cv.plsLOO(X[:, subset], Y, NVL_Model)

                metrics = np.array(
                    [q2, NV, NVL_Model, NVL_OPS, key_features, 3])
                iterSelFeats[key_features] = subset
                key_features = key_features + 1
                iterResults = np.vstack((iterResults, metrics))
        # End testing with product vetor

        bestModelIndex = np.argmax(iterResults[:, 0])
        bestModel = iterResults[bestModelIndex, :]
        if verbose == 1:
            print('Iteration\'s best model:', 'Q2:', '{0:.4f}'.format(bestModel[0]), 'NV:', int(bestModel[1]), 'NVL_Model:', int(bestModel[2]), 'NVL_OPS:', int(bestModel[3]), 'InfoVec', int(bestModel[5]))
        results = np.vstack((results, bestModel))
        selFeats[bestModel[4]] = iterSelFeats[bestModel[4]]

    maxPerf = np.argmax(results[:, 0])
    q2 = results[maxPerf, 0]
    NV = results[maxPerf, 1]
    NVL_Model = results[maxPerf, 2]
    NVL_OPS = results[maxPerf, 3]
    infoVec = results[maxPerf, 5]
    subset = selFeats[results[maxPerf, 4]]
    if verbose == 1:
        print('Best model found:', 'Q2:', '{0:.4f}'.format(q2), 'NV:', int(NV), 'NVL_Model:', int(NVL_Model), 'NVL_OPS:', int(NVL_OPS), 'InfoVec', int(infoVec))
    return subset, q2, NV, NVL_Model, NVL_OPS, infoVec
