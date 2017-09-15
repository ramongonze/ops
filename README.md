# Ordered Predictors Selection (OPS)

Implementation of OPS method for feature selection in multivariate regression. For details about this method, please see reference [1]

### System Requirements

python 3.6 (or newer version)

numpy 1.13.1 (or newer version)

scikit-learn 0.18.2 (or newer version)

### Usage

#### OPS
For using OPS with specific informative vector ('corr','reg' or 'prod'), arguments must be passed in the format:

main_crossVal.py *<predictors_path> <target_path> <output_name> <correlation_threshold> <maxNumLatentVariables_OPS> <maxNumLatentVariables_model> <maxNumSelectedFeatures> <informativeVector>*

Example:
```bash
    $ python main_crossVal.py X.txt Y.txt outputName 0.3 10 3 5 corr
```

#### OPS automatized

For using OPS implementation with automatic choice of informative vector (it runs slowly than using specific vector), arguments must be passed in the format:

main_crossVal.py *<predictors_path> <target_path> <output_name> <correlation_threshold> <maxNumLatentVariables_OPS> <maxNumLatentVariables_model> <maxNumSelectedFeatures>*

Example:
```bash
    $ python main_crossVal.py X.txt Y.txt outputName 0.3 10 3 5
```

#### Using in other Applications

You can import both implementations in others applications, please see code for more details about this functions.

Example:
```python
    from OPS import OPS
    OPS(X, Y, maxVL_OPS, maxVL_Model, maxVariables, infoVec=nameInfoVec)
    from OPS import OPS_auto
    OPS_auto(X, Y, maxVL_OPS, maxVL_Model, maxVariables)
```

### Output File

The output of the program is written in two comma-separated files:

'*<output_name>*_metrics.txt' contains the metrics about the best model found in OPS in the format: *'Q², R², RMSECV, RMSE, CORRCV, CORRMDL, numSelectedVariables, numLatentVariablesModel, numLatentVariablesOPS'*

'*<output_name>*_selectedFeatures.txt' contains the indexes of selected predictors in matrix '<predictors_path>' passed as argument.
 
### References
[1] Teófilo, R. F.; Martins, J. P. A. & Ferreira, M. M. C. (2009). Sorting variables by using informative vectors as a strategy for feature selection in multivariate regression. Journal of Chemometrics, 23(1):32--48. ISSN 0886-9383
