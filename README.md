# Ordered Predictors Selection (OPS)

Implementation of OPS method for feature selection in multivariate regression. For details about this method, please see reference [1]

### System Requirements

python 3.6 (or newer version)

numpy 1.13.1 (or newer version)

scikit-learn 0.18.2 (or newer version)

#### For parallelized version
GNU parallel 20141022 (or newer version)

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

#### OPS parallelized
Before use the script, you must be able to connect to all hosts without a password using ssh.
It can be setup by running 'ssh-keygen -t dsa; ssh-copy-id <hostname>' and using an empty passphrase.
The predictors path and the target path must be in your current directory (pwd).
Be sure the file 'transition.txt' doesn't exist in the current directory before run this script.

For using OPS implementation in the parallelized version, the arguments must be put as variables values in the file *run.sh*:

**X_dataset**="*<predictors_path>*"

**Y_dataset**="*<target_path>*"

**outputName**="*<output_name>*"

**corrTh**=*<correlation_threshold>*

**maxVL_OPS**=*<maxNumLatentVariables_OPS>*

**maxVL_model**=*<maxNumLatentVariables_model>*

**maxVariables**=*<maxNumSelectedFeatures*>

**nameInfoVec**='*<infoVec_name>*'

*Possible values for nameInfoVec: 'corr', 'reg' or 'prod', corresponding respectively to  correlation, regression and product vectors. If you want to run the OPS automatized (for all the informative vectors), let nameInfoVec=' '*

**verbose**=1

*Let verbose=1 if you want to show the progress or verbose=0 otherwise*

**servers**='*<number_of_cores>/<host_name>*'

*The variable 'servers' contains the names of all hosts which will be used. The local host is represented with ':'. The number before the '/' is the number of cores used in that host.
To add more hosts, write each host name separating with a comma. For example: servers='8/:,4/host1,8/host2'.*

**maxProcess**=*<max_number_of_process>*

*maxProcess is the max number of process running in parallel*

To use this version, just run:

```bash
    $ ./run.sh
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

[2] O. Tange (2011): GNU Parallel - The Command-Line Power Tool, ;login: The USENIX Magazine, February 2011:42-47.
