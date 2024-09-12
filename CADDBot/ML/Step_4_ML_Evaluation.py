'''
This is a collection of tools will be used in ML_models.py module

'''
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from scipy.stats import linregress, t, pearsonr, spearmanr, kendalltau


################################################################################
def CalcScores(y_pred, y_true):
    y_pred = y_pred.reshape((len(y_pred), ))
    y_true = y_true.reshape((len(y_true), ))
    print('y_pred: ', y_pred.shape, 'y_true: ', y_true.shape)

    ## 
    MAE = mean_absolute_error(y_true, y_pred)
    print("Mean absolute error: %.4f" % (MAE))

    x, y = y_pred, y_true
    pr_np = np.corrcoef(y_pred, y_true)[1, 0]    # PearsonCorrelationCoefficient
    pr_sp, pp_sp = pearsonr(y_pred, y_true)[0], pearsonr(x, y)[1]
    sr_sp, sp_sp = spearmanr(y_pred, y_true)[0], spearmanr(x, y)[1]    # Spearman's rho
    kr_sp, kp_sp = kendalltau(y_pred, y_true)[0] , kendalltau(y_pred, y_true)[1]   # Kendall's tau
    print('Pearson-R (Numpy): %.4f, Pearson-R (Scipy): %.4f (p=%.4f)' % (pr_np, pr_sp, pp_sp))
    print('Spearman-R: %.4f (p=%.4f), Kendall-R: %.4f (p=%.4f)' % (sr_sp, sp_sp, kr_sp, kp_sp))

    dataDict_result = {"MAE": MAE, "Pearson-R_np": pr_np, "Pearson-R_sp": pr_sp, "Spearman-R": sr_sp, "Kendall-R": kr_sp}
    return dataDict_result

################################################################################
def CalcScoresStats_CV(CV_results):
    CV_results_stats = {}
    scoreTypes = list(CV_results[0].keys())
    for scoreType in scoreTypes:
        scores = []

        for fold_id in CV_results.keys():
            scores.append(CV_results[fold_id][scoreType])
        CV_results_stats[scoreType] = (np.mean(scores), np.std(scores))
    return CV_results_stats

################################################################################
def PlotPrediction(ax, data_ax_x, data_ax_y, colorby=None, labels={'x':'Pred', 'y':'Exp'}, diagonal=True, title=None, ax_max=None, ax_min=None):
    if colorby is None:
        ax.scatter(data_ax_x, data_ax_y, s=20, alpha=0.5, cmap='Spectral', marker='o')
    else:
        ##  example of colorby: {'colName': 'project', 'dataTable': dataTable_all}
        colorby_col, colorby_dataTable = colorby['colName'], colorby['dataTable']
        for i in sorted(colorby_dataTable[colorby_col].unique()):
            idx = colorby_dataTable[colorby_dataTable[colorby_col]==i].index.to_list()
            ax.scatter(data_ax_x[idx], data_ax_y.loc[idx], s=20, alpha=0.5, cmap='Spectral', marker='o', label=i)
        ax.legend(loc="center right", bbox_to_anchor=(1.35, 0.5), title=colorby_col)
    
    ## ---------------------------------- plot diagonal line ----------------------------------
    if ax_max is None:
        ax_max = np.ceil(np.max([data_ax_x.max(), data_ax_y.max().values[0]]))
    if ax_min is None:
        ax_min = np.floor(np.min([data_ax_x.min(), data_ax_y.min().values[0]]))

    if diagonal:        
        diagonalLine = ax.plot([ax_min, ax_max], [ax_min, ax_max], c='lightgray', linestyle='-')
        fold1Line1 = ax.plot([ax_min+1, ax_max], [ax_min, ax_max-1], c='lightgray', linestyle='--')
        fold1Line1 = ax.plot([ax_min, ax_max-1], [ax_min+1, ax_max], c='lightgray', linestyle='--')

    ## ---------------------------------- defind the figure lable and scales ----------------------------------
    ax.set_xlabel(labels['x'], fontsize=12)
    ax.set_ylabel(labels['y'], fontsize=12)
    ax.grid(alpha=0.75)
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ## set subtitle
    if title:
        ax.set_title(title, fontsize=16)
    return ax

################################################################################
def ModelValidation(myModel, dataDict, Cmpds_train, Cmpds_test, colorby=None, label=None):
        
    dataSets = list(dataDict.keys())   
    [Desc_train, y_train] = dataDict[dataSets[0]]
    [Desc_test, y_test] = dataDict[dataSets[1]]

    dataDict_results = {}
    for dataSet in dataSets:
        dataDict_results[dataSet] = {}
    
    # Predict data of estimated models
    y_train_pred = myModel.predict(Desc_train)
    dataDict_results[dataSets[0]]['Stats'] = CalcScores(y_train_pred, y_train.to_numpy())

    y_test_pred = myModel.predict(Desc_test)
    dataDict_results[dataSets[1]] = CalcScores(y_test_pred, y_test.to_numpy())

    ## evaluate the performance
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=0.5, hspace=0.1)
    
    colorby_train = {'colName': colorby, 'dataTable': Cmpds_train}
    colorby_test = {'colName': colorby, 'dataTable': Cmpds_test}

    if label:
        labels = {'x': f'Pred {label}', 'y': f'Exp {label}'}
    else:
        labels = {'x': f'Pred', 'y': f'Exp'}

    ax_max = np.ceil(np.max([y_train_pred.max(), y_train.max().values[0]]))+1
    ax_min = np.ceil(np.min([y_train_pred.min(), y_train.min().values[0]]))-1

    axs[0] = PlotPrediction(axs[0], y_train_pred, y_train, colorby=colorby_train, labels=labels, title=dataSets[0], ax_max=ax_max, ax_min=ax_min)
    axs[1] = PlotPrediction(axs[1], y_test_pred, y_test, colorby=colorby_test, labels=labels, title=dataSets[1], ax_max=ax_max, ax_min=ax_min)
    
    return dataDict_results


################################################################################# 


def RocAUC_CrossValidation(classifier, X, y, n_splits=10, random_state=None, Annotation='Efflux Ratio > 16'):
    
    ## define the cross validation set
    if random_state:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=n_splits)
    
    ## initiate the figure axes
    fig, ax = plt.subplots(figsize=(6, 6))

    dataDict_retsult = {}
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    ## plot the ROC curve for all the fold validation
    for fold, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        dataDict_retsult[f"fold_{fold}"] = {}
        dataDict_retsult[f"fold_{fold}"]['fpr'] = mean_fpr
        dataDict_retsult[f"fold_{fold}"]['tpr'] = interp_tpr
        dataDict_retsult[f"fold_{fold}"]['auc'] = viz.roc_auc

    ## plot the mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), lw=2, alpha=0.8)

    dataDict_retsult["mean"] = {}
    dataDict_retsult["mean"]['fpr'] = mean_fpr
    dataDict_retsult["mean"]['tpr'] = mean_tpr
    dataDict_retsult["mean"]['auc'] = mean_auc
    dataDict_retsult["mean"]['std'] = std_auc

    ## fill the std area with grey
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")

    ## set the figure config
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], 
        xlabel="False Positive Rate", ylabel="True Positive Rate", 
        title=f"Mean ROC curve with variability\n(Positive label {Annotation})")
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()
    return fig, dataDict_retsult



def getROC(classifier, X, y):
    y_scores = classifier.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_scores)
    fnr = 1 - tpr
    tnr = 1 - fpr


