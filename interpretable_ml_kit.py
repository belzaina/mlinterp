import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier


def get_german_credit_data(file_path="data/GermanDataset.data"):
    """
    Load and format the German Credit dataset
    Returns a DataFrame
    """
    # Read GermanDataset.data file
    dataBase = pd.read_csv('data/GermanDataset.data', sep=" ", header=None, 
                           names=["accountStatus", "creditDuration", "creditHistory", "Purpose", "creditAmount", "Savings", "employmentDuration", 
                                  "installmentRate", "gender&PersonalStatus", "Guarantor", "residenceTime", "Property", "Age", "otherInstallmentPlan", 
                                  "Housing", "numberOfCredit", "Job", "NumberLiablePeople", "Telephone", "foreignWorker", "creditRisk"])
    
    # Encode accountStatus
    dataBase.loc[dataBase["accountStatus"]=="A11", "accountStatus"] = 1
    dataBase.loc[dataBase["accountStatus"]=="A12", "accountStatus"] = 1
    dataBase.loc[dataBase["accountStatus"]=="A13", "accountStatus"] = 1
    dataBase.loc[dataBase["accountStatus"]=="A14", "accountStatus"] = 0

    # Qualitative features to dummies
    dataBase["creditRisk"] = dataBase["creditRisk"].replace(2, 0)
    copyDataBase = dataBase.copy()
    for colName in dataBase.columns:
        if dataBase[colName].dtypes == "object":
            dataBase = pd.get_dummies(dataBase, columns=[colName])
    
    # Remove "accountStatus_0", "foreignWorker_A202", and "Telephone_A192" 
    # They don't bring new information
    dataBase = dataBase.drop(["accountStatus_0", "foreignWorker_A202", "Telephone_A192"], axis=1)

    # Re-encode gender&PersonalStatus feature
    dataBaseGender = dataBase.copy()
    dataFrameGender = pd.DataFrame(copyDataBase["gender&PersonalStatus"])
    dataBaseGender = pd.concat([dataBaseGender,dataFrameGender],axis=1)
    dataBaseGender.drop(["gender&PersonalStatus_A91", "gender&PersonalStatus_A92", "gender&PersonalStatus_A93", "gender&PersonalStatus_A94"], 
                        axis=1, inplace=True)
    male = ["A91","A93","A94"]
    female = ["A92","A95"]
    for modalites in male:
        dataBaseGender["gender&PersonalStatus"] = dataBaseGender["gender&PersonalStatus"].replace(modalites,"0")
    for modalites in female:
        dataBaseGender["gender&PersonalStatus"] = dataBaseGender["gender&PersonalStatus"].replace(modalites,"1")
    dataBaseGender.rename(columns={"gender&PersonalStatus":"gender", "Telephone_A191":"Telephone"},inplace=True)

    # Return the preprocessed DataFrame
    return dataBaseGender



def _infer_feature_type(feature, X):
    """Test if feature exists and infer its type
    
    Parameters
    ----------
    feature: string or list
        list for one-hot encoded feature
    X: DataFrame
    
    Returns
    -------
    Feature types:
        1. binary
        2. onehot
        3. numeric
    """
    if type(feature) == list:
        if len(feature) < 2: 
            raise ValueError('one-hot encoded feature should contain at least 2 elements')
        if not (set(feature) < set(X.columns.values)):
            raise ValueError(f"feature does not exist: {feature}")
        feature_type = 'onehot'
    else:
        if feature not in X.columns.values:
            raise ValueError(f"feature does not exist: {feature}")
        sorted_values = sorted(list(np.unique(X[feature])))
        if sorted_values == [0, 1] or sorted_values == ["0", "1"]:
            feature_type = 'binary'
        else:
            feature_type = 'numeric'
    return feature_type



def pdp(estimator, X, reference_feature, grid_resolution=50):
    """Generate Partial Dependence Plot
    
    Parameters
    ----------
    estimator: A fitted estimator object implementing `predict_proba`
    reference_feature: string or list
        Which feature is calculated on, list for one-hot encoded feature
    grid_resolution: int, default=50
        The number of equally spaced points on the x-axis of the plot
    
    Returns
    -------
    tuple : grid values, pdp values, feature_type
    """
    feature_type = _infer_feature_type(reference_feature, X)
    pdp_average_prediction_prob = []
    if feature_type == "numeric":
        rf_min = X[reference_feature].min()
        rf_max = X[reference_feature].max()
        grid = np.linspace(rf_min, rf_max, num=grid_resolution, endpoint=True)
        for i in grid:
            pdp_df = X.copy()
            pdp_df[reference_feature] = i
            pdp_predictions = estimator.predict_proba(pdp_df)[:, 1]
            pdp_average_prediction_prob.append(pdp_predictions.mean())
    elif feature_type == "onehot":
        grid = reference_feature
        for main_label in grid:
            pdp_df = X.copy()
            other_labels = set(grid) -  set([main_label]) 
            pdp_df[main_label] = 1
            pdp_df[list(other_labels)] = 0
            pdp_predictions = estimator.predict_proba(pdp_df)[:, 1]
            pdp_average_prediction_prob.append(pdp_predictions.mean())
    else:
        grid = [0, 1]
        for i in grid:
            pdp_df = X.copy()
            pdp_df[reference_feature] = i
            pdp_predictions = estimator.predict_proba(pdp_df)[:, 1]
            pdp_average_prediction_prob.append(pdp_predictions.mean())
        grid = [str(x) for x in grid]
    return grid, pdp_average_prediction_prob, feature_type



def generic_pretty_plot(grid, values, reference_feature, feature_type, fig_size=[10, 5]):
    # pretty feature name
    if feature_type == "onehot": reference_feature = reference_feature[0].split("_")[0]
    elif feature_type == "binary": reference_feature = reference_feature.split("_")[0]
    # set figure size
    plt.rcParams['figure.figsize'] = fig_size 
    # plotting
    pdp_df = pd.DataFrame({"feature": grid, "pdp": values})
    sns.set_theme()
    sns.set_context("notebook", font_scale=1.2)
    ax = sns.lineplot(x="feature", y="pdp", data=pdp_df)
    plt.xlabel(reference_feature, labelpad=10)
    plt.ylabel("Average Predicted Probability", labelpad=20)
    return ax, reference_feature



def pdp_pretty_plot(grid, values, reference_feature, feature_type, fig_size=[10, 5]):
    ax, reference_feature = generic_pretty_plot(grid, values, reference_feature, feature_type, fig_size)
    ax.set_title('PD Plot for Feature:\n' + reference_feature)
    

    
def ice(estimator, X, reference_feature, grid_resolution=50):
    """Generate Individual Conditional Expectation Plot
    
    Parameters
    ----------
    estimator: A fitted estimator object implementing `predict_proba`
    reference_feature: string or list
        Which feature is calculated on, list for one-hot encoded feature
    grid_resolution: int, default=50
        The number of equally spaced points on the x-axis of the plot. 
        Used only if the reference feature is numeric
    
    Returns
    -------
    tuple : grid, ice values of shape (nrows, grid_resolution), feature_type
    """
    feature_type = _infer_feature_type(reference_feature, X)
    ice_prediction_prob = []
    if feature_type == "numeric":
        rf_min = X[reference_feature].min()
        rf_max = X[reference_feature].max()
        grid = np.linspace(rf_min, rf_max, num=grid_resolution, endpoint=True)
        for i in grid:
            ice_df = X.copy()
            ice_df[reference_feature] = i
            ice_predictions = estimator.predict_proba(ice_df)[:, 1]
            ice_prediction_prob.append(ice_predictions)
        ice_prediction_prob = np.stack(ice_prediction_prob).T # shape: (nrows, grid_resolution)  
    elif feature_type == "onehot":
        grid = reference_feature
        for main_label in grid:
            ice_df = X.copy()
            other_labels = set(reference_feature) -  set([main_label]) 
            ice_df[main_label] = 1
            ice_df[list(other_labels)] = 0
            ice_predictions = estimator.predict_proba(ice_df)[:, 1]
            ice_prediction_prob.append(ice_predictions)
        ice_prediction_prob = np.stack(ice_prediction_prob).T
    else:
        grid = [0, 1]
        for i in grid:
            ice_df = X.copy()
            ice_df[reference_feature] = i
            ice_predictions = estimator.predict_proba(ice_df)[:, 1]
            ice_prediction_prob.append(ice_predictions)
        ice_prediction_prob = np.stack(ice_prediction_prob).T
        grid = [str(x) for x in grid]
    return grid, ice_prediction_prob, feature_type



def ice_pretty_plot(grid, values, reference_feature, feature_type, fig_size=[10, 5]):
    # pretty feature name
    if feature_type == "onehot": reference_feature = reference_feature[0].split("_")[0]
    elif feature_type == "binary": reference_feature = reference_feature.split("_")[0]
    # set figure size
    plt.rcParams['figure.figsize'] = fig_size 
    # plotting
    sns.set_theme()
    sns.set_context("notebook", font_scale=1.2)
    for i in range(values.shape[0]):
        plt.plot(grid, values[i,], color="C0", linewidth=0.5)
    # plot average (i.e. pdp)
    pdp = np.mean(values, axis=0)
    plt.plot(grid, pdp, color="red")
    plt.title('ICE Plot for Feature:\n' + reference_feature)
    plt.xlabel(reference_feature, labelpad=10)
    plt.ylabel("Average Predicted Probability", labelpad=20)
    # add custom legend
    custom_line = Line2D([0], [0], color="red", lw=1)
    plt.legend([custom_line], ['Average'], shadow=True, loc="upper left", bbox_to_anchor=(1, 1))
    
    
    
def ale(estimator, X, reference_feature, bins=10):
    """Generate Accumulated Local Effects Plot
    
    Parameters
    ----------
    estimator: A fitted estimator object implementing `predict_proba`
    reference_feature: string or list
        Which feature is calculated on, list for one-hot encoded feature
    bins: int, default=10
        The number of quantiles is calculated as (bins + 1). 
        Used only if the reference feature is numeric
    
    Returns
    -------
    tuple : quantiles, ale, feature_type
    """
    feature_type = _infer_feature_type(reference_feature, X)
    if feature_type == "numeric":
        quantiles = np.unique(np.quantile(X[reference_feature], q = np.linspace(0, 1, bins + 1), interpolation="lower"))
        bins = len(quantiles) - 1
        # -1 to start counting from 0
        # np.clip ensure min age to be assigned to 0 not -1
        indices = np.clip(np.digitize(X[reference_feature], quantiles, right=True) - 1, 0, None)
        ale = np.zeros(bins)
        bins_sample_size = np.zeros(bins)
        for i in range(bins):
            x_subset = X[indices == i]
            subset_size = x_subset.shape[0]
            if (subset_size > 0):
                x_upper = x_subset.copy()
                x_lower = x_subset.copy()
                x_upper[reference_feature] = quantiles[i + 1]
                x_lower[reference_feature] = quantiles[i]
                individual_effect = estimator.predict_proba(x_upper)[:, 1] - estimator.predict_proba(x_lower)[:, 1] 
                ale[i] = individual_effect.mean()
                bins_sample_size[i] = subset_size
        ale = np.array([0, *ale.cumsum()])
        ale = (ale[1:] + ale[:-1]) / 2
        ale -= np.sum(ale * bins_sample_size) / X.shape[0]
        quantiles = (quantiles[1:] + quantiles[:-1]) / 2
    elif feature_type == "onehot":
        num_cat = len(reference_feature)
        ale = np.zeros(num_cat)
        sample_size = np.zeros(num_cat)
        for i in range(num_cat):
            main_feature = reference_feature[i]
            neighbor_feature = reference_feature[i - 1]
            x_subset = X[X[main_feature] == 1]
            subset_size = x_subset.shape[0]
            if (subset_size > 0):
                x_lower = x_subset.copy()
                x_lower[main_feature] = 0
                x_lower[neighbor_feature] = 1
                individual_effect = estimator.predict_proba(x_subset)[:, 1] - estimator.predict_proba(x_lower)[:, 1] 
                ale[i] = individual_effect.mean()
                sample_size[i] = subset_size
        ale = ale.cumsum()
        ale -= np.sum(ale * sample_size) / X.shape[0]
        quantiles = reference_feature
    else:
        ale = np.zeros(2)
        sample_size = np.zeros(2)
        for i in range(2):
            x_subset = X[X[reference_feature] == i]
            subset_size = x_subset.shape[0]
            if (subset_size > 0):
                x_neighbor = x_subset.copy()
                x_neighbor[reference_feature] = (1 - i)
                individual_effect = estimator.predict_proba(x_subset)[:, 1] - estimator.predict_proba(x_neighbor)[:, 1] 
                ale[i] = individual_effect.mean()
                sample_size[i] = subset_size
        ale = ale.cumsum()
        ale -= np.sum(ale * sample_size) / X.shape[0]
        quantiles = ["0", "1"]
    return quantiles, ale, feature_type



def ale_pretty_plot(grid, values, reference_feature, feature_type, fig_size=[10, 5]):
    ax, reference_feature = generic_pretty_plot(grid, values, reference_feature, feature_type, fig_size)
    ax.set_title('ALE Plot for Feature:\n' + reference_feature)