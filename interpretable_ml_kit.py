import pandas as pd
import numpy as np

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