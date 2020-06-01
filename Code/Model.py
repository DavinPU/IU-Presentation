import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import Algorithms   # other python Code
import Graphing
import Tests

fileName = "Z:\IU_Project\ConeFeatureData.xlsx"


def normalizeDF(df, scaler):

    np_scaled = scaler.fit_transform(df)
    df_norm = pd.DataFrame(np_scaled)
    return df_norm

def readData(fileName):
    min_max_scaler = preprocessing.MinMaxScaler()   # creates scaler for normalizing later

    fullExcel = pd.ExcelFile(fileName)
    df53 = pd.read_excel(fullExcel, "Subject_53")    # creates a data frame for each subject (sheet) in the excel file
    df130 = pd.read_excel(fullExcel, "Subject_130")
    df151 = pd.read_excel(fullExcel, "Subject_151")

    df151.rename(columns = {'Cone_Type(L=1;M=2;S=3; Undertermined=4, can ignore)':'Cone_Type(L=1;M=2;S=3)'}, inplace = True)

    df53 = df53.dropna(how='any')   # drops a row if any of the values in that row are nan  ## CLEANS DATA
    df151 = df151.dropna(how='any')
    df130 = df130.dropna(how='any')

    df53["OSL"] = df53["COST_z"] - df53["ISOS_z"]
    df151["OSL"] = df151["COST_z"] - df151["ISOS_z"]
    df130["OSL"] = df130["COST_z"] - df130["ISOS_z"]

    dfTotal = df53.append(df130.append(df151))   # NON-normalized dataframe
    #####dfTotal = df53
    df53_norm = normalizeDF(df53, min_max_scaler)   # normalizes dataframes

    df130_norm = normalizeDF(df130, min_max_scaler)
    df151_norm = normalizeDF(df151, min_max_scaler)

    dfTotal_norm = df53_norm.append(df130_norm.append(df151_norm))  # adds all the dataframes together as one\
    #####dfTotal_norm = df53_norm
    # we don't want to normalize cone type, so we drop from data and add it back as it was before
    dfTotal_norm = dfTotal_norm.drop(8, axis = 1)

    dfTotal_norm.rename(columns = {0:"Coord_x", 1:"Coord_y", 2:"ISOS_z", 3:"ISOS_Size_x", 4:"ISOS_Size_y",
                              5:"COST_z", 6:"COST_Size_x", 7:"COST_Size_y", 9:"OSL"} ,inplace = True)

    dfTotal_norm["Cone_Type(L=1;M=2;S=3)"] = dfTotal["Cone_Type(L=1;M=2;S=3)"].tolist()  # re-adds column

    return dfTotal, dfTotal_norm


def sortByType(df):
    df["ISOSr"] = (abs(df["ISOS_Size_x"] ** 2 - df["ISOS_Size_y"] ** 2)) ** .5
    df["COSTr"] = (abs(df["COST_Size_x"] ** 2 - df["COST_Size_y"] ** 2)) ** .5

    shortCone = df[df["Cone_Type(L=1;M=2;S=3)"] == 3.0]   # seperates into 3 dataframes based on length of cone
    mediumCone = df[df["Cone_Type(L=1;M=2;S=3)"] == 2.0]
    longCone = df[df["Cone_Type(L=1;M=2;S=3)"] == 1.0]

    return shortCone, mediumCone, longCone


dfTotal, dfTotal_norm = readData(fileName)  # dfTotal = combined dataframe for all data in the given excel file


# creates TRAINING and TESTING data
ISOS_xTrain, ISOS_xTest, ISOS_yTrain, ISOS_yTest,\
    outerLenTrain, outerLenTest, ISOS_zTrain, ISOS_zTest,\
    coneValueTrain, coneValueTest,\
    COST_xTrain, COST_xTest, COST_yTrain, COST_yTest,\
    COST_zTrain, COST_zTest = train_test_split(dfTotal_norm["ISOS_Size_x"], dfTotal_norm["ISOS_Size_y"], dfTotal_norm["OSL"],
                                                     dfTotal_norm["ISOS_z"], dfTotal_norm["Cone_Type(L=1;M=2;S=3)"], dfTotal_norm["COST_Size_x"], dfTotal_norm["COST_Size_y"],
                                               dfTotal_norm["COST_z"])

# pcaDF = pd.DataFrame({"ISOS_Size_x": ISOS_xTrain, "ISOS_Size_y": ISOS_yTrain, "ISOS z": ISOS_zTrain,
#                       "COST_Size_x": COST_xTrain, "COST_Size_y": COST_yTrain, "COST z": COST_zTrain, "OSL": outerLenTrain})

# per_var, labels, pca_df = Algorithms.pcaAnalysis(pcaDF)
#
# Graphing.pcaPlot(per_var, labels, pca_df, coneValueTrain)
#
trainingDF = np.array(zip(outerLenTrain, COST_zTrain, ISOS_zTrain))  # use zip to turn into tuple
testingDF = np.array(zip(outerLenTest, COST_zTest, ISOS_zTest))

#
# result = Algorithms.knnClassify(trainingDF, coneValueTrain, testingDF, coneValueTest, 2.0, 5)
#

# Graphing.qqPlot(dfTotal["ISOS_Size_y"], "ISOS y")

# Algorithms.neuralNet(trainingDF, coneValueTrain, testingDF, coneValueTest, 3.0)

Graphing.graphPanda3D(ISOS_xTrain, ISOS_yTrain, ISOS_zTrain, coneValueTrain, "ISOS x", "ISOS y", "ISOS z")

#clusterResult = Algorithms.cluster(trainingDF, testingDF, coneValueTest, 2.0)
#
# Graphing.graphPanda3D(outerLenTrain, COST_zTrain, ISOS_zTrain, coneValueTrain, "OSL", "COST z", "ISOS z")
# Graphing.graphKMean(trainingDF, clusterResult, "OSL", "COST z", "ISOS z")

# svmResult, svmModel = Algorithms.svmAnalysis(trainingDF, coneValueTrain, testingDF, coneValueTest, 3.0)
#
# Graphing.svcPlot(svmModel, trainingDF)

### dfShort, dfMedium, dfLong = sortByType(dfTotal_norm)
# Tests.ANOVA(dfShort, dfMedium, dfLong)

# Tests.tukeyTest(dfShort, dfMedium, dfLong, "ISOS_z")
# Tests.tukeyTest(dfShort, dfMedium, dfLong, "ISOS_Size_x")
# Tests.tukeyTest(dfShort, dfMedium, dfLong, "ISOS_Size_y")
# Tests.tukeyTest(dfShort, dfMedium, dfLong, "COST_z")
# Tests.tukeyTest(dfShort, dfMedium, dfLong, "COST_Size_y")
# Tests.tukeyTest(dfShort, dfMedium, dfLong, "COST_Size_x")
# Tests.tukeyTest(dfShort, dfMedium, dfLong, "OSL")
# Tests.tukeyTest(dfShort, dfMedium, dfLong, "ISOSr")
# Tests.tukeyTest(dfShort, dfMedium, dfLong, "COSTr")

# writer = pd.ExcelWriter("Z:\IU_Project\Output.xlsx")
# dfShort.to_excel(writer, sheet_name = "Sheet1")
# dfMedium.to_excel(writer, sheet_name = "Sheet2")
# dfLong.to_excel(writer, sheet_name = "Sheet3")
#
# writer.save()
