import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multicomp as ml

import matplotlib.pyplot as plt

def ANOVA(df1, df2, df3):
    # some columns are no longer useful
    try:
        df1 = df1.drop(["Coord_x", "Coord_y", "Cone_Type(L=1;M=2;S=3)"], axis = 1)
        df2 = df2.drop(["Coord_x", "Coord_y", "Cone_Type(L=1;M=2;S=3)"], axis = 1)
        df3 = df3.drop(["Coord_x", "Coord_y", "Cone_Type(L=1;M=2;S=3)"], axis=1)

    except:
        print "Error occured in dropping columns from dataframe"

    F, p = stats.f_oneway(df1, df2, df3)

    print F
    print p


def tukeyTest(dfShort, dfMedium, dfLong, col):
    # need to recombine the data sets for using Tukey in Python
    dfTotal = dfShort.append(dfMedium.append(dfLong))

    # group 1: quantitative data
    # group 2: categorical variable
    group1 = dfTotal[col]
    group2 = dfTotal["Cone_Type(L=1;M=2;S=3)"]

    mc = ml.MultiComparison(group1, group2)
    out = mc.tukeyhsd(alpha=.05)

    print col + ":"
    print out

