import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from scipy import stats

def graphKMean(X, kmeans, xlabel, ylabel, zlabel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    try:
        ax.scatter(X[:,0], X[:,1], X[:,2], c = kmeans.labels_, cmap= "rainbow")
        ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], color='black')

    except:
        ax.scatter(X[:,0], X[:,1], c = kmeans.labels_, cmap="rainbow")
        ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = "black")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    plt.show()


def qqPlot(x, label):
    res = stats.probplot(x, plot=plt)
    plt.title(label + " QQ plot")
    plt.show()


def pcaPlot(per_var, labels, pca_df, coneValue):
    colorDir = ["black", "gray", "blue", "red", "white"]   # get color via list index (1 = L cone = gray, etc.)

    plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
    plt.ylabel("Percentage of Explained Variance")
    plt.xlabel("Principal Component")
    plt.title("Scree Plot")
    plt.show()

    for sample in pca_df.index:
        col = int(coneValue.values[sample])
        plt.scatter(pca_df.PC1.iloc[sample], pca_df.PC2.iloc[sample], color = colorDir[col])  # easier way to add color?
    plt.title("PCA Graph")
    plt.show()


def svcPlot(model, data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.scatter(data[:, 0], data[:, 1], c="y", s=50, cmap="autumn")

    xlim = ax.get_xlim()  # from 0 to 1
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    x = np.linspace(xlim[0], xlim[1], 30)  # return 30 even intervals between points
    y = np.linspace(ylim[0], ylim[1], 30)
    z = np.linspace(zlim[0], zlim[1], 30)
    Z, Y, X = np.meshgrid(y,x,z)   # not sure why we have to mesh them together. We ravel these vars next step


    xyz = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T   # transposes the ravelled points
    print xyz.reshape(30)
    P = model.decision_function(xyz)

    print len(X)
    print len(Y)
    print len(Z)
    print len(xyz)
    print len(P)


    # ax.contour(X, Y, Z, P, colors = "k",
    #            levels = [-1, 0, 1], alpha = .5, linestyles = ["--", "-", "--"])
    #
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    # ax.set_zlim(zlim)



def graphPanda3D(x, y, z, id, xlab, ylab, zlab):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    s_x, s_y, s_z, m_x, m_y, m_z, l_x, l_y, l_z = sort(id, x, y, z)

    ax.scatter(s_x, s_y, s_z, c="r")
    ax.scatter(m_x, m_y, m_z, c = "b")
    ax.scatter(l_x, l_y, l_z, c = "gray")

    ax.legend(["S cones", "M cones", "L cones"])

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)

    plt.show()

def graphArray3D(array, id, xlab, ylab, zlab):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    x = array[:,0]
    y = array[:,1]
    z = array[:,2]

    s_x, s_y, s_z, m_x, m_y, m_z, l_x, l_y, l_z = sort(id, x, y, z)

    ax.scatter(s_x, s_y, s_z, c="r")
    ax.scatter(m_x, m_y, m_z, c="b")
    ax.scatter(l_x, l_y, l_z, c="gray")

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)

    plt.show()


def graphArray2D(array, id, xlab, ylab):
    x = array[:,0]
    y = array[:,1]

    s_x, s_y, m_x, m_y, l_x, l_y = sort(id, x, y, None)

    plt.scatter(s_x, s_y, c="r")
    plt.scatter(m_x, m_y, c="b")
    plt.scatter(l_x, l_y, c="gray")

    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.show()


def sort(id, x, y, z):
    # need to classify x, y, and z cords to show distinction in color
    # Note: there should be a much more efficient way to do this
    s_cone_x = []
    s_cone_y = []
    s_cone_z = []
    m_cone_x = []
    m_cone_y = []
    m_cone_z = []
    l_cone_x = []
    l_cone_y = []
    l_cone_z = []

    if type(x) == pd.core.series.Series:
        print "panda"
        for i in range(len(x)):
            if id.iloc[i] == 1.0:
                l_cone_x.append(x.iloc[i])
                l_cone_y.append(y.iloc[i])
                try:    # using try and except in case of 2D graph
                    l_cone_z.append(z.iloc[i])
                except:
                    pass
            elif id.iloc[i] == 2.0:
                m_cone_x.append(x.iloc[i])
                m_cone_y.append(y.iloc[i])
                try:
                    m_cone_z.append(z.iloc[i])
                except:
                    pass
            elif id.iloc[i] == 3.0:
                s_cone_x.append(x.iloc[i])
                s_cone_y.append(y.iloc[i])
                try:
                    s_cone_z.append(z.iloc[i])
                except:
                    pass

    else:   # for regular/numpy arrays
        print "array"
        for i in range(len(x)):
            if id.iloc[i] == 1.0:
                l_cone_x.append(x[i])
                l_cone_y.append(y[i])
                try:
                    l_cone_z.append(z[i])
                except:
                    pass
            elif id.iloc[i] == 2.0:
                m_cone_x.append(x[i])
                m_cone_y.append(y[i])
                try:
                    m_cone_z.append(z[i])
                except:
                    pass
            elif id.iloc[i] == 3.0:
                s_cone_x.append(x[i])
                s_cone_y.append(y[i])
                try:
                    s_cone_z.append(z[i])
                except:
                    pass

    total_z = s_cone_z + l_cone_z + m_cone_z
    if len(total_z) == 0:
        return s_cone_x, s_cone_y, m_cone_x, m_cone_y, l_cone_x, l_cone_y

    return s_cone_x, s_cone_y, s_cone_z, m_cone_x, m_cone_y, m_cone_z, l_cone_x, l_cone_y, l_cone_z