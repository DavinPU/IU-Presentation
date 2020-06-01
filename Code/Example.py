import matplotlib.pyplot as plt
import numpy as np

data = [[3, 1.5, 1],     # each point is length, width, type(0,1)
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, .5, 1],
        [2, .5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]

mystery_flower = [4.5, 1]

# network

#    o flower type
#   /  \   w1, w2, b
#  o    o   length, width

w1 = np.random.randn()   # generates a random number via a normal distribution
w2 = np.random.randn()
b = np.random.randn()

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_p(x):   # derivative of sigma
    return sigmoid(x) * (1-sigmoid(x))

T = np.linspace(-5, 5, 100)  # -5 to 5 with 10 subdivisions


learning_rate = 0.1
costs = []

# training loop
for i in range(1, 100):
    ri = np.random.randint(len(data))
    point = data[ri]    # gets a random index of our data

    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)

    target = point[2]  # 0 or 1
    cost = np.square(pred - target)

    costs.append(cost)
    dcost_pred = 2 * (pred - target)   # derivatives calculations
    dpred_dz = sigmoid_p(z)

    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
    dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
    dcost_db = dcost_pred * dpred_dz * dz_db

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db

plt.plot(costs)

# scatter data
# for i in range(len(data)):
#     point = data[i]
#     color = "r"
#     if point[2] == 0:
#         color = "b"
#     plt.scatter(point[0], point[1], c = color)
# a = [1,1.2,1.3,1.4,1.5,1.6, 1.65]
# b = [2,5,8,15,30,45,70]
# dfScaler = preprocessing.MinMaxScaler()
# exDF = pd.DataFrame(a)
# exDF.rename(columns = {0:"a"}, inplace = True)
# exDF["b"] = b
#
# print exDF
#
# exDF_norm = normalizeDF(exDF, dfScaler)
#
# print exDF_norm


plt.show()