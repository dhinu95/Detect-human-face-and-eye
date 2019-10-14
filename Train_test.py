from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
X_train1, X_test1, y_train1, y_test1 = X_train, X_test, y_train, y_test
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

def normalize(x):
    """
    Inputs:
        x: numpy array
    Outputs:
        x_n: numpy array, elements normalized to be between (0, 1)
    """    
    x_n = (x - 0)/(255)
    return x_n    
X_train = normalize(X_train)
X_test = normalize(X_test) 

def reshape(x):
    """
    We need to reshape our train_data to be of shape (samples, height, width, channels) pass to Conv2D layer of keras
    Inputs:
        x: numpy array of shape(samples, height, width)
    Outputs:
        x_r: numpy array of shape(samples, height, width, 1)
    """
    x_r = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    return x_r
X_train = reshape(X_train)
X_test = reshape(X_test) 

def oneHot(y, Ny):
    """
    Inputs:
        y: numpy array if shape (samples, ) with class labels
        Ny: number of classes
    Outputs:
        y_oh: numpy array of shape (samples, Ny) of one hot vectors
    """
    y_oh = np.zeros((len(y), Ny))
    for i in range(len(y)):
      y_oh[i][y[i]] = 1
    return y_oh
y_train = oneHot(y_train, 6)
y_test = oneHot(y_test, 6)
