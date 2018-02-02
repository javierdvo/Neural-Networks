import numpy as np


def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))



def SGDNetwork(train_in, train_out,test_in,test_out, learnRate, hiddenUnits, features, epochs):
    print("\nStochastic Gradient Descent with parameters:")
    print("Learn Rate: "+str(learnRate))
    print("Hidden Units: " + str(hiddenUnits))
    print("Features: " + str(features))
    print("Epochs: "+str(epochs))
    W1 = np.ones([hiddenUnits, features]) * 0.01
    W2 = np.ones(hiddenUnits) * 0.01
    a = np.zeros([hiddenUnits])
    z = np.zeros([hiddenUnits])
    aux = np.zeros([hiddenUnits, features])
    results=np.zeros(len(train_out))
    for t in range(0, epochs):
        err=0
        for m in range(0,len(train_out)):
            for j in range(0, hiddenUnits):
                a[j] = np.sum(W1[j, :].dot(train_in[:,m]))
            z=sigmoid(a)
            ak = W2.dot(z)
            zk = sigmoid(ak)
            deltak = dsigmoid(zk) * (train_out[m] - zk)
            deltaj = dsigmoid(z) * (W2.dot(deltak))
            np.outer(deltaj,train_in[:,m],aux)
            W2 = W2 + learnRate * a.dot(deltak)
            W1 = W1 + learnRate * aux
            err=err+deltak
            if(zk)>0.5:
                results[m]=1
            else:
                results[m]=0
        if t in [9,19,29,39,49,59,69,79,89,99]:
            print("\nEpoch:" + str(t+1))
            print("Correct Predictions on train: " + str(np.sum(results==train_out)))
            print("Accuracy on train:"+ str(np.sum(results==train_out)/len(train_out)))
    b = np.zeros([hiddenUnits])
    output=np.zeros(len(test_out))
    for n in range(0,len(test_out)):
        for jb in range(0,hiddenUnits):
            b[jb]=np.sum(W1[jb, :].dot(test_in[:,n]))
        zb=sigmoid(b)
        bk=W2.dot(zb)
        zkb=sigmoid(bk)
        if (zkb) > 0.5:
            output[n] = 1
        else:
            output[n] = 0
    print("\nCorrect Predictions on test: "+str(np.sum(output == test_out)))
    print("Accuracy on test: "+str(np.sum(output == test_out) / len(test_out)))




def SBDNetwork(train_in, train_out,test_in,test_out, learnRate, hiddenUnits, features, epochs):
    print("\nBatch Gradient Descent with parameters:")
    print("Learn Rate: "+str(learnRate))
    print("Hidden Units: " + str(hiddenUnits))
    print("Features: " + str(features))
    print("Epochs: "+str(epochs))
    W1 = np.ones([hiddenUnits, features]) * 0.01
    W2 = np.ones(hiddenUnits) * 0.01
    a = np.zeros([hiddenUnits])
    z = np.zeros([hiddenUnits])
    aux = np.zeros([hiddenUnits, features])
    results=np.zeros(len(train_out))
    for t in range(0, epochs):
        err=0
        W1aux = np.zeros([hiddenUnits, features])
        W2aux = np.zeros(hiddenUnits)
        for m in range(0,len(train_out)):
            for j in range(0, hiddenUnits):
                a[j] = np.sum(W1[j, :].dot(train_in[:,m]))
            z=sigmoid(a)
            ak = W2.dot(z)
            zk = sigmoid(ak)
            deltak = dsigmoid(zk) * (train_out[m] - zk)
            deltaj = dsigmoid(z) * (W2.dot(deltak))
            np.outer(deltaj,train_in[:,m],aux)
            W1aux=W1aux+aux
            W2aux=W2aux+a.dot(deltak)
            err=err+deltak
            if(zk)>0.5:
                results[m]=1
            else:
                results[m]=0
        W2 = W2 + learnRate * W2aux / len(train_out)
        W1 = W1 + learnRate * W1aux / len(train_out)
        if t in [9,19,29,39,49,59,69,79,89,99]:
            print("\nEpoch:" + str(t+1))
            print("Correct Predictions on train: " + str(np.sum(results==train_out)))
            print("Accuracy on train:"+ str(np.sum(results==train_out)/len(train_out)))
    b = np.zeros([hiddenUnits])
    output=np.zeros(len(test_out))
    for n in range(0,len(test_out)):
        for jb in range(0,hiddenUnits):
            b[jb]=np.sum(W1[jb, :].dot(test_in[:,n]))
        zb=sigmoid(b)
        bk=W2.dot(zb)
        zkb=sigmoid(bk)
        if (zkb) > 0.5:
            output[n] = 1
        else:
            output[n] = 0
    print("\nCorrect Predictions on test: "+str(np.sum(output == test_out)))
    print("Accuracy on test: "+str(np.sum(output == test_out) / len(test_out)))


features = 784
hiddenUnits = int(np.round(1.2 * features))
learnRate = 0.3
epochs = 100

test_in = np.loadtxt("mnist_small_test_in.txt", delimiter=',')
test_outpre = np.loadtxt("mnist_small_test_out.txt", delimiter=',')
train_in = np.loadtxt("mnist_small_train_in.txt", delimiter=',')
train_outpre = np.loadtxt("mnist_small_train_out.txt", delimiter=',')
np.set_printoptions(threshold=np.nan)
indexFives = np.where(test_outpre == 5)
test_out = np.zeros(len(test_outpre))
test_out[indexFives] = 1
indexFives = np.where(train_outpre == 5)
train_out = np.zeros(len(train_outpre))
train_out[indexFives] = 1

SGDNetwork(train_in.transpose(), train_out.transpose(),test_in.transpose(),test_out.transpose(), learnRate, hiddenUnits, features, epochs)
SBDNetwork(train_in.transpose(), train_out.transpose(),test_in.transpose(),test_out.transpose(), learnRate, hiddenUnits, features, epochs)