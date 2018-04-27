import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt import solvers
from sklearn.metrics import f1_score
#from scikitlearn import GridSearchCV,SVC


def fit(x, y):
    gammap=np.empty([len(y),10])
    gammam=np.empty([len(y),10])
    C=np.empty(len(x))
    
    for i in range(0,len(y)):
        for j in range(0,10):
            gammap[i][j]=0.1
            gammam[i][j]=0.1
     
    N = len(x)
    NUM = x.shape[0]
    for i in range(0,N):
        C[i]=100
    
    K = (y[:, None] * x)
    K=K+gammap+gammam
    
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((NUM, 1)))
    G = matrix(-np.eye(NUM))
    h = matrix(np.zeros(NUM))
    y = y.astype(np.double)
    
    A = matrix(y.reshape(1,-1))
    z = np.zeros(1)
    b = matrix(z)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    #alphas = np.array(sol['x'])
    
    alpha = np.array(sol['x']).reshape(N)
    support_ = [i for i in xrange(N) if alpha[i] > 1e-2]
    w = (x * (alpha * y)[:, np.newaxis]).sum(axis=0)
    for i in xrange(N):
        if 0 < alpha[i] < 1:
            bias = y[i] - np.dot(w, x[i])
            break
    return w,bias       

def predict(data,w,bias):
    if len(data.shape) <= 1:
        predict(data.reshape(1, data.shape[0]))
    return np.sign(np.dot(data,w) + bias)   

def score(data, labels,w,bias):
    pr = predict(data,w,bias)    
    
    sc=f1_score(labels,pr, average='micro') 
    print sc*100
    correct = 0.
    N = len(data)
    for i in xrange(N):
        if pr[i] * labels[i] > 0 :
            correct+=1
    
    #print (correct/N)*100
    #return correct / N  
    return sc

def normalize_X(X):
    i = 0
    while i < X.shape[1]:
        min_i = min(X[:, i])
        max_i = max(X[:, i])
        if max_i - min_i != 0:
            X[:, i] = (X[:, i] - min_i)/(max_i - min_i)
        return X    

def getData(filename):
    f=open(filename)
    strD=[]
    for line in f:
        if any((c =='?') for c in line):
            a=1
        else:
            rowD=line.split(",")
            strD.append(rowD)
    f.close()
    feature=len(strD[0])
    floatD=np.empty([len(strD), len(strD[0])])
    i=0
    j=0
    for row in strD:
        j=0
        for col in row:
            floatD[i][j]=float(col)
            j=j+1
        i=i+1
    return floatD

def getParts(dataset):
    start=0
    end=0
    offset=int(0.1*len(dataset))
    parts=[]
    for i in range(0,9):
        end=min(len(dataset),start+offset)
        part=dataset[start:end]
        parts.append(part)
        start+=offset
    part=dataset[start:]    
    parts.append(part)
    return parts

def getScore(i,parts):
    for j in range(0,10):
        if i==j:
            test=parts[j]
        else:
            if 'train' in locals(): 
                train=np.concatenate((train, parts[j]))
            else:
                train=parts[j]
    #Calculating accuracy
    
    trainX,trainY=train[:,:-1],train[:,-1]
    testX,testY=test[:,:-1],test[:,-1]
    #params_grid = {'C': [0.00001,100000],'gamma': [0.001,0.1]}
    #grid_clf = GridSearchCV(SVC(class_weight='balanced'), params_grid)
    #w,bias = grid_clf.fit(normalize_X(trainX),trainY)
    w,bias=fit(normalize_X(trainX),trainY)
    
    return score(testX,testY,w,bias)

def getAccuracy(dataset):
    sacc=0.0
    for k in range(0,len(dataset)):
        if dataset[k][-1]==2:
            dataset[k][-1]=1
        else:
            dataset[k][-1]=-1
    np.random.shuffle(dataset)
    parts=getParts(dataset)
    for i in range(0,10):
        print "Iteration "+str(i+1)
        f=getScore(i,parts)
        sacc+=f
    fm=(sacc/10.0)*100.0
    return fm

dataset=getData('breastC.txt')
print getAccuracy(dataset) 
