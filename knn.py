import numpy as np
def euclidian_dist(x,X):
    return np.sqrt(np.sum((x-X)**2))

def KNN(X,Y,x,K=5):
    m=X.shape[0]
    val=[]
    x=x.reshape((64,))
    for i in range(m):
        xi=X[i]
        xi=xi.reshape((64,))
        dist=euclidian_dist(x, xi)
        val.append((dist,Y[i]))
    val=sorted(val,key=lambda x:x[0])[:K]
    val=np.asarray(val)
    new_vals=np.unique(val[:,1],return_counts=True)
    index=new_vals[1].argmax()
    output=new_vals[0][index]
    return output