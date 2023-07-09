

X,y = make_blobs(n_samples=100,n_features=2,centers=2,random_state=0)
def model(X,W,b):
    Z=X.dot(W) + b
    A = 1/(1+np.exp(-Z))
    return A
A = model(X,W,b)
A.shape 
