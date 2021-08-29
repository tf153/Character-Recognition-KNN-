import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from knn import KNN

data =load_digits()

X=data.images
y=data.target
plt.imshow(X[153],cmap=plt.get_cmap('gray'))
sum=0
for i in range(1700):
    sum+=(KNN(X,y,X[i]))
print(sum)