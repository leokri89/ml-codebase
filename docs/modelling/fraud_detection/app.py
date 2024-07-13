from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

X, y = make_blobs(n_samples=[4,96], centers=[[5,3,3,10],[9,3,6,11]], n_features=4, random_state=0, shuffle="True")

clf = IsolationForest(n_estimators=300,max_samples=10,random_state=0,max_features=4,contamination=0.1)
clf.fit(X)

y_pred = clf.predict(X)
y_pred[y_pred == -1] = 0

print("The accuracy to detect fraud is {:.1f}  %" .format( accuracy_score(y,y_pred) * 100 ) )
print(confusion_matrix(y, y_pred))

def frauddetection(trans):
    transaction_type=(clf.predict([trans]))
    if  transaction_type[0] < 0:
        print("Suspect fraud")
    else:
        print("Normal transaction")
    return