import numpy as np 
from sklearn.cross_validation import train_test_split, cross_val_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import svm 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def name_count(name):
    arr = np.zeros(52+26*26+3)
    # Iterate each character
    name = str(name)
    for ind, x in enumerate(name):
        arr[ord(x)-ord('a')] += 1
        arr[ord(x)-ord('a')+26] += ind+1
    # Iterate every 2 characters
    '''
    for x in range(len(name)-1):
        #arr = np.array.reshape(1, -1)
        ind = (ord(name[x])-ord('a'))*26 + (ord(name[x+1])-ord('a')) + 52
        #print (x,ind)
    arr[ind] += 1
    '''
    
    # Last character
    arr[-3] = ord(name[-1])-ord('a')
    # Second Last character
    arr[-2] = ord(name[-2])-ord('a')
    # Length of name
    arr[-1] = len(name)
    
    return arr

#print (name_count('Pavan'))
#my_data = np.genfromtxt('service/Gender_Prediction/yob2014.txt', delimiter=',',  dtype=[('name','S50'), ('gender','S1'),('count','i4')], converters={0: lambda s:s.lower()})
my_data = np.genfromtxt('service/Gender_Prediction/yob2014.txt', delimiter=',',  dtype=[('name','S50'), ('gender','S1')], converters={0: lambda s:s.lower()})
#my_data = np.array([row for row in my_data if row[2]>=20])
my_data = np.array([row for row in my_data])
name_map = np.vectorize(name_count, otypes=[np.ndarray])
Xname = my_data['name']
Xlist = name_map(Xname)
X = np.array(Xlist.tolist())
y = my_data['gender']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=150, min_samples_split=20)
clf.fit(Xtr, ytr)
idx = np.random.choice(np.arange(len(Xlist)), 10, replace=False)
#xs = Xname[idx]
xs = ['Pavan','prajwal','thirumalaisamy','satya','arpitha','Jade','July','arihant','shoaib','savya']
ys = y[idx]
pred = clf.predict(X[idx])
print ("Train Accuracy :: ", accuracy_score(ytr, clf.predict(Xtr)))
print ("Test Accuracy  :: ", accuracy_score(yte, clf.predict(Xte)))

for a,b, p in zip(xs,ys, pred):
    print (str(a),str(b), str(p))