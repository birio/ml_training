#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

nb_l=[]

for nb in range(1, 20):
   clf_neigh = KNeighborsClassifier(n_neighbors=nb)
   clf_neigh.fit(features_train, labels_train)
   neigh_pred = clf_neigh.predict(features_test)
   neigh_acc = metrics.accuracy_score(neigh_pred, labels_test)
   
   nb_l.append(neigh_acc)
   print "neigh_acc = ", neigh_acc
  
nb_l_copy = nb_l[:]
nb_l.sort()
print "max = ", nb_l[-1]
print "k = ", nb_l_copy.index(nb_l[-1])+1

clf_neigh = KNeighborsClassifier(n_neighbors=8)
clf_neigh.fit(features_train, labels_train)
neigh_pred = clf_neigh.predict(features_test)
neigh_acc = metrics.accuracy_score(neigh_pred, labels_test)

print "neigh_acc = ", neigh_acc

try:
    prettyPicture(clf_neigh, features_test, labels_test, "_neigh_8")
except NameError:
    print "no prettyPicture"

#try:
#    prettyPicture(clf_ada, features_test, labels_test, "_ada")
#except NameError:
#    print "no prettyPicture"
#
# from sklearn.ensemble import RandomForestClassifier
# 
# ne_l = [10, 30, 50, 100, 1000]
# ms_l = [2, 10, 20, 40, 50]
# 
# rf_l = []
# 
# for ne in ne_l:
#    for ms in ms_l:
#       print " ne = ", ne
#       print " ms = ", ms
# 
#       clf_rf = RandomForestClassifier(min_samples_split = ms, n_estimators=ne)
#       clf_rf.fit(features_train, labels_train)
#       rf_pred = clf_rf.predict(features_test)
#       rf_acc = metrics.accuracy_score(rf_pred, labels_test)
#       
#       rf_l.append(rf_acc)
#       print "rf_acc = ", rf_acc
#    
#      try:
#          prettyPicture(clf_rf, features_test, labels_test, "_rf")
#      except NameError:
#          print "no prettyPicture"
#
#from sklearn.svm import SVC
#
#clf_svm = SVC()
#clf_svm.fit(features_train, labels_train)
#svm_pred = clf_svm.predict(features_test)
#svm_acc = metrics.accuracy_score(labels_test, svm_pred)
#print "svm_acc = ", svm_acc
#
#try:
#    prettyPicture(clf_rf, features_test, labels_test, "_svm")
#except NameError:
#    print "no prettyPicture"

# from sklearn.ensemble import AdaBoostClassifier
# 
# ne_l = [10, 30, 50, 100, 1000]
# lr_l = [0.1, 0.2, 0.5, 1, 1.5]
# 
# ada_l = []
# 
# for ne in ne_l:
#    for lr in lr_l:
#       print " ne = ", ne
#       print " lr = ", lr
#       clf_ada = AdaBoostClassifier(n_estimators= ne, learning_rate= lr) 
#       clf_ada.fit(features_train, labels_train)
#       ada_pred = clf_ada.predict(features_test)
#       ada_acc = metrics.accuracy_score(ada_pred, labels_test)
#       
#       ada_l.append(ada_acc)
#       print "ada_acc = ", ada_acc
# 
# ada_l_max = ada_l.sort()
# acc_max = ada_l[-1]
# print "acc_max = ", acc_max
# print "index[acc_max] = ", ada_l.index(acc_max)
   
