#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pdb

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

def computeFraction( poi_messages, all_messages ):
    fraction = 0.
    if (all_messages != "NaN"):
        if (poi_messages != "NaN"):
           fraction = float(poi_messages)/float(all_messages)
    return fraction

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list_min = ['poi','salary'] # You will need to use more features
features_list = ['poi', 'salary', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'from_this_person_to_poi', 'from_poi_to_this_person']
features_list_all = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'email_address', 'from_poi_to_this_person']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#check that all elements in features_list_min, features_list and features_list_all are in data_dict[0]
for fl in features_list_min, features_list,  features_list_all:
    for i in fl:
       assert (i in data_dict["TOTAL"].keys())

#check that all elements in data_dict are equal to features_list_all
for ddl in data_dict.keys():
     assert (set(data_dict[ddl].keys())==set(features_list_all))

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 3: Create new feature(s)

for name in data_dict:
    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi
    data_dict[name]["fraction_from_poi"] = fraction_from_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi
    data_dict[name]["fraction_to_poi"] = fraction_to_poi

features_list.append("fraction_from_poi")
features_list.append("fraction_to_poi")
features_list_all.append("fraction_from_poi")
features_list_all.append("fraction_to_poi")

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

pdb.set_trace()

from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

estimators = [('norm_l1', preprocessing.Normalizer()), ('reduce_dim', PCA()), ('clf', GaussianNB())]
pipe = Pipeline(estimators)
param_grid = dict(reduce_dim__n_components=[5],
                  # clf__C=[1e3, 5e3, 1e4],
                  # clf__gamma=[0.0005, 0.001, 0.01],
                  norm_l1__norm=["l1"]
                 )
clf = GridSearchCV(pipe, param_grid=param_grid)

# estimators = [('norm_l1', preprocessing.Normalizer()), ('reduce_dim', PCA()), ('clf', RandomForestClassifier())]
# pipe = Pipeline(estimators)
# param_grid = dict(reduce_dim__n_components=[5, 10, 12],
#                   clf__min_samples_split=[2, 10, 20, 40, 50],
#                   clf__n_estimators=[10, 30, 50, 100, 1000],
#                   norm_l1__norm=["l1"])
# clf = GridSearchCV(pipe, param_grid=param_grid)

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = clf.fit(features_train, labels_train)

from sklearn import metrics
pred = clf.predict(features_test)
acc = metrics.accuracy_score(pred, labels_test)
print "acc = ", acc
pre = metrics.precision_score(pred, labels_test)
print "pre = ", pre
rec = metrics.recall_score(pred, labels_test)
print "rec = ", rec

X_train_norm = preprocessing.Normalizer(norm="l1").fit_transform(features_train)
X_test_norm  = preprocessing.Normalizer(norm="l1").fit_transform(features_test)
X_train_pca  = PCA(n_components=5).fit(X_train_norm).transform(X_train_norm)
X_test_pca   = PCA(n_components=5).fit(X_test_norm).transform(X_test_norm)
clf          = RandomForestClassifier().fit(X_train_pca, labels_train)
pred = clf.predict(X_test_pca)

acc = metrics.accuracy_score(pred, labels_test)
print "acc = ", acc
pre = metrics.precision_score(pred, labels_test)
print "pre = ", pre
rec = metrics.recall_score(pred, labels_test)
print "rec = ", rec

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
