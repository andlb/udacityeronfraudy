#!/usr/bin/python
import sys
import pickle
import codecs
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import  SelectKBest, f_classif
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Features from the dataset
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
all_features = ['poi'] + email_features + financial_features


def load_data():
    """ Load the dictionary containing the dataset     
    Returns:
        dict:
            the dictionary key is the person name 
                for each person name, a dictionary with these keys:
                'salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus',,'restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','poi','director_fees','deferred_income','long_term_incentive','email_address','from_poi_to_this_person'    
    """    
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    return data_dict

def count_nan(dictionary,features):
    """
    count the number of nan.
    returns:
        dict:
          the key is the feature name and the value is the number of NA
    """
    return_list = {}
    keys = dictionary.keys()
    for key in keys:
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key {0} not present".format(feature)
                return
            value = dictionary[key][feature]
            if value == "NaN":
                if not feature in return_list.keys():
                    return_list[feature] = 0
                return_list[feature] = return_list[feature] + 1
    return return_list

def get_number_poi(dictionary):
    """
    sum the number of poi
    return 
        int: 
            Number of True poi
    """
    poi = 0
    keys = dictionary.keys()
    for key in keys:
        if dictionary[key]['poi']:
            poi += 1
    return poi


def delete_outilier(data_dict):
    """
        delete the outlier from the data_dict.
        returns:
            dict:
                return the data_dict without the outlier
    """
    try:        
        dataframe = pd.DataFrame.from_dict(data_dict, orient = 'index',dtype = 'float64')
        q1 = dataframe.quantile(q = 0.25)
        q3 = dataframe.quantile(q = 0.75)
        IQR = dataframe.quantile(q=0.75) - dataframe.quantile(q=0.25)
        outliers = dataframe[(dataframe > (q3 + 1.5 * IQR) ) | (dataframe < (q1 - 1.5*IQR) )].count(axis=1)
        outliers.sort_values(axis = 0, ascending=False, inplace=True)
        print outliers.head(14)
        #we are going to remove only the total.        
        data_dict.pop(outliers.keys()[0],0)
        return data_dict
    except KeyError:
        print "error: key ", KeyError
        return ""
    return ""

def print_scatter(data_dict,features_list, x_position,y_position):
    """
    print the scatter plot 
    """
    data = featureFormat(data_dict, features_list)

    for point in data:
        xvalue = point[x_position]
        yvalue = point[y_position]     
        color = 'red'
        if point[0]:
            color = 'blue'
        matplotlib.pyplot.scatter( xvalue, yvalue, c = color  )
    matplotlib.pyplot.xlabel(features_list[x_position])
    matplotlib.pyplot.ylabel(features_list[y_position])
    matplotlib.pyplot.show()
    
def scaling_feature(features):
    """
    Scale the features 
    Return
        Numpy:
            return the features value scaled.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)


def compute_fraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if all_messages > 0 and poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = float(poi_messages) / float(all_messages)
    return fraction   

def create_feature(data_dict, features_list):
    """
        Add a new feature in the data_dict named total_income.
        returns
            dict:
                return the data_dict with the new feature fraction_from_poi and fraction_to_poi.
                return the features_list with the new features names.
            
    """

    my_dataset = data_dict
    keys = my_dataset.keys()
    for key in keys:
        my_dataset[key]["fraction_from_poi"] = compute_fraction(my_dataset[key]['from_poi_to_this_person'], my_dataset[key]['to_messages'])
        my_dataset[key]["fraction_to_poi"] = compute_fraction(my_dataset[key]["from_this_person_to_poi"], my_dataset[key]["from_messages"])                                     
    features_list.append("fraction_from_poi")
    features_list.append("fraction_to_poi")
    
    print_scatter(my_dataset,['poi','fraction_from_poi','fraction_to_poi'], 1,2)
    return my_dataset, features_list


def metrics(pred, labels_test, model, n_feature):
    """
    Print the metrics Accuracy,precision, recall and f1-score and the number of features.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    accuracy = accuracy_score(pred,labels_test)
    print 'Report from {0} model. N_feature {2}. Accuracy: {1} '.format(model,accuracy,n_feature)
    print classification_report(labels_test, pred )
    
   
def analysis_tune_parameters(features_train, features_test, labels_train, labels_test, n_feature):    
    """
    Print the metrics from the SVC algorithm before and after tunning the algorithm.
    The parameters tunned were
        kernel': ['linear','rbf']
        C': [0.001, 0.01, 0.1, 1, 10]
        gamma': [0.001, 0.01, 0.1, 1]          
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    clf = SVC(random_state = 42)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    metrics(pred, labels_test, "No Tune decision_tree", n_feature)

    parameters = [{'svc__kernel': ['linear','rbf'],
                    'svc__C': [0.001, 0.01, 0.1, 1, 10],
                    'svc__gamma': [0.001, 0.01, 0.1, 1]      
                    }]
    pipeline = Pipeline([('svc',SVC(random_state = 42))])
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)
    metrics(pred, labels_test, "SVC", n_feature)
    print grid_search.best_estimator_

def best_algorithm(features_train, labels_train):    
    """
    The best classifier algorithm finded to fit and predict this dataset.
    Return
        sklearn.svm.SVC
            The classifier after fit the trainning data
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    clf = SVC(C=10, kernel='rbf', gamma=1,random_state=42)    
    clf.fit(features_train,labels_train)    
    return clf


def best_classify_algorithm(features_train, features_test, labels_train, labels_test, n_feature):    
    """
    Print the metrics accuracy, precision, recall and F1-SCORE from the GaussianNB, Decision Tree, SVC, KNeighbors classifier algorithm:
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA    
        
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    clf = GaussianNB()    
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)    
    metrics(pred, labels_test, "GaussianNB",n_feature)

    pipeline = Pipeline([('decision_tree',DecisionTreeClassifier(random_state = 42))])
    parameters = [{'decision_tree__min_samples_split': [2,3,4], 'decision_tree__criterion': ['gini', 'entropy']}]
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)
    metrics(pred, labels_test, "Decision_tree", n_feature)
  
    parameters = [{'svc__kernel': ['linear','rbf'],
                    'svc__C': [0.001, 0.01, 0.1, 1, 10],
                    'svc__gamma': [0.001, 0.01, 0.1, 1]      
                    }]
    pipeline = Pipeline([('svc',SVC(random_state = 42))])
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)
    metrics(pred, labels_test, "SVC", n_feature)
    
    k = [1,3,5,7,10]
    parameters = [{'knn__n_neighbors': k}]
    pipeline = Pipeline([('knn',KNeighborsClassifier())])
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)
    metrics(pred, labels_test, "KNN", n_feature)    

def data_analyse():
    """
    Generate informaton about the dataset and find the best indentifier.
    """
    features_list = ['poi','salary','expenses','total_payments','exercised_stock_options','bonus','restricted_stock','total_stock_value','from_poi_to_this_person','shared_receipt_with_poi','from_messages'] # You will need to use more features
    data_dict = load_data()
    #get information about the dataset
    list_nan = count_nan(data_dict,all_features)
    total_data_point = len(data_dict )
    poi = get_number_poi(data_dict)
    nonpoi = total_data_point - poi 
    #print the information about the dataset
    print "Number of data point {0}".format(total_data_point)
    print "Number of poi {0}".format(poi)
    print "Number of features {0}".format(len(all_features))
    print "Number of features used {0}".format(len(features_list))
    print "Allocation across classes {0:.2f}".format(float(poi)/nonpoi)
    print "List of nan "
    for key in list_nan.keys():
        print "{0} : {1} - {2:.2f} %".format(key, list_nan[key], float(list_nan[key])/len(data_dict)*100)

    ### Task 2: Remove outliers
    print_scatter(data_dict,features_list,1,2)
    data_dict = delete_outilier(data_dict)
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
    print_scatter(data_dict,features_list,1,2)
    #create new feature
    my_dataset, features_list  = create_feature(data_dict,features_list)
    #show the reason to scale.
    key = 'HANNON KEVIN P'
    print "Index {0} : Salary {1:.2f},  Total stock value {2:.2f}".format(key, float(data_dict[key]['salary']), float(data_dict[key]['total_stock_value']))

    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    #scaling the features
    features = scaling_feature(features)
    #split to trainning and testing data.
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.4, random_state=12)
    #print the metrics prediction for all features.
    print "All features"
    best_classify_algorithm(features_train, features_test, labels_train, labels_test,len(features_list))
    #reduce the number of features using the selectkbest function
    print "Selectk best"
    n_features_options = [2, 4, 7, 9]
    for n_feature in n_features_options:
        print "k features {0}".format(n_feature)
        new_features_train = SelectKBest(f_classif, k=n_feature).fit_transform(features_train, labels_train)
        new_features_test = SelectKBest(f_classif, k=n_feature).fit_transform(features_test, labels_test)
        best_classify_algorithm(new_features_train, new_features_test, labels_train, labels_test, n_feature)        
    print ""    
    #reduce the number of features using the pca function
    print "PCA - Principal component analysis"
    for n_feature in n_features_options:
        print "k features {0}".format(n_feature)
        pca = PCA(n_components=n_feature, whiten=True,svd_solver='randomized').fit(features_train)
        new_features_train = pca.transform(features_train)
        new_features_test = pca.transform(features_test)
        best_classify_algorithm(new_features_train, new_features_test, labels_train, labels_test, n_feature)
    print ""

    print "SVC tunning analysis"
    selectkbest = SelectKBest(f_classif, k=2)
    selectkbest.fit(features_train, labels_train)
    supported_list = selectkbest.get_support()
    #creating the feature list after run the selectkbestfeature.
    features_list = [features_list[x+1] for x in range(0,len(supported_list)) if supported_list[x]==True ]
    new_features_train = selectkbest.transform(features_train)
    new_features_test = SelectKBest(f_classif, k=2).fit_transform(features_test, labels_test)
    analysis_tune_parameters(new_features_train, new_features_test, labels_train, labels_test, 2)
    # Return the best algorithm
    clf = best_algorithm(new_features_train, labels_train)
    dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ == '__main__':
    data_analyse()