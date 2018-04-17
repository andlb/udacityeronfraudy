#!/usr/bin/python
import sys
import pickle
import codecs
import numpy as np
import pandas as pd
import math
import time

import matplotlib.pyplot
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import  SelectKBest, f_classif
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA    
    
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Features from the dataset
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
all_features = ['poi'] + email_features + financial_features

time_decision_tree = []
time_decision_tree_tunned = []
time_knn = []
time_knn_tunned = []
time_svc = []
time_svc_tunned = []

precision = {}
accuracy  = {}
recall = {}
f1score = {}

best_clf = GaussianNB()
best_precision = 0.1

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


def analyze_outilier(data_dict):
    """
        analyze the outlier from the data_dict.
        returns:
            dict:                
                return the data_dict with the outlier that we think we should delete
    """
    try:        
        dataframe = pd.DataFrame.from_dict(data_dict, orient = 'index',dtype = 'float64')
        q1 = dataframe.quantile(q = 0.25)
        q3 = dataframe.quantile(q = 0.75)
        IQR = dataframe.quantile(q=0.75) - dataframe.quantile(q=0.25)
        outliers = dataframe[(dataframe > (q3 + 1.5 * IQR) ) | (dataframe < (q1 - 1.5*IQR) )].count(axis=1)
        outliers.sort_values(axis = 0, ascending=False, inplace=True)
        #print the first 14 outiliers
        print outliers.head(14)
        #After analysis, we will delete the outlier TOTAL
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
    
def scaling_feature(features, features_list, my_dataset):
    """
    Scale the features and update my_dataset with the new values scaled
    Return
        Numpy:
            return the features with the value scaled.
            return my_dataset with the values scaled.
    """
    scaler = MinMaxScaler()
    new_values = scaler.fit_transform(features)    
    pos_row = 0    
    for key in my_dataset.keys():        
        pos_col = 0        
        for feature in features_list[1:]:
            my_dataset[key][feature] = new_values[pos_row,pos_col]            
            pos_col += 1
        pos_row +=1
    return new_values, my_dataset


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


def print_metrics(pred, labels_test, model, n_feature):
    """
    Print the metrics Accuracy,precision, recall and f1-score and the number of features.
    """    
    accuracy = accuracy_score(pred,labels_test)
    print 'Report from {0} model. N_feature {2}. Accuracy: {1} '.format(model,accuracy,n_feature)
    print classification_report(labels_test, pred )     

def clean_precision_metric():
    """
    Initialize the global metric(precision, accuracy, recall, f1_score) variable
    """
    precision = {}    
    accuracy  = {}
    recall = {}
    f1score = {}
    

def store_precision_metric(pred, estimator, labels_test, model, n_feature):
    """
    Add the model, the number of features and the precision in the precion global variabel
    """
    global best_precision
    global best_clf
    if not model in precision:
        precision[model] = {}
        accuracy[model] = {}
        recall[model] = {}
        f1score[model] = {}
    ac_precision = precision_score(labels_test,pred,average='weighted')
    precision[model][n_feature] = ac_precision
    accuracy[model][n_feature] = accuracy_score(labels_test,pred)    
    recall[model][n_feature] = recall_score(labels_test,pred,average='weighted')
    f1score[model][n_feature] = f1_score(labels_test,pred,average='weighted')
    if ac_precision > best_precision:
        best_precision = ac_precision
        best_clf = estimator

def bar_chart_algorithm(title,n_feature):     
    """
    Show a bar chart algorithm with the metrics result by model - 
    GaussianNB, Decision_tree,SVC, KNN
    """
    models = ['GaussianNB','Decision_tree','SVC','KNN']    
    bar_width = 0.20
    index = np.arange(len(models))
    colors = ['g','r','c','b']
    count = 0
    for model in models:
        valores = (precision[model][n_feature], accuracy[model][n_feature], recall[model][n_feature], f1score[model][n_feature])        
        between = bar_width
        if count==0:
            between = 0
        index = index + between
        matplotlib.pyplot.bar(index, valores, width=bar_width,label=model)
        count += 1
    matplotlib.pyplot.xlabel('Models')
    matplotlib.pyplot.ylabel('Metrics')
    matplotlib.pyplot.title('Models by Metrics')
    index = np.arange(len(models))
    matplotlib.pyplot.xticks(index + bar_width, ('Precision', 'Accuracy', 'Recall', 'F1-score'))    
    matplotlib.pyplot.legend(bbox_to_anchor=(0.99,1), loc="upper left")
    
    matplotlib.pyplot.show()            

def prec_chart_line_by_model(title):
    """
    Show the chart line of the precision colected from the 'GaussianNB','Decision_tree','SVC','KNN' models
    for 2, 4, 7, 9 features.
    """
    models = ['GaussianNB','Decision_tree','SVC','KNN']
    colors = ['g','r','c','b']
    n_features_options = [2, 4, 7, 9]    
    count = 0
    for model in models:
        array_precision = []
        
        for n_feature in n_features_options:
            try:
                 precision[model][n_feature]
            except KeyError:
                print "error: Model {0} with the feature {1}  are not present".format(model,n_feature)
                return            
            value_precision = precision[model][n_feature]
            array_precision.append(value_precision)               
        matplotlib.pyplot.plot(n_features_options,array_precision, color=colors[count], label=model)
        count += 1
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.axis([2, 9, 0.7, 1])
    matplotlib.pyplot.legend(loc='upper right')
    matplotlib.pyplot.xlabel('Number of features')
    matplotlib.pyplot.ylabel('Precision')
    matplotlib.pyplot.show()


def print_best_estimator(grid_search,model):
    try:
            grid_search.best_estimator_
    except KeyError:
        print "error: Model {0} with the feature {1}  are not present".format(model,n_feature)
        return  
    print "The best estimator parameter for the model {0} is {1}".format(model,grid_search.best_estimator_)


def best_classify_algorithm(features_train, features_test, labels_train, labels_test, n_feature):    
    """
    Print the metrics accuracy, precision, recall and F1-SCORE from the GaussianNB, Decision Tree, SVC, KNeighbors classifier algorithm:
    """
    clf = GaussianNB()    
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)    
    print_metrics(pred, labels_test, "GaussianNB",n_feature)
    store_precision_metric(pred, clf,labels_test, "GaussianNB", n_feature)

    pipeline = Pipeline([('decision_tree', DecisionTreeClassifier(random_state = 42))])
    parameters = [{'decision_tree__min_samples_split': [2,3,4], 'decision_tree__criterion': ['gini', 'entropy']}]
    start = time.time()
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)
    time_decision_tree_tunned.append(time.time() - start)        
    print_metrics(pred, labels_test, "Decision_tree tunned", n_feature)    
    store_precision_metric(pred, grid_search.best_estimator_, labels_test, "Decision_tree", n_feature)
    print_best_estimator(grid_search,"Decision_tree")

    start = time.time()
    clf = DecisionTreeClassifier(random_state = 42)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    time_decision_tree.append(time.time() - start)
    print_metrics(pred, labels_test, "Decision_tree ", n_feature)    
    
    c = np.linspace(0.1,10,10)
    gamma  = np.linspace(0.01,1,10)    
    parameters = [{'svc__kernel': ['linear','rbf','poly','sigmoid'],
                    'svc__C': c,
                    'svc__gamma': gamma 
                    }]
    pipeline = Pipeline([('svc', SVC(random_state = 42))])
    start = time.time()
    grid_search = GridSearchCV(estimator = pipeline, param_grid = parameters)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)
    time_svc_tunned.append(time.time() - start)    
    print_metrics(pred, labels_test, "SVC Tunned", n_feature)    
    store_precision_metric(pred, grid_search.best_estimator_, labels_test, "SVC", n_feature)
    print_best_estimator(grid_search,"SVC")
    
    start = time.time()
    clf = SVC(random_state = 42)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    time_svc.append(time.time() - start)
    print_metrics(pred, labels_test, "SVC ", n_feature)
    
    k = np.arange(4) + 1
    parameters = [{'knn__n_neighbors': k, 'knn__weights':['uniform','distance']}]
    pipeline = Pipeline([('knn',KNeighborsClassifier())])
    start = time.time()
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)
    time_knn_tunned.append(time.time() - start)
    print_metrics(pred, labels_test, "KNN Tunned", n_feature)    
    store_precision_metric(pred, grid_search.best_estimator_, labels_test, "KNN", n_feature)
    print_best_estimator(grid_search,"knn")

    start = time.time()
    clf = KNeighborsClassifier()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    time_knn.append(time.time() - start)
    store_precision_metric(pred, clf, labels_test, "KNNnoTunned", n_feature)
    print_metrics(pred, labels_test, "KNN ", n_feature)
    

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
    data_dict = analyze_outilier(data_dict)
    #THE AGENCY IS NOT A EMPLOYER, AND WE ARE GOING TO DELETE.
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
    print_scatter(data_dict,features_list,1,2)
    #create new feature    
    my_dataset, features_list  = create_feature(data_dict,features_list)
    #show the reason to scale.
    key = 'HANNON KEVIN P'
    print "Index {0} : Salary {1:.2f},  Total stock value {2:.2f}".format(key, float(data_dict[key]['salary']), float(data_dict[key]['total_stock_value']))

    data = featureFormat(my_dataset, features_list,remove_all_zeroes = False, sort_keys = False)
    labels, features = targetFeatureSplit(data)
    #scaling the features
    features, my_dataset = scaling_feature(features, features_list, my_dataset)

    data = featureFormat(my_dataset, features_list, remove_all_zeroes=True, sort_keys = False)
    labels, features = targetFeatureSplit(data) 
    
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.4, random_state=12)

    #print the metrics prediction for all features.
    print "All features"
    best_classify_algorithm(features_train, features_test, labels_train, labels_test,len(features_list))
    print "precision {0}".format(precision)
    clean_precision_metric()
    #reduce the number of features using the selectkbest function
    print "Selectk best"
    n_features_options = [2, 4, 7, 9]
    for n_feature in n_features_options:
        print "k features {0}".format(n_feature)
        new_features = SelectKBest(f_classif, k=n_feature).fit_transform(features, labels)        
        features_train, features_test, labels_train, labels_test = \
            train_test_split(new_features, labels, test_size=0.4, random_state=12)
        best_classify_algorithm(features_train, features_test, labels_train, labels_test, n_feature)
    print ""    
    prec_chart_line_by_model('SelectKBest Performance')
    clean_precision_metric()
    #reduce the number of features using the pca function
    print "PCA - Principal component analysis"
    for n_feature in n_features_options:
        print "k features {0}".format(n_feature)
        pca = PCA(n_components=n_feature, whiten=True,svd_solver='randomized').fit(features)
        new_features = pca.transform(features)
        features_train, features_test, labels_train, labels_test = \
            train_test_split(new_features, labels, test_size=0.4, random_state=12)
        best_classify_algorithm(features_train, features_test, labels_train, labels_test, n_feature)

    print ""
    prec_chart_line_by_model('PCA Performance')
    print "Average time"

    print "Decision Tree average {0} ".format(np.average(time_decision_tree))
    print "Decision Tree tunned average {0} ".format(np.average(time_decision_tree_tunned))
    print "SVC average {0} ".format(np.average(time_svc))
    print "SVC tunned average {0} ".format(np.average(time_svc_tunned))
    print "KNN average {0} ".format(np.average(time_knn))
    print "KNN tunned average {0} ".format(np.average(time_knn_tunned))
        
    selectkbest = SelectKBest(f_classif, k=2)
    selectkbest.fit(features, labels)
    new_features = selectkbest.transform(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(new_features, labels, test_size=0.4, random_state=12)    
    supported_list = selectkbest.get_support()
    #creating the feature list after run the selectkbestfeature.
    features_list = ['poi'] + [features_list[x+1] for x in range(0,len(supported_list)) if supported_list[x]==True ]
    print "Feature list after apply the selectkbest with 2 features: {0}".format(features_list)

    clean_precision_metric()
    best_classify_algorithm(features_train, features_test, labels_train, labels_test, 2)
    bar_chart_algorithm("Algorithm with the best number of features - 2",2)   
        
    # Return the best algorithm    
    dump_classifier_and_data(best_clf, my_dataset, features_list)

if __name__ == '__main__':
    data_analyse()
    