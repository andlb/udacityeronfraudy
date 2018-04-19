#classification_report(labels_test, pred )     


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
best_precision = 0.0
best_recall = 0.0
N_FEATURES = 12

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

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






def test_classifier(clf, dataset, feature_list, folds, model, n_feature):

    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
    
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        t_accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        t_precision = 1.0*true_positives/(true_positives+false_positives)
        t_recall = 1.0*true_positives/(true_positives+false_negatives)
        t_f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        t_f2 = (1+2.0*2.0) * t_precision*t_recall/(4*t_precision + t_recall)                
    except:        
        print 'Error: {0} model. N_feature {1}.'.format(model,n_feature)        
        t_accuracy = 0.0
        t_precision = 0.0
        t_recall = 0.0
        t_f1 = 0.0
        t_f2 = 0.0

    best_class = store_verify_precision_metric(clf, model, n_feature, t_precision, t_accuracy, t_recall, t_f1)
    print ""
    print 'Report from {0} model. N_feature {1}.'.format(model,n_feature)
    print PERF_FORMAT_STRING.format(t_accuracy, t_precision, t_recall, t_f1, t_f2, display_precision = 5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    print ""
    return best_class

def clean_precision_metric():
    """
    Initialize the global metric(precision, accuracy, recall, f1_score) variable
    """
    precision = {}    
    accuracy  = {}
    recall = {}
    f1score = {}
    

def store_verify_precision_metric(estimator, model, n_feature, p_precision, p_accuracy, p_recall, p_f1score ):
    """
    Add the model, the number of features and the precision in the precion global variabel
    return
        bollean:
            return True when the precision is the best
    """
    global best_precision
    global best_recall
    global best_clf
    if not model in precision:
        precision[model] = {}
        accuracy[model] = {}
        recall[model] = {}
        f1score[model] = {}
    precision[model][n_feature] = p_precision
    accuracy[model][n_feature] = p_accuracy
    recall[model][n_feature] = p_recall
    f1score[model][n_feature] = p_f1score
    if p_precision > 0.3 and p_recall > 0.3 and p_precision > best_precision and p_recall > best_recall:
        best_precision = p_precision
        best_recall = p_recall
        best_clf = estimator
        return True
    return False

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
    n_features_options = [2, 4, 7, 9, 13]        
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
    matplotlib.pyplot.axis([2, 13, 0, 1.1])
    matplotlib.pyplot.legend(loc='upper right')
    matplotlib.pyplot.xlabel('Number of features')
    matplotlib.pyplot.ylabel('Precision')
    matplotlib.pyplot.show()

def rec_chart_line_by_model(title):
    """
    Show the chart line of the recall colected from the 'GaussianNB','Decision_tree','SVC','KNN' models
    for 2, 4, 7, 9 features.
    """
    models = ['GaussianNB','Decision_tree','SVC','KNN']
    colors = ['g','r','c','b']
    n_features_options = [2, 4, 7, 9, 13]    
    print precision
    count = 0    
    for model in models:
        array_precision = []        
        for n_feature in n_features_options:
            try:
                 recall[model][n_feature]
            except KeyError:
                print "error: Model {0} with the feature {1}  are not present".format(model,n_feature)
                return            
            value_precision = recall[model][n_feature]
            array_precision.append(value_precision)               
        matplotlib.pyplot.plot(n_features_options,array_precision, color=colors[count], label=model)
        count += 1
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.axis([2, 9, 0.0, 1])
    matplotlib.pyplot.legend(loc='upper right')
    matplotlib.pyplot.xlabel('Number of features')
    matplotlib.pyplot.ylabel('Recall')
    matplotlib.pyplot.show()

def print_best_estimator(grid_search,model):
    try:
            grid_search.best_estimator_
    except KeyError:
        print "error: Model {0} with the feature {1}  are not present".format(model,n_feature)
        return  
    print "The best estimator parameter for the model {0} is {1}".format(model,grid_search.best_estimator_)


def best_classify_algorithm(features_train, features_test, labels_train, labels_test, n_feature, feature_list, my_dataset):
    """
    Print the metrics accuracy, precision, recall and F1-SCORE from the GaussianNB, Decision Tree, SVC, KNeighbors classifier algorithm:
    return: 
        boolean
            return True when the best precision is found
    """
    l_best_precision = False
    K_FOLD = 500
    clf = GaussianNB()    
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)    
    ver = test_classifier(clf, my_dataset, feature_list, K_FOLD, "GaussianNB", n_feature)    
    if ver:
        l_best_precision = True

    pipeline = Pipeline([('decision_tree', DecisionTreeClassifier(random_state = 42))])
    parameters = [{'decision_tree__min_samples_split': [2,3,4], 'decision_tree__criterion': ['gini', 'entropy']}]
    start = time.time()
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)
    time_decision_tree_tunned.append(time.time() - start)            
    ver = test_classifier(grid_search.best_estimator_, my_dataset, feature_list,K_FOLD, "Decision_tree", n_feature)
    if ver:
        l_best_precision = True        

    start = time.time()
    clf = DecisionTreeClassifier(random_state = 42)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    time_decision_tree.append(time.time() - start)
    test_classifier(clf, my_dataset, feature_list, 300, "Decision_treeNoTunned", n_feature)
    
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
    ver = test_classifier(grid_search.best_estimator_, my_dataset, feature_list, K_FOLD, "SVC", n_feature)
    if ver:
        l_best_precision = True
  
    start = time.time()
    clf = SVC(random_state = 42)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    time_svc.append(time.time() - start)
    test_classifier(clf, my_dataset, feature_list,300, "SVCNoTunned", n_feature)
    
    
    k = np.arange(4) + 2
    parameters = [{'knn__n_neighbors': k, 'knn__weights':['uniform','distance']}]
    pipeline = Pipeline([('knn',KNeighborsClassifier())])
    start = time.time()
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)
    time_knn_tunned.append(time.time() - start)
    ver = test_classifier(grid_search.best_estimator_, my_dataset, feature_list, K_FOLD, "KNN", n_feature)    
    if ver:
        l_best_precision = True    
    #print_best_estimator(grid_search,"knn")

    start = time.time()
    clf = KNeighborsClassifier()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    time_knn.append(time.time() - start)
    test_classifier(clf, my_dataset, feature_list, K_FOLD, "KNNnoTunned", n_feature)

    
    return l_best_precision

def data_analyse():
    """
    Generate informaton about the dataset and find the best indentifier.
    """
    TEST_SIZE = 0.1
    global N_FEATURES
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
    #my_dataset = increase_size_dataset(data_dict, 20)
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
        train_test_split(features, labels, test_size=TEST_SIZE, random_state=12)

    #print the metrics prediction for all features.
    print "All features"
    N_FEATURE = len(features_list)
    
    #reduce the number of features using the selectkbest function
    print "Selectk best"
    n_features_options = [2, 4, 7, 9]
    final_feature_list = features_list
    for n_feature in n_features_options:
        print "k features {0}".format(n_feature)
        selectkbest = SelectKBest(f_classif, k=n_feature).fit(features, labels)                
        new_features = selectkbest.transform(features)
        supported_list = selectkbest.get_support()
        t_features_list = ['poi'] + [features_list[x+1] for x in range(0,len(supported_list)) if supported_list[x]==True ]        
        features_train, features_test, labels_train, labels_test = \
            train_test_split(new_features, labels, test_size=TEST_SIZE, random_state=12)
        ver = best_classify_algorithm(features_train, features_test, labels_train, labels_test, n_feature, t_features_list,my_dataset)
        print "is the best? {0} ".format(ver)
        print "feature list {0}".format(t_features_list)
        if ver:
            N_FEATURES = n_feature
            final_feature_list = t_features_list
    print ""    
    ver = best_classify_algorithm(features_train, features_test, labels_train, labels_test,len(features_list),features_list, my_dataset)            
    if ver:
        N_FEATURE = len(features_list)
        final_feature_list = features_list
    prec_chart_line_by_model('Precision SelectKBest Performance')
    rec_chart_line_by_model('Recall SelectKBest Performance')    
    
    #reduce the number of features using the pca function
    print "PCA - Principal component analysis"
    for n_feature in n_features_options:
        print "k features {0}".format(n_feature)
        pca = PCA(n_components=n_feature, whiten=True,svd_solver='randomized').fit(features)
        new_features = pca.transform(features)
        features_train, features_test, labels_train, labels_test = \
            train_test_split(new_features, labels, test_size=TEST_SIZE, random_state=12)
        ver = best_classify_algorithm(features_train, features_test, labels_train, labels_test, n_feature,features_list,my_dataset)

    print ""
    prec_chart_line_by_model('Precision PCA Performance')
    rec_chart_line_by_model('Recall PCA Performance')
    print "Average time"

    print "Decision Tree average {0} ".format(np.average(time_decision_tree))
    print "Decision Tree tunned average {0} ".format(np.average(time_decision_tree_tunned))
    print "SVC average {0} ".format(np.average(time_svc))
    print "SVC tunned average {0} ".format(np.average(time_svc_tunned))
    print "KNN average {0} ".format(np.average(time_knn))
    print "KNN tunned average {0} ".format(np.average(time_knn_tunned))
 
    #creating the feature list after run the selectkbestfeature.
    print "N_FEAUTRES {0}".format(N_FEATURES)
    selectkbest = SelectKBest(f_classif, k=N_FEATURES).fit(features, labels)                
    new_features = selectkbest.transform(features)
    supported_list = selectkbest.get_support()
    t_features_list = ['poi'] + [features_list[x+1] for x in range(0,len(supported_list)) if supported_list[x]==True ]        
    features_train, features_test, labels_train, labels_test = \
        train_test_split(new_features, labels, test_size=TEST_SIZE, random_state=12)    
    ver = best_classify_algorithm(features_train, features_test, labels_train, labels_test, N_FEATURES, t_features_list,my_dataset)    
    print "Feature list after apply the selectkbest: {0}".format(t_features_list)            
    bar_chart_algorithm("Algorithm with the best number of features", N_FEATURES)   
        
    # Return the best algorithm    
    dump_classifier_and_data(best_clf, my_dataset, final_feature_list)

if __name__ == '__main__':
    data_analyse()
    