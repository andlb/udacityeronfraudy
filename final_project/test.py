import sys
import pickle
import numpy as np
import pandas as pd

with open("final_project_dataset.pkl", "r") as data_file:    
    print "entro ak"
    data_dict = pickle.load(data_file)
    dataframe = pd.DataFrame.fromdict(data_dict,orient="index")
    print "Number of data point {0}".format(len(data_dict ))
    


# 'salary'
# 'to_messages'
# 'deferral_payments'
# 'total_payments'
# 'exercised_stock_options'
# 'bonus'
# 'restricted_stock'
# 'shared_receipt_with_poi'
# 'restricted_stock_deferred'
# 'total_stock_value'
# 'expenses'
# 'loan_advances'
# 'from_messages'
# 'other'
# 'from_this_person_to_poi'
# 'poi'
# 'director_fees'
# 'deferred_income'
# 'long_term_incentive'
# 'email_address'
# 'from_poi_to_this_person'    