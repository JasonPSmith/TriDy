# Load packages
import pandas as pd
import numpy as np
from subprocess import call
import subprocess
import concurrent.futures as cf
import os
import sys
import json
import networkx as nx
from scipy.sparse import load_npz
import scipy
import pickle
from pyflagsercontain import compute_cell_count

config_address = sys.argv[1]
with open(config_address, 'r') as f:
    config_dict = json.load(f)

# Load toolbox functions
exec(open('toolbox.py').read())
print("\nToolbox loaded")

# Config
number_nbhds = config_dict['values']['number_nbhds']                      # The number of top neighboorhoods to consider
timebin = config_dict['values']['timebin']                                # Size of timebins to split spiketrains into
train_set_perc = config_dict['values']['train_set_perc']                  # Size of train set vs test set
test_repetitions = config_dict['values']['test_repetitions']              # Number of times to repeat classification
spike_gid_shift = config_dict['values']['spike_gid_shift']
start_time = config_dict['values']['start_time']
bin_number = config_dict['values']['bin_number']

feature_parameter = config_dict['values']['feature_parameter']            # Possible parameters are any column header in df dataframe, without _low, _high, _radius for spectral parameters
feature_gap = config_dict['values']['feature_gap']                        # Must be high or low or radius. Relevant only for spectral parameters.

selection_parameters = config_dict['values']['selection_parameters']      # Possible parameters are any column header in df dataframe
selection_order = config_dict['values']['selection_order']                # Must be top or bottom
selection_ascending = (selection_order=='bottom')                         # Will be True or False

# Optional config
try: # Random3
    new_tribes_for_chiefs = config_dict['random_tests']['new_tribes_for_chiefs']
    if new_tribes_for_chiefs:
        print('Random initiated: Giving chiefs new tribes by choosing random vertices', flush=True)
except:
    new_tribes_for_chiefs = False
try: # Random4
    permute_activity_data = config_dict['random_tests']['permute_activity_data']
    if permute_activity_data:
        print('Random initiated: permuting activity data', flush=True)
except:
    permute_activity_data = False
try: # Random5
    new_tribes_for_chiefs_var = config_dict['random_tests']['new_tribes_for_chiefs_var']
    if new_tribes_for_chiefs_var:
        print('Random initiated: Giving chiefs new tribes by moving around chiefs edges', flush=True)
except:
    new_tribes_for_chiefs_var = False


# Addresses
spike_trains_address = config_dict['paths']['spike_trains_address']
savefolder = config_dict['paths']['savefolder']

# Load spiketrains
spiketrains = np.load(spike_trains_address,allow_pickle=True,encoding='latin1')

feature_failure = 0

def computeFeature(nhbds_for_featurization, contains_chief, feature_tracker, matrix):
    feature_vector = []
    for i in range(len(nhbds_for_featurization)):
        feature_tracker[1] += 1
        try:
            if feature_parameter == "ec":
                x = euler_characteristic(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])])
            elif feature_parameter == "tribe_size":
                x = len(nhbds_for_featurization[i])
            elif feature_parameter == "deg":
                x = degree_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])])
            elif feature_parameter == "in_deg":
                x = in_degree_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])])
            elif feature_parameter == "out_deg":
                x = out_degree_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])])
            elif feature_parameter == "rc":
                x = reciprocal_connections_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])])
            elif feature_parameter == "rc_chief":
                x = reciprocal_connections_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])], chief_only=True)
            elif feature_parameter == "tcc":
                x = tcc_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])]) if contains_chief[i] else 0
            elif feature_parameter == "ccc":
                x = ccc_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])])
            elif feature_parameter == "asg":
                x = asg_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])])
            elif feature_parameter == "tpsg":
                x = tpsg_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])], gap=feature_gap)
            elif feature_parameter == "tpsg_reversed":
                x = tpsg_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])], in_deg=True, gap=feature_gap)
            elif feature_parameter == "clsg":
                x = clsg_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])], gap=feature_gap)
            elif feature_parameter == "blsg":
                x = blsg_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])], gap=feature_gap)
            elif feature_parameter == "blsg_reversed":
                x = blsg_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])], reverse_flow=True, gap=feature_gap)
            elif feature_parameter == "nbc":
                x = nbc_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])])
            elif feature_parameter[:2] == "dc":
                x = dc_adjacency(matrix[np.ix_(nhbds_for_featurization[i],nhbds_for_featurization[i])], coeff_index=int(feature_parameter[2])) if contains_chief[i] else 0
            elif feature_parameter == "binary":
                x = 1 if contains_chief[i] else 0
        except Exception as e:
            #print(e)
            x = 0
            feature_tracker[0] += 1
        feature_vector.append(x)
    return feature_vector


# Parameter computation
def ConstructFeatureVector(parameter, nhbds, feature_tracker, matrix):
    # This function returns feature vectors for all the spiketrains in all classes, along with the class labels for subsequent ML
    feature_vectors = []
    all_zero = True

    # Random4 validation: Make a permutation vector if necessary
    if permute_activity_data:
        p = np.random.permutation(len(matrix))

    # nhbds_for_featurization contains active subnhbds in each time bin: first n positions are the n active subnhbds for the first timebin,
    # in descending order by the structural parameter. Then follows the n active subnhbds for the second timebin etc.
    for activation_class in range(len(spiketrains)):
        experiment_ID = 0

        for experiment in spiketrains[activation_class]:
            nhbds_for_featurization = []
            contains_chief = []

            for t in range(bin_number):
                spikers = (experiment[(experiment[:,0] > t*timebin+start_time) & (experiment[:,0] <= (t+1)*timebin+start_time)][:,1]-spike_gid_shift).astype(dtype=int)

                # Random4 validation: Permute activity data
                if permute_activity_data:
                    spikers = tuple(map(lambda x: p[x], spikers))

                for nhbd in nhbds:
                    #nbhds_for_featurization.append(np.intersect1d(nhbd[0],spikers))
                    nhbds_for_featurization.append(np.intersect1d(nhbd,spikers))
                    contains_chief.append(nhbd[0] in spikers)

            # Get Euler characteristic of the active subnhbds
            feature_vector = computeFeature(nhbds_for_featurization, contains_chief, feature_tracker, matrix)
            if np.any(feature_vector):
                all_zero = False
            feature_vector.append(activation_class)
            feature_vectors.append(feature_vector)
            experiment_ID += 1

    #assert not all_zero, "ERROR: all the features are zero"
    if all_zero:
        print('All the features are zero', flush=True)

    return np.array(feature_vectors)


# Main function creating the feature vectors according to implemented method
def featurization(parameter, matrix):
    feature_tracker = [0,0]
    current_matrix = np.copy(matrix)
    nbhds = top_nbhds(parameter, number=number_nbhds, order_by_ascending=selection_ascending, matrix=current_matrix)

    # Random3 validation
    if new_tribes_for_chiefs:
        nbhds = new_nbhds(nbhds, len(matrix))

    # Random5 validation
    elif new_tribes_for_chiefs_var:
        current_chiefs = top_chiefs(parameter, number=number_nbhds, order_by_ascending=selection_ascending)
        current_matrix = permute_list_but_all(matrix,current_chiefs)
        nbhds = top_nbhds(parameter, number=number_nbhds, order_by_ascending=selection_ascending, matrix=current_matrix)

    features_and_labels = ConstructFeatureVector(parameter, nbhds, feature_tracker, current_matrix)
    np.save(savefolder + parameter + '_feature_vectors.npy',features_and_labels)
    print('Finished '+ parameter)
    print('Feature computation failed for '+str(feature_tracker[0])+' out of '+str(feature_tracker[1]))

# Does the classification, requires the featurisation to be done first separately
def classify():
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    clf_SVM = SVC(cache_size = 500)

    output = open(savefolder + 'classification_accuracies_'+feature_parameter+'.txt','w')

    for parameter in selection_parameters:
        features = np.load(savefolder + parameter + '_feature_vectors.npy',allow_pickle = True)

        # X -> features, y -> label
        X = StandardScaler().fit_transform(features[:,:-1])
        y = features[:,-1]

        # dividing X, y into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_set_perc, random_state = 0)

        # doing everything cross-validated
        output.write(parameter + '\n')
        clf = clf_SVM
        cv_scores = cross_val_score(clf, X, y, cv=4)
        test_scores = []

        for _ in range(test_repetitions):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_set_perc)
            test_scores.append(clf.fit(X_train,y_train).score(X_test,y_test))

        # Use this if you want to save the fitted model for later use, otherwise comment
        #model_file = open('nest_test/fitted_SVM.pkl', 'wb')
        #pickle.dump(clf, model_file)
        #model_file.close()

        output.write('CV accuracy: %0.2f +/- %0.2f,  test accuracy: %0.2f +/- %0.2f \n' % (cv_scores.mean(),cv_scores.std()*2,np.array(test_scores).mean(),np.array(test_scores).std()*2))
        print('Finished '+parameter+' classification')

        output.write('\n')

    output.close()

if __name__ == '__main__':
    for i in selection_parameters:
        feature_vector_exists = subprocess.Popen(['ls',savefolder + i + '_feature_vectors.npy'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output = feature_vector_exists.communicate()[0].decode('utf-8')
        if output == '':
            featurization(i, matrix=adj)
        else:
            print('Feature vectors for parameter '+i+' already exist, skipping',flush=True)

    print("Featurisation Complete")
    print("Starting Classification")
    classify()
    print("Classification Complete")
