import random
import string
import math
import time
import nltk
from nltk.corpus import names
from nltk.classify import apply_features
import numpy as np
import pickle
import csv
import argparse

start = time.time()
from service.Gender_Prediction.data_util import PROJECT_DIR, str2bool, clean_str, time_since, split_dataset, _gender_features, load_dataset, DATASET_FN #use this for end to end testing on browser
#from data_util import PROJECT_DIR, str2bool, clean_str, time_since, split_dataset, _gender_features, load_dataset, DATASET_FN #use this for unit test with askai_recr as path in terminal

parser = argparse.ArgumentParser(description='Name2Gender Naive Bayes Classification')
parser.add_argument('names', metavar='N', type=str, nargs='*', help="Any number of names to be classified")
parser.add_argument('--weights', default="nb/naive_bayes_weights", help='File to save weights to within weights subdir')
parser.add_argument('--infile', default=None, type=str, help='Plaintext file to read names from')
parser.add_argument('--outfile', default=None, type=str, help='CSV file to store name classifications in')
parser.add_argument('--verbose', default=True, type=str2bool, help="Set to False to prevent printing each classification")
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")

args = parser.parse_args()

args.weights = PROJECT_DIR+"/Data/Gender_Prediction_data/weights/"+args.weights

TRAINSET, VALSET, TESTSET = split_dataset()

def load_classifier(weight_file=args.weights, verbose=False):
    with open(weight_file, 'rb') as f:
        classifier = pickle.load(f)
        f.close()
    if verbose: print('Loaded weights from "%s"...\n' % (weight_file))
    return classifier

def predict_gender(name, classifier, verbose=args.verbose):
    _name = _gender_features(clean_str(name))
    dist = classifier.prob_classify(_name)
    m, f = dist.prob("male"), dist.prob("female")
    d = {m: "male", f: "female"}
    prob = max(m,f)
    guess = d[prob]
    #if verbose: print("%s -> %s (%.2f%%)" % (name, guess, prob * 100))
    
    #return guess, prob
    return guess

def classify_Gender(names, weight_file=args.weights):
    classifier = load_classifier(weight_file)
    dataset = load_dataset(DATASET_FN)
    for name in names:
         try:
            gender = [tup[1] for tup in dataset if name in tup]
            prediction = gender[0]
         except:
            prediction = predict_gender(name, classifier)
    return prediction


def write_classifications(names, weight_file=args.weights, outfile=args.outfile):
    classifier = load_classifier(weight_file)
    headers = ["name", "gender", "probability"]
    with open(outfile, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(headers)
        for name in names:
            guess, prob = predict_gender(name, classifier)
            writer.writerow([name, guess, prob])
        f.close()
    print('\nWrote %d names to "%s" (%s)' % (len(names), outfile, time_since(start)))

def read_names(filename=args.infile):
    with open(filename, 'r') as f:
        names = [clean_str(line.rstrip('\n')) for line in f]
    print("Loaded %d names from %s" % (len(names), filename))
    return names
        
def get_errors(names, weight_file=args.weights):
    errors = []
    full_list_including_errors = []
    classifier = load_classifier(weight_file)
    for (name,tag) in names:
        guess,prob = predict_gender(name,classifier)
        if guess != tag:
            errors.append([tag,guess,name])
        else:
            full_list_including_errors.append([tag,guess,name])
    #print (errors)
    csvfile = PROJECT_DIR+"/data/wrongly_guessed1.csv"
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(errors)
    csvfile1 = PROJECT_DIR+"/data/correct_to_guessed_full1.csv"
    with open(csvfile1, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(full_list_including_errors)

import pandas as pd
def prediction(file_name):
    dataset = pd.read_csv(file_name)
    data = dataset.values.tolist()
    
    y_out = []
    for i in data:
        y_out.append(classify_Gender([i]))
    
    res = [['Serial Number','Name','Gender']]
    for i in range(len(data)):
        res.append((data[i][0],data[i][1],y_out[i]))
    return res
#if __name__ == '__main__':

    #names = read_names(filename=PROJECT_DIR+'data/test_nb.csv') #if args.infile is not None else args.names
    #if args.outfile is not None:
        #print('print to file')
    #write_classifications(names,outfile=PROJECT_DIR+'/data/out_nb.csv')
        #write_classifications(names)
    #else:
        # print to screen only
    #classify(TESTSET)
    #print(classify_Gender(['Abigail','Ahmad','Jane','John','Muttusamy','Parag','Paras','Param','Kiran','Lakshmi','Arpitha','Arpit']))
    #get_errors(TESTSET)
print (classify_Gender(['dhanashree ']))
#classify_Gender(['ravindra'])
