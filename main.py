#Here are some libraries you're likely to use. You might want/need others as well.
#%%
import re
import sys
import json
import random
import numpy as np
from math import log
from collections import defaultdict
import argparse

#%%
# parser = argparse.ArgumentParser(  
#         description='sum the integers at the command line'
#     )  
# parser.add_argument(  
#     'training_file', 
#     type='str,', 
#     required=True,
#     help='input file path')  
# parser.add_argument(  
#     '--log', default=sys.stdout, type=argparse.FileType('w'),  
#     help='the file where the sum should be written')  
# args = parser.parse_args()

#%%
#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
# if len(sys.argv) != 2:
#     print("Usage: ", sys.argv[0], "<training_file>")
#     sys.exit(1)

# infile = sys.argv[1] #get input argument: the training file

#%%
# preprocessing lines
def preprocess_line(line):
    return re.sub('[^A-Za-z0-9 .]+', '', line)

#%%
#This bit of code gives an example of how you might extract trigram counts
#from a file, line by line. If you plan to use or modify this code,
#please ensure you understand what it is actually doing, especially at the
#beginning and end of each line. Depending on how you write the rest of
#your program, you may need to modify this code.

def generate_trigrams_model(infile, outfile):
    tri_counts=defaultdict(int) #counts of all trigrams in input
    with open(infile) as f:
        for line in f:
            line = preprocess_line(line) #doesn't do anything yet.
            for j in range(len(line)-(3)):
                trigram = line[j:j+3]
                tri_counts[trigram] += 1

    length_dict = len(tri_counts.keys())
    for key, val in tri_counts.items():
        tmp = []
        tmp.append(val)
        tmp.append(val / length_dict)
        tri_counts[key] = tmp
        
    #Some example code that prints out the counts. For small input files
    #the counts are easy to look at but for larger files you can redirect
    #to an output file (see Lab 1).
    # also save probability model to files, sorted alphabetically
    print("Trigram counts in ", infile, ", sorted alphabetically:")
    with open(outfile, 'w') as f:
        for key in sorted(tri_counts.keys()):
            print(key, ": ", tri_counts[key])
            tmp = f'{key}\t{format(tri_counts[key][1], ".3e")}\t{tri_counts[key][0]}\n'
            f.writelines(tmp)
    print("Trigram counts in ", infile, ", sorted numerically:")
    for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
        print(tri_count[0], ": ", str(tri_count[1]))

#%%
def load_model(model_path, pretrain=False):
    try:
        if not pretrain:
            with open(model_path, 'r') as f:
                rows = (line.strip('\n').split('\t') for line in f)
                model = {row[0]:[float(row[1]), float(row[2])] for row in rows}
                f.close()
        else:
            with open(model_path, 'r') as f:
                rows = (line.strip('\n').split('\t') for line in f)
                model = {row[0]:float(row[1]) for row in rows}
                f.close()
    except FileNotFoundError:
        return('needs to specify specific path, file not found in default path!')
    return model

#%%    
def generate_from_LLM(char_length):
    # load pre-train model
    pretrain_model = load_model('dataset/assignment1-data/model-br.en', pretrain=True)
    
    # load trained model
    trained_model = load_model('output/model-tr.en')
    
    # char_length needs to be devided by 3 since the model is trigram (3-chars per keys)
    random_pretrain = random.sample(pretrain_model.keys(), char_length//3)
    random_trained = random.sample(trained_model.keys(), char_length//3)
    
    # return pretrain_model, trained_model
    return random_pretrain, random_trained

#%%
def perplexity_test_sentences_skipped(test_file):
    # load trained model from 3 languages
    # trained en model
    trained_model_en = load_model('output/model-tr.en')
    # trained de model
    trained_model_de = load_model('output/model-tr.de')
    # trained es model
    trained_model_es = load_model('output/model-tr.es')
    
    # load test file, process into trigram
    try:
        sentences = []
        with open(test_file, 'r') as f:
            for line in f:
                trigram = []
                line = preprocess_line(line)
                for j in range(len(line)-(3)):
                    trigram.append(line[j:j+3])
                sentences.append(trigram)
    except FileNotFoundError:
        print(f'File not found in the {test_file} path!')
    
    # looping to calculate perplexity from 3 models
    result_dict = dict()
    for it, trigrams in enumerate(sentences):
        en_list, de_list, es_list = [], [], []
        for trig in trigrams:
            try:
                en_list.append(trained_model_en[trig][0])
            except KeyError:
                continue
            try:
                de_list.append(trained_model_de[trig][0])
            except KeyError:
                continue
            try:
                es_list.append(trained_model_es[trig][0])
            except KeyError:
                continue
        en_score = np.prod(en_list) ** (-1/len(en_list))
        de_score = np.prod(de_list) ** (-1/len(de_list))
        es_score = np.prod(es_list) ** (-1/len(es_list))
        
        result_dict[it] = {
            "en": en_score,
            "de": de_score,
            "es": es_score,
        } 
    
    return result_dict
#%%
def perplexity_test_sentences_smoothing(test_file, alpha=1.0):
    # load trained model from 3 languages
    # alpha = 1
    # N => count corpus keys
    # V => length corpus keys (distinct)
    # trained en model
    trained_model_en = load_model('output/model-tr.en')
    N_en = float(np.sum([val[1] for key, val in trained_model_en.items()]))
    V_en = float(len(trained_model_en.keys()))
    # trained de model
    trained_model_de = load_model('output/model-tr.de')
    N_de = float(np.sum([val[1] for key, val in trained_model_de.items()]))
    V_de = float(len(trained_model_de.keys()))
    # trained es model
    trained_model_es = load_model('output/model-tr.es')
    N_es = float(np.sum([val[1] for key, val in trained_model_es.items()]))
    V_es = float(len(trained_model_en.keys()))

    # load test file, process into trigram
    try:
        sentences = []
        with open(test_file, 'r') as f:
            for line in f:
                trigram = []
                line = preprocess_line(line)
                for j in range(len(line)-(3)):
                    trigram.append(line[j:j+3])
                sentences.append(trigram)
    except FileNotFoundError:
        print(f'File not found in the {test_file} path!')
    
    # looping to calculate perplexity from 3 models
    result_dict = dict()
    for it, trigrams in enumerate(sentences):
        en_list, de_list, es_list = [], [], []
        for trig in trigrams:
            try:
                en_list.append((trained_model_en[trig][0] * N_en + alpha) / (N_en + alpha * V_en))
            except KeyError:
                en_list.append(1/N_en + 1 * V_en)
            try:
                de_list.append((trained_model_de[trig][0] * N_de + alpha) / (N_de + alpha * V_de))
            except KeyError:
                en_list.append(1/N_de + 1 * V_de)
            try:
                es_list.append((trained_model_es[trig][0] * N_es + alpha) / (N_es + alpha * V_es))
            except KeyError:
                es_list.append(1/N_es + 1 * V_es)
                
        en_score = np.prod(en_list) ** (-1/len(en_list))
        de_score = np.prod(de_list) ** (-1/len(de_list))
        es_score = np.prod(es_list) ** (-1/len(es_list))
        
        result_dict[it] = {
            "en": en_score,
            "de": de_score,
            "es": es_score,
        } 
    
    return result_dict

# %%
if __name__ == 'main':
    for files in ['en', 'de', 'es']:
        infile = f'dataset/assignment1-data/training.{files}'
        outfile = f'output/model-tr.{files}'
        print(f'processing {files} language..')
        generate_trigrams_model(infile=infile, outfile=outfile)
    # %%
    perplexity_test_sentences_smoothing('dataset/assignment1-data/test')
    # %%
    perplexity_test_sentences_skipped('dataset/assignment1-data/test')
    #%%
    pretrain, train = generate_from_LLM(300)
    #%%
    print("pretrain\n", pretrain)
    # %%
    print("train\n", train)
