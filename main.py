# import library
import re
import sys
import json
import random
import numpy as np
from math import log
from collections import Counter
from collections import defaultdict
import argparse

# add argument to specify training & output path
parser = argparse.ArgumentParser(  
        description='sum the integers at the command line'
    )  
parser.add_argument(  
    'training_path', 
    type=str, 
    default="dataset/assignment1-data",
    help='training path folder'
)
parser.add_argument(  
    'output_path', 
    type=str, 
    default="output",
    help='training path folder',
)

def preprocess_line(line):
    # function to preprocess lines
    # add <bos> and <eos> in end of Line
    
    line = re.sub('[^A-Za-z0-9 .#]+', '', line)
    line = re.sub('[0-9]', '0', line).lower()
    return ("##"+line+"##")


def generate_trigrams_model(infile, outfile_tr, outfile_bg):
    # function to generate trigram model, 
    # calculate and save trigram with respect to its bigram
    
    tri_counts=defaultdict(int) #counts of all trigrams in input
    big_counts=defaultdict(int) #counts of all bigrams in input
    with open(infile) as f:
        for line in f:
            line = preprocess_line(line)
            for j in range(len(line)-(2)): 
                trigram = line[j:j+3]
                bigram = line[j:j+2]
                tri_counts[trigram] += 1
                big_counts[bigram] += 1
            # add the last bigrams
            big_counts[line[-2:]] += 1
    
    # loop to calculate probability foreach trigram (given bigram)
    # count(num of trigram) / count(num of its bigram)
    for key, val in tri_counts.items():
        tmp = []
        tmp.append(val)
        tmp.append(val / big_counts[key[:-1]])
        tri_counts[key] = tmp
        
    # also save probability model (trigram, bigram) to files, sorted alphabetically
    print("Trigram counts in ", infile, ", sorted alphabetically:")
    with open(outfile_tr, 'w') as f:
        for key in sorted(tri_counts.keys()):
            print(key, ": ", tri_counts[key])
            tmp = f'{key}\t{format(tri_counts[key][1], ".3e")}\t{tri_counts[key][0]}\n'
            f.writelines(tmp)
        f.close()
    print("Trigram counts in ", infile, ", sorted numerically:")
    for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
        print(tri_count[0], ": ", str(tri_count[1]))
    print("writing bigrams to file")
    with open(outfile_bg, "w") as f:
        for key in sorted(big_counts.keys()):
            print(key, ": ", big_counts[key])
            tmp = f'{key}\t{big_counts[key]}\n'
            f.writelines(tmp)
    

def load_model(model_path, type="pretrain"):
    # function to load model from file
    # differentiate between trigram, bigram, and pretrain files
    # different format foreach models
    
    try:
        if type == "trigram":
            with open(model_path, 'r') as f:
                rows = (line.strip('\n').split('\t') for line in f)
                model = {row[0]:[float(row[1]), float(row[2])] for row in rows}
                f.close()
        elif type == "bigram":
            with open(model_path, 'r') as f:
                rows = (line.strip('\n').split('\t') for line in f)
                model = {row[0]:float(row[1]) for row in rows}
                f.close()
        elif type == "pretrain":
            with open(model_path, 'r') as f:
                rows = (line.strip('\n').split('\t') for line in f)
                model = {row[0]:float(row[1]) for row in rows}
                f.close()
    except FileNotFoundError:
        return('needs to specify specific path, file not found in default path!')
    return model

    
def generate_from_LLM(char_length):
    # function to generate trigram randomly
    # to compare between pre-train and trained model
    
    # load pre-train model
    pretrain_model = load_model('dataset/assignment1-data/model-br.en', type="pretrain")
    
    # load trained model
    trained_model = load_model('output/model-tr.en', type="trigram")
    
    # char_length needs to be devided by 3 since the model is trigram (3-chars per keys)
    random_pretrain = random.sample(pretrain_model.keys(), char_length//3)
    random_trained = random.sample(trained_model.keys(), char_length//3)
    
    return random_pretrain, random_trained


def perplexity_test_sentences_skipped(test_file):
    # calculate perplexity on test sentences
    # but this time ignore unknown word
    
    # load trained model from 3 languages
    # trained en model
    trained_tr_model_en = load_model('output/model-tr.en', type="trigram")
    trained_bg_model_en = load_model('output/model-bg.en', type="bigram")
    # trained de model
    trained_tr_model_de = load_model('output/model-tr.de', type="trigram")
    trained_bg_model_de = load_model('output/model-bg.de', type="bigram")
    # trained es model
    trained_tr_model_es = load_model('output/model-tr.es', type="trigram")
    trained_bg_model_es = load_model('output/model-bg.es', type="bigram")
    
    # load test file, process into trigram
    try:
        sentences = []
        with open(test_file, 'r') as f:
            for line in f:
                trigram = []
                line = preprocess_line(line)
                for j in range(len(line)-(2)):
                    trigram.append(line[j:j+3])
                sentences.append(trigram)
    except FileNotFoundError:
        print(f'File not found in the {test_file} path!')
    
    # looping to calculate perplexity from 3 models
    # if not found, skipped
    result_dict = dict()
    for it, trigrams in enumerate(sentences):
        en_list, de_list, es_list = [], [], []
        for trig in trigrams:
            try:
                en_list.append(trained_tr_model_en[trig][0])
            except KeyError:
                continue
            try:
                de_list.append(trained_tr_model_de[trig][0])
            except KeyError:
                continue
            try:
                es_list.append(trained_tr_model_es[trig][0])
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

def perplexity_test_sentences_smoothing(test_file, alpha=1.0):
    # calculate perplexity on test sentences
    # but this time calculate unknown/known word using smoothing
    # alpha = 1
    # N_trigram => count corpus keys (trigram)
    # N_bigram => count corpus keys (bigram)
    # V => length corpus keys (distinct) => trigram
    
    # load trained model from 3 languages
    # trained en model
    trained_tr_model_en = load_model('output/model-tr.en', type="trigram")
    trained_bg_model_en = load_model('output/model-bg.en', type="bigram")
    N_tr_en = float(np.sum([val[1] for key, val in trained_tr_model_en.items()]))
    N_bg_en = float(np.sum([val for key, val in trained_bg_model_en.items()]))
    V_tr_en = float(len(trained_tr_model_en.keys()))
    
    # trained de model
    trained_tr_model_de = load_model('output/model-tr.de', type="trigram")
    trained_bg_model_de = load_model('output/model-bg.de', type="bigram")
    N_tr_de = float(np.sum([val[1] for key, val in trained_tr_model_de.items()]))
    N_bg_de = float(np.sum([val for key, val in trained_bg_model_de.items()]))
    V_tr_de = float(len(trained_tr_model_de.keys()))
    
    # trained es model
    trained_tr_model_es = load_model('output/model-tr.es', type="trigram")
    trained_bg_model_es = load_model('output/model-bg.es', type="bigram")
    N_tr_es = float(np.sum([val[1] for key, val in trained_tr_model_es.items()]))
    N_bg_es = float(np.sum([val for key, val in trained_bg_model_es.items()]))
    V_tr_es = float(len(trained_tr_model_es.keys()))

    # load test file, process into trigram
    try:
        sentences = []
        with open(test_file, 'r') as f:
            for line in f:
                trigram = []
                line = preprocess_line(line)
                for j in range(len(line)-(2)):
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
                en_list.append(trained_tr_model_en[trig][0])
            except KeyError:
                en_list.append((N_tr_en + alpha) / (N_bg_en + alpha * V_tr_en))
            try:
                de_list.append(trained_tr_model_de[trig][0])
            except KeyError:
                de_list.append((N_tr_de + alpha) / (N_bg_de + alpha * V_tr_de))
            try:
                es_list.append(trained_tr_model_es[trig][0])
            except KeyError:
                es_list.append((N_tr_es + alpha) / (N_bg_es + alpha * V_tr_es))
                
        en_score = np.prod(en_list) ** (-1/len(en_list))
        de_score = np.prod(de_list) ** (-1/len(de_list))
        es_score = np.prod(es_list) ** (-1/len(es_list))
        
        result_dict[it] = {
            "en": en_score,
            "de": de_score,
            "es": es_score,
        } 
    
    return result_dict

def language_detect(perplexity):
    # function to decide how many number of sentences foreach language
    # iterate dictionary results from the perplexity calculation function
    res = []
    for key, dict2 in perplexity.items():
        # the lower, the better
        mn = 999999
        lang = ''
        for key, val in dict2.items():
            if val < mn:
                mn = val
                lang = key
        res.append(lang)
    return Counter(res)

# main function
if __name__ == 'main':
    # parse argument
    args = parser.parse_args()
    
    # looping all files:
    # train and generate trigram models foreach dataset
    for files in ['en', 'de', 'es']:
        # infile = f'dataset/assignment1-data/training.{files}'
        infile = f'{args.training_path}/training.{files}'
        outfile_tr = f'{args.output_path}/model-tr.{files}'
        outfile_bg = f'{args.output_path}/model-bg.{files}'
        print(f'processing {files} language..')
        generate_trigrams_model(infile=infile, outfile_tr=outfile_tr, outfile_bg=outfile_bg)
    
    # calculate perplexity on test set, smoothing and skip
    smoothing = perplexity_test_sentences_smoothing('dataset/assignment1-data/test')
    skipped = perplexity_test_sentences_skipped('dataset/assignment1-data/test')
    
    print("smoothing\n")
    print(language_detect(smoothing))
    print(smoothing)
    
    print("skipped\n")
    print(language_detect(skipped))
    print(skipped)
    
    # get random trigrams from pre-train and trained model 
    pretrain, train = generate_from_LLM(300)
    print("pretrain\n", pretrain)
    print("train\n", train)
