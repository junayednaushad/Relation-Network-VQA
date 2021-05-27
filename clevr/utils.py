import json
import os
import pickle
import re

import torch
from tqdm import tqdm

classes = {
            'number':['0','1','2','3','4','5','6','7','8','9','10'],
            'material':['rubber','metal'],
            'color':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape':['sphere','cube','cylinder'],
            'size':['large','small'],
            'exist':['yes','no']
        }
def build_dictionaries(clevr_dir):

    def compute_class(answer):
        for name,values in classes.items():
            if answer in values:
                return name
        
        raise ValueError('Answer {} does not belong to a known class'.format(answer))
        
        
    cached_dictionaries = os.path.join(clevr_dir, 'questions', 'CLEVR_built_dictionaries.pkl')
    if os.path.exists(cached_dictionaries):
        print('==> using cached dictionaries: {}'.format(cached_dictionaries))
        with open(cached_dictionaries, 'rb') as f:
            return pickle.load(f)
            
    quest_to_ix = {}
    answ_to_ix = {}
    answ_ix_to_class = {}
    json_train_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_trainA_questions.json')
    #load all words from all training data
    with open(json_train_filename, "r") as f:
        questions = json.load(f)['questions']
        for q in tqdm(questions):
            question = tokenize(q['question'])
            answer = q['answer']
            #pdb.set_trace()
            for word in question:
                if word not in quest_to_ix:
                    quest_to_ix[word] = len(quest_to_ix)+1 #one based indexing; zero is reserved for padding
            
            a = answer.lower()
            if a not in answ_to_ix:
                    ix = len(answ_to_ix)+1
                    answ_to_ix[a] = ix
                    answ_ix_to_class[ix] = compute_class(a)

    ret = (quest_to_ix, answ_to_ix, answ_ix_to_class)    
    with open(cached_dictionaries, 'wb') as f:
        pickle.dump(ret, f)

    return ret

def to_dictionary_indexes(dictionary, sentence):
    """
    Outputs indexes of the dictionary corresponding to the words in the sequence.
    Case insensitive.
    """
    split = tokenize(sentence)
    idxs = torch.LongTensor([dictionary[w] for w in split])
    return idxs

def tokenize(sentence):
    # punctuation should be separated from the words
    s = re.sub('([.,;:!?()])', r' \1 ', sentence)
    s = re.sub('\s{2,}', ' ', s)

    # tokenize
    split = s.split()

    # normalize all words to lowercase
    lower = [w.lower() for w in split]
    return lower