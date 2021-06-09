import json
import os
import pickle
import re

import torch
from tqdm import tqdm

def build_dictionaries(gqa_dir):
    
    cached_dictionaries = os.path.join(gqa_dir, 'GQA_built_dictionaries.pkl')
    if os.path.exists(cached_dictionaries):
        print('==> using cached dictionaries: {}'.format(cached_dictionaries))
        with open(cached_dictionaries, 'rb') as f:
            return pickle.load(f)
            
    quest_to_ix = {}
    answ_to_ix = {}
    json_train_filename = os.path.join(gqa_dir, 'train_balanced_questions.json')
    #load all words from all training data
    with open(json_train_filename, "r") as f:
        train_q = json.load(f)
        for q in tqdm(train_q):
            question = tokenize(train_q[q]['question'])
            answer = train_q[q]['answer']
            for word in question:
                if word not in quest_to_ix:
                    quest_to_ix[word] = len(quest_to_ix)+1 #one based indexing; zero is reserved for padding
            
            a = answer.lower()
            if a not in answ_to_ix:
                ix = len(answ_to_ix)+1
                answ_to_ix[a] = ix

        quest_to_ix['UNK'] = len(quest_to_ix)+1
        answ_to_ix['UNK'] = len(answ_to_ix)+1

    names = {}
    attributes = {}
    train_sg_filename = os.path.join(gqa_dir, 'train_sceneGraphs.json')
    with open(train_sg_filename, 'r') as f:
        train_sg = json.load(f)
        for sg in train_sg:
            objects = train_sg[sg]['objects']
            for obj in objects:
                name = train_sg[sg]['objects'][obj]['name']
                if name not in names:
                    names[name] = len(names)
                obj_attributes = train_sg[sg]['objects'][obj]['attributes']
                for attr in obj_attributes:
                    if attr not in attributes:
                        attributes[attr] = len(attributes)

    names['UNK'] = len(names)
    attributes['UNK'] = len(attributes)
    
    ret = (quest_to_ix, answ_to_ix, names, attributes)    
    with open(cached_dictionaries, 'wb') as f:
        pickle.dump(ret, f)

    return ret

def to_dictionary_indexes(dictionary, sentence, to_tokenize=False):
    """
    Outputs indexes of the dictionary corresponding to the words in the sequence.
    Case insensitive.
    """
    if to_tokenize:
        split = tokenize(sentence)
    else:
        # don't tokenize answers
        split = [sentence]
    idxs = torch.LongTensor([dictionary[w] if w in dictionary else dictionary['UNK'] for w in split])
    return idxs

def collate_samples(batch):
    """
    Used by DatasetLoader to merge together multiple samples into one mini-batch.
    """
    batch_size = len(batch)

    images = [d['image'] for d in batch]
    answers = [d['answer'] for d in batch]
    questions = [d['question'] for d in batch]

    # questions are not fixed length: they must be padded to the maximum length
    # in this batch, in order to be inserted in a tensor
    max_len = max(map(len, questions))

    padded_questions = torch.LongTensor(batch_size, max_len).zero_()
    for i, q in enumerate(questions):
        padded_questions[i, :len(q)] = q
        
    max_len = 46 # will need to change this 
    #even object matrices should be padded (they are variable length)
    padded_objects = torch.FloatTensor(batch_size, max_len, images[0].size()[1]).zero_()
    for i, o in enumerate(images):
        padded_objects[i, :o.size()[0], :] = o
    images = padded_objects
    
    collated_batch = dict(
        image=images,
        answer=torch.stack(answers),
        question=padded_questions
    )
    return collated_batch


def tokenize(sentence):
    # punctuation should be separated from the words
    s = re.sub('([.,;:!?()])', r' \1 ', sentence)
    s = re.sub('\s{2,}', ' ', s)

    # tokenize
    split = s.split()

    # normalize all words to lowercase
    lower = [w.lower() for w in split]
    return lower

def load_tensor_data(data_batch, cuda, invert_questions):
    qst = data_batch['question']
    if invert_questions:
        # invert question indexes in this batch
        qst_len = qst.size()[1]
        qst = qst.index_select(1, torch.arange(qst_len - 1, -1, -1).long())

    img = data_batch['image']
    label = data_batch['answer']
    if cuda:
        img, qst, label = img.cuda(), qst.cuda(), label.cuda()

    label = (label - 1).squeeze(1)
    return img, qst, label