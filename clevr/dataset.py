import os
import json
import torch
import pickle
from torch.utils.data import Dataset
import utils

class ClevrDataset(Dataset):
    def __init__(self, clevr_dir, train, dictionaries, cogen=False):
        self.clevr_dir = clevr_dir
        self.dictionaries = dictionaries

        if train:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_trainA_questions.json')
            scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_trainA_scenes.json')
        else:
            if cogen:
                quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_valB_questions.json')
                scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_valB_scenes.json')
            else:
                quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_valA_questions.json')
                scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_valA_scenes.json')

        with open(quest_json_filename, 'r') as f:
            print('loading questions...')
            self.questions = json.load(f)['questions']
        
        cached_scenes = scene_json_filename.replace('.json', '.pkl')
        if os.path.exists(cached_scenes):
            print('==> using cached scenes: {}'.format(cached_scenes))
            with open(cached_scenes, 'rb') as f:
                self.objects = pickle.load(f)
        else:
            all_scene_objs = []
            with open(scene_json_filename, 'r') as json_file:
                scenes = json.load(json_file)['scenes']
                print('caching all objects in all scenes...')
                for s in scenes:
                    objects = s['objects']
                    objects_attr = []
                    for obj in objects:
                        attr_values = []
                        for attr in sorted(obj):
                            # convert object attributes in indexes
                            if attr in utils.classes:
                                attr_values.append(utils.classes[attr].index(obj[attr])+1)  #zero is reserved for padding
                            else:
                                '''if attr=='rotation':
                                    attr_values.append(float(obj[attr]) / 360)'''
                                if attr=='3d_coords':
                                    attr_values.extend(obj[attr])
                        objects_attr.append(attr_values)
                    all_scene_objs.append(torch.FloatTensor(objects_attr))
                self.objects = all_scene_objs
            with open(cached_scenes, 'wb') as f:
                pickle.dump(all_scene_objs, f)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        scene_idx = current_question['image_index']
        obj = self.objects[scene_idx]
        
        question = utils.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
        answer = utils.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])
        sample = {'image': obj, 'question': question, 'answer': answer}
        
        return sample