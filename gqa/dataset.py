import os
import json
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import utils

class GQADataset(Dataset):
    def __init__(self, gqa_dir, train, dictionaries, load_scenes=False):
        self.gqa_dir = gqa_dir
        self.dictionaries = dictionaries

        if train:
            quest_json_filename = os.path.join(gqa_dir, 'train_balanced_questions.json')
            scene_json_filename = os.path.join(gqa_dir, 'train_sceneGraphs.json')
        else:
            quest_json_filename = os.path.join(gqa_dir, 'val_balanced_questions.json')
            scene_json_filename = os.path.join(gqa_dir, 'val_sceneGraphs.json')

        with open(quest_json_filename, 'r') as f:
            print('loading questions...')
            self.questions = json.load(f)
            self.qids = list(self.questions.keys())
        
        if load_scenes:
            print('==> using cached scenes')
        else:
            print('Saving scene graph data')
            names = self.dictionaries[2]
            attributes = self.dictionaries[3]

            with open(scene_json_filename, 'r') as json_file:
                train_sg = json.load(json_file)
                for sg in tqdm(train_sg):
                    sg_width = train_sg[sg]['width']
                    sg_height = train_sg[sg]['height']
                    objects = train_sg[sg]['objects']
                    scene_objects = []
                    for obj in objects:
                        obj_x = train_sg[sg]['objects'][obj]['x']
                        obj_y = train_sg[sg]['objects'][obj]['y']
                        obj_w = train_sg[sg]['objects'][obj]['w']
                        obj_h = train_sg[sg]['objects'][obj]['h']
                        obj_name = train_sg[sg]['objects'][obj]['name']
                        obj_attributes = train_sg[sg]['objects'][obj]['attributes']

                        obj_vector = np.zeros(23956, dtype=np.float32)
                        obj_vector[0] = obj_x / sg_width
                        obj_vector[1] = obj_y / sg_height
                        obj_vector[2] = obj_w / sg_width
                        obj_vector[3] = obj_h / sg_height
                        if obj_name in names:
                            obj_vector[4+names[obj_name]] = 1
                        else:
                            obj_vector[4+names['UNK']] = 1

                        attr_start_idx = 1708
                        for attr in obj_attributes:
                            if attr in attributes:
                                obj_vector[attr_start_idx + attributes[attr]] = 1
                            else:
                                obj_vector[attr_start_idx + attributes['UNK']] = 1
                            attr_start_idx += 618
                        
                        scene_objects.append(obj_vector)
                    np.save('.\\data\\scene_graphs\\'+sg+'.npy', np.array(scene_objects, dtype=np.float32))


            # with open(scene_json_filename, 'r') as json_file:
            #     scenes = json.load(json_file)['scenes']
            #     print('caching all objects in all scenes...')
            #     for s in scenes:
            #         objects = s['objects']
            #         objects_attr = []
            #         for obj in objects:
            #             attr_values = []
            #             for attr in sorted(obj):
            #                 # convert object attributes in indexes
            #                 if attr in utils.classes:
            #                     attr_values.append(utils.classes[attr].index(obj[attr])+1)  #zero is reserved for padding
            #                 else:
            #                     '''if attr=='rotation':
            #                         attr_values.append(float(obj[attr]) / 360)'''
            #                     if attr=='3d_coords':
            #                         attr_values.extend(obj[attr])
            #             objects_attr.append(attr_values)
            #         all_scene_objs.append(torch.FloatTensor(objects_attr))
            #     self.objects = all_scene_objs
            

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        current_question = self.questions[self.qids[idx]]
        scene_idx = current_question['imageId']
        obj = torch.from_numpy(np.load('.\\data\\scene_graphs\\'+scene_idx+'.npy'))
        
        question = utils.to_dictionary_indexes(self.dictionaries[0], current_question['question'], True)
        answer = utils.to_dictionary_indexes(self.dictionaries[1], current_question['answer'], False)
        sample = {'image': obj, 'question': question, 'answer': answer}
        
        return sample