import argparse
import json
import os
import pickle
import re
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import utils
import math
from dataset import ClevrDataset
from model import RN

def train(data, model, criterion, optimizer, epoch, args):
    model.train()

    avg_loss = 0.0
    n_batches = 0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        img, qst, label = utils.load_tensor_data(sample_batched, args.cuda, args.invert_questions)

        # forward and backward pass
        optimizer.zero_grad()
        output = model(img, qst)
        loss = criterion(output, label)
        loss.backward()

        # Gradient Clipping
        if args.clip_norm:
            clip_grad_norm_(model.parameters(), args.clip_norm)

        optimizer.step()

        # Show progress
        progress_bar.set_postfix(dict(loss=loss.item()))
        avg_loss += loss.item()
        n_batches += 1

        if batch_idx % args.log_interval == 0:
            avg_loss /= n_batches
            processed = batch_idx * args.batch_size
            n_samples = len(data) * args.batch_size
            progress = float(processed) / n_samples
            print('Train Epoch: {} [{}/{} ({:.0%})] Train loss: {}'.format(
                epoch, processed, n_samples, progress, avg_loss))
            avg_loss = 0.0
            n_batches = 0

def test(data, model, criterion, epoch, dictionaries, args):
    model.eval()

    # accuracy for every class
    class_corrects = {}
    # for every class, among all the wrong answers, how much are non pertinent
    class_invalids = {}
    # total number of samples for every class
    class_n_samples = {}
    # initialization
    for c in dictionaries[2].values():
        class_corrects[c] = 0
        class_invalids[c] = 0
        class_n_samples[c] = 0

    corrects = 0.0
    invalids = 0.0
    n_samples = 0

    inverted_answ_dict = {v: k for k,v in dictionaries[1].items()}
    sorted_classes = sorted(dictionaries[2].items(), key=lambda x: hash(x[1]) if x[1]!='number' else int(inverted_answ_dict[x[0]]))
    sorted_classes = [c[0]-1 for c in sorted_classes]

    confusion_matrix_target = []
    confusion_matrix_pred = []

    sorted_labels = sorted(dictionaries[1].items(), key=lambda x: x[1])
    sorted_labels = [c[0] for c in sorted_labels]
    sorted_labels = [sorted_labels[c] for c in sorted_classes]

    avg_loss = 0.0
    progress_bar = tqdm(data)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(progress_bar):
            img, qst, label = utils.load_tensor_data(sample_batched, args.cuda, args.invert_questions)
            
            output = model(img, qst)
            pred = output.data.max(1)[1]

            loss = criterion(output, label)

            # compute per-class accuracy
            pred_class = [dictionaries[2][int(o+1)] for o in pred]
            real_class = [dictionaries[2][int(o+1)] for o in label.data]
            for idx,rc in enumerate(real_class):
                class_corrects[rc] += (pred[idx] == label.data[idx])
                class_n_samples[rc] += 1

            for pc, rc in zip(pred_class,real_class):
                class_invalids[rc] += (pc != rc)

            for p,l in zip(pred, label.data):
                confusion_matrix_target.append(sorted_classes.index(l))
                confusion_matrix_pred.append(sorted_classes.index(p))
            
            # compute global accuracy
            corrects += (pred == label.data).sum()
            assert corrects == sum(class_corrects.values()), 'Number of correct answers assertion error!'
            invalids = sum(class_invalids.values())
            n_samples += len(label)
            assert n_samples == sum(class_n_samples.values()), 'Number of total answers assertion error!'
            
            avg_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                accuracy = corrects / n_samples
                invalids_perc = invalids / n_samples
                progress_bar.set_postfix(dict(acc='{:.2%}'.format(accuracy), inv='{:.2%}'.format(invalids_perc)))
    
    avg_loss /= len(data)
    invalids_perc = invalids / n_samples      
    accuracy = corrects / n_samples

    print('Test Epoch {}: Accuracy = {:.2%} ({:g}/{}); Invalids = {:.2%} ({:g}/{}); Test loss = {}'.format(epoch, accuracy, corrects, n_samples, invalids_perc, invalids, n_samples, avg_loss))
    for v in class_n_samples.keys():
        accuracy = 0
        invalid = 0
        if class_n_samples[v] != 0:
            accuracy = class_corrects[v] / class_n_samples[v]
            invalid = class_invalids[v] / class_n_samples[v]
        print('{} -- acc: {:.2%} ({}/{}); invalid: {:.2%} ({}/{})'.format(v,accuracy,class_corrects[v],class_n_samples[v],invalid,class_invalids[v],class_n_samples[v]))

    # dump results on file
    filename = os.path.join(args.test_results_dir, 'test.pickle')
    dump_object = {
        'class_corrects':class_corrects,
        'class_invalids':class_invalids,
        'class_total_samples':class_n_samples,
        'confusion_matrix_target':confusion_matrix_target,
        'confusion_matrix_pred':confusion_matrix_pred,
        'confusion_matrix_labels':sorted_labels,
        'global_accuracy':accuracy
    }
    pickle.dump(dump_object, open(filename,'wb'))
    return accuracy

def main(args):
    #load hyperparameters from configuration file
    with open(args.config) as config_file: 
        hyp = json.load(config_file)['hyperparams'][args.model]
    #override configuration dropout
    if args.dropout > 0:
        hyp['dropout'] = args.dropout
    if args.question_injection >= 0:
        hyp['question_injection_position'] = args.question_injection

    print('Loaded hyperparameters from configuration {}, model: {}: {}'.format(args.config, args.model, hyp))
    args.model_dirs = './models'
    # args.model_dirs = './model_{}_drop{}_bstart{}_bstep{}_bgamma{}_bmax{}_lrstart{}_'+ \
    #                   'lrstep{}_lrgamma{}_lrmax{}_invquests-{}_clipnorm{}_glayers{}_qinj{}_fc1{}_fc2{}'
    # args.model_dirs = args.model_dirs.format(
    #                     args.model, hyp['dropout'], args.batch_size, args.bs_step, args.bs_gamma, 
    #                     args.bs_max, args.lr, args.lr_step, args.lr_gamma, args.lr_max,
    #                     args.invert_questions, args.clip_norm, hyp['g_layers'], hyp['question_injection_position'],
    #                     hyp['f_fc1'], hyp['f_fc2'])
    if not os.path.exists(args.model_dirs):
        os.makedirs(args.model_dirs)
    #create a file in this folder containing the overall configuration
    args_str = str(args)
    hyp_str = str(hyp)
    all_configuration = args_str+'\n\n'+hyp_str
    filename = os.path.join(args.model_dirs,'config.txt')
    with open(filename,'w') as config_file:
        config_file.write(all_configuration)

    args.features_dirs = './features'
    args.test_results_dir = './test_results'
    if not os.path.exists(args.test_results_dir):
        os.makedirs(args.test_results_dir)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('Building word dictionaries from all the words in the dataset...')
    dictionaries = utils.build_dictionaries(args.clevr_dir)
    print('Word dictionary completed!')

    print('Initializing CLEVR dataset...')
    clevr_dataset_train = ClevrDataset(args.clevr_dir, True, dictionaries, args.cogen)
    clevr_dataset_test = ClevrDataset(args.clevr_dir, False, dictionaries, args.cogen)
    print('CLEVR dataset initialized!')

    # Build the model
    args.qdict_size = len(dictionaries[0])
    args.adict_size = len(dictionaries[1])

    model = RN(args, hyp)

    if torch.cuda.device_count() > 1 and args.cuda:
        model = torch.nn.DataParallel(model)
        model.module.cuda()  # call cuda() overridden method

    if args.cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    if args.resume:
        filename = args.resume
        if os.path.isfile(filename):
            print('==> loading checkpoint {}'.format(filename))
            checkpoint = torch.load(filename)

            #removes 'module' from dict entries, pytorch bug #3805
            if torch.cuda.device_count() == 1 and any(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}
            if torch.cuda.device_count() > 1 and not any(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {'module.'+k: v for k,v in checkpoint.items()}

            model.load_state_dict(checkpoint)
            print('==> loaded checkpoint {}'.format(filename))
            start_epoch = int(re.match(r'.*epoch_(\d+).pt', args.resume).groups()[0]) + 1


    progress_bar = trange(start_epoch, args.epochs + 1)
    if args.test:
        # perform a single test
        print('Testing epoch {}'.format(start_epoch))
        clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=args.test_batch_size,
                                    shuffle=False, collate_fn=utils.collate_samples)
        test(clevr_test_loader, model, criterion, start_epoch, dictionaries, args)
    else:
        bs = args.batch_size

        # perform a full training
        #TODO: find a better solution for general lr scheduling policies
        candidate_lr = args.lr * args.lr_gamma ** (start_epoch-1 // args.lr_step)
        lr = candidate_lr if candidate_lr <= args.lr_max else args.lr_max

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=1e-6, verbose=True)
        scheduler = lr_scheduler.StepLR(optimizer, args.lr_step, gamma=args.lr_gamma)
        scheduler.last_epoch = start_epoch
        print('Training ({} epochs) is starting...'.format(args.epochs))
        best_test_accuracy = 0
        for epoch in progress_bar:
            
            if(((args.bs_max > 0 and bs < args.bs_max) or args.bs_max < 0 ) and (epoch % args.bs_step == 0 or epoch == start_epoch)):
                bs = math.floor(args.batch_size * (args.bs_gamma ** (epoch // args.bs_step)))
                if bs > args.bs_max and args.bs_max > 0:
                    bs = args.bs_max
                clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=bs,
                                    shuffle=True, collate_fn=utils.collate_samples)
                clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=args.test_batch_size,
                                    shuffle=False, collate_fn=utils.collate_samples)

                #restart optimizer in order to restart learning rate scheduler
                #for param_group in optimizer.param_groups:
                #    param_group['lr'] = args.lr
                #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, step, min_lr)
                print('Dataset reinitialized with batch size {}'.format(bs))
            
            # if((args.lr_max > 0 and scheduler.get_lr()[0]<args.lr_max) or args.lr_max < 0):
            #     scheduler.step()
                    
            print('Current learning rate: {}'.format(optimizer.param_groups[0]['lr']))
                
            # TRAIN
            progress_bar.set_description('TRAIN')
            train(clevr_train_loader, model, criterion, optimizer, epoch, args)

            if((args.lr_max > 0 and scheduler.get_last_lr()[0]<args.lr_max) or args.lr_max < 0):
                scheduler.step()

            # TEST
            if epoch % 10 == 1:
                progress_bar.set_description('TEST')
                acc = test(clevr_test_loader, model, criterion, epoch, dictionaries, args)

                # SAVE MODEL
                if acc > best_test_accuracy:
                    best_test_accuracy = acc
                    print('Better accuracy found at epoch: {}'.format(epoch))
                    print('Saving model')
                    filename = '/RN_epoch_{:02d}.pt'.format(epoch)
                    torch.save(model.state_dict(), args.model_dirs + filename)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR')
    parser.add_argument('--cogen', type=bool, default=False,
                        help='use different data for validation')
    parser.add_argument('--batch-size', type=int, default=640, metavar='N',
                        help='input batch size for training (default: 640)')
    parser.add_argument('--test-batch-size', type=int, default=640,
                        help='input batch size for training (default: 640)')
    parser.add_argument('--epochs', type=int, default=151, metavar='N',
                        help='number of epochs to train (default: 151)')
    parser.add_argument('--lr', type=float, default=0.000005, metavar='LR',
                        help='learning rate (default: 0.000005)')
    parser.add_argument('--clip-norm', type=int, default=50,
                        help='max norm for gradients; set to 0 to disable gradient clipping (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str,
                        help='resume from model stored')
    parser.add_argument('--clevr-dir', type=str, default='.',
                        help='base directory of CLEVR dataset')
    parser.add_argument('--model', type=str, default='original-sd',
                        help='which model is used to train the network')
    parser.add_argument('--no-invert-questions', action='store_true', default=False,
                        help='invert the question word indexes for LSTM processing')
    parser.add_argument('--test', action='store_true', default=False,
                        help='perform only a single test. To use with --resume')
    parser.add_argument('--conv-transfer-learn', type=str,
                    help='use convolutional layer from another training')
    parser.add_argument('--lr-max', type=float, default=0.0005,
                        help='max learning rate')
    parser.add_argument('--lr-gamma', type=float, default=2, 
                        help='increasing rate for the learning rate. 1 to keep LR constant.')
    parser.add_argument('--lr-step', type=int, default=20,
                        help='number of epochs before lr update')
    parser.add_argument('--bs-max', type=int, default=-1,
                        help='max batch-size')
    parser.add_argument('--bs-gamma', type=float, default=1, 
                        help='increasing rate for the batch size. 1 to keep batch-size constant.')
    parser.add_argument('--bs-step', type=int, default=20, 
                        help='number of epochs before batch-size update')
    parser.add_argument('--dropout', type=float, default=-1,
                        help='dropout rate. -1 to use value from configuration')
    parser.add_argument('--config', type=str, default='./config.json',
                        help='configuration file for hyperparameters loading')
    parser.add_argument('--question-injection', type=int, default=-1, 
                        help='At which stage of g function the question should be inserted (0 to insert at the beginning, as specified in DeepMind model, -1 to use configuration value)')
    args = parser.parse_args()
    args.invert_questions = not args.no_invert_questions
    main(args)

    # Train: python train.py --clevr-dir D:\\272_project\\data\\CLEVR | tee logfile.log
    # Resume Training: python train.py --clevr-dir D:\\272_project\\data\\CLEVR --resume model.pt | tee logfile.log
    # Test: python train.py --clevr-dir D:\\272_project\\data\\CLEVR --resume model.pt --test 
    # Plot: python plot.py -i -trl -tsl -a logfile.log
