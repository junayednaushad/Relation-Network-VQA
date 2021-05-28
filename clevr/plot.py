import re
import argparse
import os
import matplotlib
import numpy as np

'''def parse_log(log, pattern):
    with open(log, 'r') as log_file:
        for line in log_file:
            match = re.search(pattern, line)
            if match:
                # yield the first group of the pattern;
                # i.e. the one delimited in parenthesis
                # inside the pattern (...)
                yield match.group(1)'''

def parse_log(log, pattern):
    with open(log, 'r') as log_file:
        for i, line in enumerate(log_file):
            match = re.search(pattern, line)
            if match and '(0%)' not in line:
                # yield the first group of the pattern;
                # i.e. the one delimited in parenthesis
                # inside the pattern (...)
                yield i, match.group(1)

def plot_train_loss(args):
    losses = np.mean(np.array([float(i) for _,i in parse_log(args.log_file, r'Train loss: (.*)')]).reshape(-1,10), axis=1)
    epochs = np.arange(1,len(losses)+1)

    plt.clf()
    plt.plot(epochs, losses)
    plt.ylabel('Train Loss')
    plt.xlabel('Epoch')
    if args.y_max != 0:
        plt.ylim(ymax=args.y_max)
    if args.y_min != 0:
        plt.ylim(ymin=args.y_min)
    plt.grid(True)
    plt.savefig(os.path.join(args.img_dir, 'train_loss.png'))
    if not args.no_show:
        plt.show()

def plot_test_loss(args):
    losses = [float(i) for _,i in parse_log(args.log_file, r'Test loss = (.*)')]
    epochs = np.arange(0,len(losses)) * 10 + 1
    plt.clf()
    plt.plot(epochs, losses)
    plt.ylabel('Test Loss')
    if args.y_max != 0:
        plt.ylim(ymax=args.y_max)
    if args.y_min != 0:
        plt.ylim(ymin=args.y_min)
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(args.img_dir, 'test_loss.png'))
    if not args.no_show:
        plt.show()

def plot_train_test_loss(args):
    train_losses = np.mean(np.array([float(i) for _,i in parse_log(args.log_file, r'Train loss: (.*)')]).reshape(-1,10), axis=1)
    train_epochs = np.arange(1,len(train_losses)+1)
    test_losses = [float(i) for _,i in parse_log(args.log_file, r'Test loss = (.*)')]
    test_epochs = np.arange(0,len(test_losses)) * 10 + 1
    
    plt.clf()
    plt.plot(train_epochs, train_losses, label='train loss')
    plt.plot(test_epochs, test_losses, label='test loss')
    plt.legend()
    plt.ylabel('Loss')
    if args.y_max != 0:
        plt.ylim(ymax=args.y_max)
    if args.y_min != 0:
        plt.ylim(ymin=args.y_min)
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(args.img_dir, 'train_test_loss.png'))
    if not args.no_show:
        plt.show()

def plot_accuracy(args):
    accuracy = [float(i) for _,i in parse_log(args.log_file, r'.* Accuracy = (\d+\.\d+)%')]
    epochs = np.arange(0,len(accuracy)) * 10 + 1
    details = ['exist', 'number', 'material', 'size', 'shape', 'color']
    
    accs = {k: [float(i) for _,i in parse_log(args.log_file, '{} -- acc: (\d+\.\d+)%'.format(k))]
            for k in details}

    plt.clf()
    for k, v in accs.items():
        plt.plot(epochs, v, label=k)

    plt.plot(epochs, accuracy, linewidth=2, label='total')
    plt.ylim(0,100)
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.savefig(os.path.join(args.img_dir, 'accuracy.png'))
    if not args.no_show:
        plt.show()

def plot_invalids(args):
    invalids = [float(i) for _,i in parse_log(args.log_file, r'.* Invalids = (\d+\.\d+)%')]
    epochs = np.arange(0,len(invalids)) * 10 + 1
    '''details = ['exist', 'number', 'material', 'size', 'shape', 'color']
    
    invds = {k: [float(i) for i in parse_log(log, '.* invalid: (\d+\.\d+)%'.format(k))]
            for k in details}
    
    for k, v in invds.items():
        plt.plot(v, label=k)'''
    
    plt.clf()
    plt.plot(epochs, invalids, linewidth=2, label='total')
    plt.legend(loc='best')
    plt.title('Invalid rate')
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.grid(True)
    plt.savefig(os.path.join(args.img_dir, 'invalids.png'))
    if not args.no_show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot RN training logs')
    parser.add_argument('log_file', type=str, help='Log file to plot')
    parser.add_argument('-trl', '--train-loss', action='store_true', help='Show training loss plot')
    parser.add_argument('-tsl', '--test-loss', action='store_true', help='Show test loss plot')
    parser.add_argument('-a', '--accuracy', action='store_true', help='Show accuracy plot')
    parser.add_argument('-i', '--invalids', action='store_true', help='Show invalid rate plot')
    parser.add_argument('--no-show', action='store_true', help='Do not show figures, store only on file')
    parser.add_argument('--y-max', type=float, default=0,
                        help='upper bound for y axis of loss plots (0 to leave default)')
    parser.add_argument('--y-min', type=float, default=0,
                        help='lower bound for y axis of loss plots (0 to leave default)')
    parser.add_argument('--img_dir', type=str, default='imgs',
                        help='Directroy where plots will be saved')
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use('Agg')    
    import matplotlib.pyplot as plt
    plt.style.use('bmh')

    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    # if args.train_loss:
    #   plot_train_loss(args)

    # if args.test_loss:
    #   plot_test_loss(args)

    plot_train_test_loss(args)

    if args.accuracy:
      plot_accuracy(args)

    if args.invalids:
      plot_invalids(args)