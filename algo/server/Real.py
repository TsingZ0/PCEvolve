import time
from algo.client.Real import Client as Real


class Server(object):
    def __init__(self, args):
        self.client = Real(args)
        self.args = args

        self.train_acc = []
        self.train_loss = []
        self.test_acc = []
        

    def eval(self):
        train_metrics = self.client.train_metrics()
        eval_metrics = self.client.eval_metrics()

        print('\nTrain metrics:')
        if train_metrics:
            for metric, value in train_metrics.items():
                if metric == 'Loss':
                    print(metric, '=', f'{value:.4f}')
                else:
                    print(metric, '=', f'{value[0]:.4f}')
                    print(metric, 'per label =', *[f'{i}:{v:.4f}' for i, v in enumerate(value[1])])
            self.train_acc.append(train_metrics['Accuracy'][0])
            self.train_loss.append(train_metrics['Loss'])
            
        print('\nTest metrics:')
        for metric, value in eval_metrics.items():
            print(metric, '=', f'{value[0]:.4f}')
            print(metric, 'per label =', *[f'{i}:{v:.4f}' for i, v in enumerate(value[1])])
        self.test_acc.append(eval_metrics['Accuracy'][0])

    def run(self):
        print(f"\n-------------Initial evaluation-------------")
        self.eval()
        start = time.time()
        print(f"\n-------------Iter. number: {0}-------------")

        self.client.run()
        self.eval()

        print(f"\nTotal running cost per iter.: {round(time.time()-start, 2)}s.\n")
