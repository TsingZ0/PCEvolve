import time
from algo.server.ServerBase import ServerBase
from algo.client.selector.PCEvolve import Client as PCEvolve


class Server(ServerBase):
    def __init__(self, args):
        if args.selector == 'PCEvolve':
            args.Client = PCEvolve
        else:
            raise NotImplementedError
        super().__init__(args)
        
    def callback_per_iter(self):
        pass

    def run(self):
        print(f"\n-------------Initial evaluation-------------")
        self.eval()
        for i in range(self.args.iterations):
            self.it = i
            start = time.time()
            print(f"\n-------------Iter. number: {i}-------------")

            if not self.args.use_generated:
                self.receive()
                if self.done:
                    break
                self.generate()
                self.send()
            self.client.run()

            if i % self.args.eval_gap == 0:
                self.eval()
                
            if self.args.auto_break and self.check_done([self.test_acc]):
                break

            print(f"Total running cost per iter.: {round(time.time()-start, 2)}s.")
            self.callback_per_iter()

        if len(self.test_acc) > 2:
            train_best = max(self.train_acc[1:])
            test_best = max(self.test_acc[1:])
            FID_best = min(self.FID[1:])
            PSNR_best = min(self.PSNR[1:])
            print('Best train accuracy: {:.4f} at iter.'.format(train_best), self.train_acc[1:].index(train_best))
            print('Best test accuracy: {:.4f} at iter.'.format(test_best), self.test_acc[1:].index(test_best))
            print('Best FID: {:.4f} at iter.'.format(FID_best), self.FID[1:].index(FID_best))
            print('Best PSNR: {:.4f} at iter.'.format(PSNR_best), self.PSNR[1:].index(PSNR_best))

        self.callback()