from byz import Server_torch_auto, Worker_torch
import torch
from torch.utils.data import DataLoader, Sampler, Subset, Dataset
import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import random
from read_logs import run_meta, read_meta_folder, run_stats, run_meta_update_logs, run_diagnosis
from new_plots import generate_plots
import datetime


class FedServer(Server_torch_auto):

    def __init__(self, **kwargs):
        self.worker_learning_rate = kwargs.pop('learning_rate', 0.001)
        self.worker_lr_schedule = kwargs.pop('worker_lr_schedule', 'const')
        self.worker_lr_decay = kwargs.pop('worker_lr_decay', 0)
        super(FedServer).__init__(**kwargs)

    def assign_workers(self):
        worker_speeds = torch.randint(1, 5, (self.num_workers,)).numpy()

        for k in range(self.num_workers):
            subset = Subset(self.train_data, self.partitions[k])
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True, drop_last=False)
            worker = FedWorker(self, k, loader, self.batch_size, self.model_name,
                               self.device, self.loss_fn_,
                               type_=self.adv_ids[k],
                               abs_=self.abs_,
                               worker_speed=worker_speeds[k],
                               worker_lr=0.001,
                               worker_regularization=0.1)

            self.workers.append(worker)
            self.worker_kappas.append(worker.const)


class FedWorker(Worker_torch):

    def __init__(self, *args, **kwargs):
        self._worker_iterations = kwargs.pop('worker_speed', 1)
        self._learning_rate = kwargs.pop('worker_lr', 0.001)
        self._init_lr = self._learning_rate
        self._regularization = kwargs.pop('worker_regularization', 1)
        self._rl_schedule = kwargs.pop('worker_rl_schedule', 'const')
        self._rl_decay = kwargs.pop('worker_rl_decay', 0)
        super(FedWorker).__init__(*args, **kwargs)

        self._step_counter = 0
        self._round_counter = 0

    def _update_rl(self):
        """
        Call at each round or each step?
        :return:
        """
        if self._rl_schedule == 'const':
            self._learning_rate = self._init_lr
        elif self._rl_schedule == 'decay':
            self._learning_rate = self._init_lr / (1 + self._rl_decay * self._step_counter)
        elif self._rl_schedule == 'new':
            self._learning_rate = self._init_lr / (1 + self._rl_decay * self._step_counter)

        return

    def _update_parameters(self, updated_grads):
        self.model.cpu()
        updates = []
        with torch.no_grad():
            for param1, param2 in zip(self.model.parameters(), updated_grads):
                update = param2.data + self._regularization * param1.data
                param1.data -= self._learning_rate * update
                updates.append(update)
        self.model.to(self.device)
        return updates

    def _compute_gradient(self):
        self._step_counter += 1

        data_X, data_Y = self.get_data()
        self.get_server_weights()

        out = self.model.forward(data_X)
        loss = self.loss_fn(out, data_Y)
        _, predicted = torch.max(out.data, 1)

        correct = 0
        if self.loss_fn_ == "cross_entropy":
            correct = (predicted == data_Y).sum().item()

        self.model.zero_grad()

        loss.backward()

        if self.adversary_type == 0:
            # benign gradient
            grads = [param.grad.data.cpu() for param in self.model.parameters()]

        elif self.adversary_type == 1:
            # random mean 0 gradient
            # since self.const = 0
            grads = [self.abs * torch.randn_like(param.grad.data, device='cpu') for param in self.model.parameters()]

        elif self.adversary_type == 2:
            # -ve sign gradient
            grads = [self.mean * param.grad.data.cpu() for param in self.model.parameters()]

        elif self.adversary_type == 3:
            # -ve gradient with random scale with scale mean > 0
            # temp = self.const * (torch.randn(1) + 1)
            # since self.const = -1, and a random number around it is generated
            temp = self.sampler.sample() * self.abs
            grads = [temp * param.grad.data.cpu() for param in self.model.parameters()]

        else:
            raise NotImplementedError

        return grads, loss.data, correct, data_Y.size(0)

    def compute_gradient(self, iterations=None):
        self._round_counter += 1
        if iterations is None:
            iterations = self._worker_iterations
        gradient, last_loss, last_correct, last_batch_size = [None] * 4
        for _ in iterations:
            grad, last_loss, last_correct, last_batch_size = self._compute_gradient()
            updates = self._update_parameters(grad)
            if gradient is None:
                gradient = updates
            else:
                for p_1, p_2 in zip(gradient, updates):
                    p_1 += p_2

        return gradient, last_loss, last_correct, last_batch_size


def get_arguments():
    parser = argparse.ArgumentParser(description='Byzantine')

    lr_schedules = ["decay", "const", "new"]
    datasets = ["synth", "MNIST", "CIFAR10"]
    loss_fns = ["cross_entropy", "MSELoss"]
    models = ["vggsmall", "vggbig", "lenet", "lenetsmall", "linear", "3lenet"]
    update_rules = ["ByGARS", "ByGARS++", "benign_aux", "average", "aux", "custom"]
    q_init_choices = ["zero", "random", "uniform"]

    parser.add_argument('--dataset', type=str, required=True, default="synth", choices=datasets)
    parser.add_argument('--num-epochs', type=int, required=True, default=0)
    parser.add_argument('--batch-size', type=int, required=True, default=64, help='Num samples per batch')
    parser.add_argument('--super-meta-name', type=str, default='update_rule', required=True,
                        choices=['update_rule', 'aux_size', 'num_meta_itr'], help='Type of experiment')
    parser.add_argument('--num-trials', required=True, type=int, default=1)

    parser.add_argument('--num-good-workers', type=int, default=1, help='Non-adversarial workers')
    parser.add_argument('--log-freq', type=int, default=10, help='test error is evaluated at this freq')
    # 0 log freq implies, log when iter == 0 (once for every epoch)
    parser.add_argument('--verbose', type=bool, default=False, help='print details at a freq')
    parser.add_argument('--manual-seed', type=int, default=1, help='seed of simulation')

    # optional arguments
    # parser.add_argument('--num-workers', type=int, default=5, metavar='Nw', help='Number of workers')
    parser.add_argument('--aux-size', type=int, default=100, metavar='Ax', help='Size of auxiliary dataset')
    parser.add_argument('--num-meta-itr', type=int, default=1, help='num meta updates to q')

    parser.add_argument('--location', type=str, default=data_folder, help='folder')
    parser.add_argument('--filename', type=str, default="log.csv", help='log file name')
    parser.add_argument('--loss-fn', type=str, default="MSELoss", choices=loss_fns, help='loss function')
    parser.add_argument('--model', type=str, default="linear", choices=models, help='Pytorch model')

    parser.add_argument('--update-rule', type=str, default='ByGARS', choices=update_rules,
                        help='Gradient aggregation rule')
    parser.add_argument('--custom-update-rule', type=str, default='None', help='Not implemented, ignore')

    # step size arguments
    parser.add_argument('--learning-rate', type=float, default=0.05, help='learning rate ')
    parser.add_argument('--regularization', type=float, default=0, help='regularization')
    parser.add_argument('--lr-decay', type=float, default=0.0, help='lr decay parameter')
    parser.add_argument('--meta-learning-rate', type=float, default=0.05, help='lr for q updates')
    parser.add_argument('--meta-regularization', type=float, default=0, help='meta regularization for q update')
    parser.add_argument('--meta-lr-decay', type=float, default=0.1, help='decay parameter of meta-lr')
    parser.add_argument('--lr-schedule', type=str, default="new", choices=lr_schedules, help='lr decay type')

    # gradient norms related
    parser.add_argument('--max-grad-norm', type=float, default=5.0, help='clip gradients with this max norm')
    parser.add_argument('--normalize-weights', type=bool, default=False, help='normalize q_t reputation score to 1')
    parser.add_argument('--clip-aux', type=bool, default=True, help='clip aux gradient with max-grad-norm')
    parser.add_argument('--clip-with-aux', type=bool, default=False, help='clip gradients with norm of aux gradient')
    parser.add_argument('--abs', type=int, default=100, help='scaling constant of adversaries')

    # not used now or automatically set based on previous args chosen
    parser.add_argument('--implicit-reg', type=bool, default=False,
                        help='use gradient of regularization term in meta update')
    parser.add_argument('--q-init', type=str, default="zero", choices=q_init_choices,
                        help='initialize reputation scores')
    parser.add_argument('--use-lr-inmeta', type=bool, default=False,
                        help='use lr in meta update of q_t. True for ByGARS, False for ByGARS++')
    parser.add_argument('--meta-update-type', type=str, default='gradupdate', choices=['gradupdate', 'stochavg'],
                        help=r'stochavg induces regularization -> q = (1-a)q + a \grad; gradupdate -> q = q + a \grad')

    return parser.parse_args()


if __name__ == '__main__':
    # num_workers = 5
    # adv_ids = [0] * 5
    #
    # server_config = {'num_workers': num_workers,
    #                  'adv_ids': adv_ids,
    #                  'aux_size': 100,
    #                  'model': "vggsmall",
    #                  'loss_fn': "cross_entropy",
    #                  'dataset': "CIFAR10",
    #                  'location': "data/",
    #                  'batch_size': 64,
    #                  'folder': "logs/",
    #                  'filename': "log_00.csv",
    #                  'num_meta_itr': 0,
    #                  'learning_rate': 0.05,
    #                  'meta_learning_rate': 0.01,
    #                  'meta_regularization': 1,
    #                  'lr_decay': 0.05,
    #                  'lr_schedule': 'decay',
    #                  'meta_lr_decay': 0.05,
    #                  'q_init': 'zero'}
    #
    # fed_server = FedServer(**server_config)

    logs_folder = "new_final_logs/"
    data_folder = "data/"

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    args = get_arguments()

    dataset = args.dataset
    if not os.path.exists(logs_folder + dataset + "/"):
        os.makedirs(logs_folder + dataset + "/")

    config = vars(args)

##############################################
    """
    Directory structure: 
    -logs_folder -> dataset -> comparision [updaterule, auxsize, etc] -> config name -> trials
    """

    if config["dataset"] == "synth":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float)

    sup_time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    super_meta_folder = logs_folder + dataset + '/' + config['super_meta_name'] + "_" + sup_time_stamp + '/'
    if not os.path.exists(super_meta_folder):
        os.makedirs(super_meta_folder)

    meta_config = {}
    key = config['super_meta_name']

    # use super meta to determine the type of experiment to simulate
    if config['super_meta_name'] == 'update_rule':
        vals = ['ByGARS++', 'ByGARS', 'benign_aux']
        # vals = ['ByGARS++', 'ByGARS', 'benign_aux', 'average']
    elif config['super_meta_name'] == 'aux_size':
        vals = [50, 100, 150, 200, 250]
    elif config['super_meta_name'] == 'num_meta_itr':
        vals = [1, 2, 3, 4]
        config['update_rule'] = 'ByGARS'
    else:
        raise NotImplementedError

    meta_config[key] = vals
    with open(super_meta_folder + "config.json", 'w') as fp:
        json.dump(meta_config, fp, indent=2)

    for val in meta_config[key]:

        exp_folder = super_meta_folder + key + "_" + str(val) + '/'
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        config[key] = val

        if config['dataset'] in ['MNIST', 'synth']:
            """
                {0: benign
                 1: random mean 0
                 2: -ve flipped gradient with constant scale
                 3: -ve flipped gradient with random scale
            """

            if config['num_good_workers'] == 0:
                adv_ids_ = [1, 2, 2, 2, 3, 3]
            else:
                adv_ids_ = [0, 1, 2, 2, 3, 3]

        elif config['dataset'] == 'CIFAR10':
            if config['num_good_workers'] == 0:
                adv_ids_ = [1, 2, 3, 3]
            else:
                adv_ids_ = [0, 1, 2, 3]

        else:
            raise NotImplementedError

        if config['dataset'] == 'MNIST':
            config['model'] = 'lenet'
            config['loss_fn'] = "cross_entropy"
        elif config['dataset'] == 'CIFAR10':
            config['model'] = 'vggsmall'
            config['loss_fn'] = "cross_entropy"
        else:
            config['model'] = 'linear'
            config['loss_fn'] = "MSELoss"

        config["num_workers"] = len(adv_ids_)
        config["adv_ids"] = adv_ids_

        if config['update_rule'] == "ByGARS":
            config['meta_update_type'] = 'gradupdate'
            config['use_lr_inmeta'] = True

            if config['dataset'] == 'synth':
                config['learning_rate'] = 0.05
                config['lr_decay'] = 0.5
                config['regularization'] = 0
                config['meta_learning_rate'] = 0.05
                config['meta_lr_decay'] = 0.5
                config['meta_regularization'] = 0
                config['lr_schedule'] = 'new'

            elif config['dataset'] == 'MNIST':
                config['learning_rate'] = 0.05
                config['lr_decay'] = 0.5
                config['regularization'] = 0
                config['meta_learning_rate'] = 0.05
                config['meta_lr_decay'] = 0.5
                config['meta_regularization'] = 0
                config['lr_schedule'] = 'new'

            elif config['dataset'] == 'CIFAR10':
                config['learning_rate'] = 0.05
                config['lr_decay'] = 0.9
                config['regularization'] = 0.005
                config['meta_learning_rate'] = 0.05
                config['meta_lr_decay'] = 0.5
                config['meta_regularization'] = 0
                config['lr_schedule'] = 'new'

            else:
                raise NotImplementedError

        elif config['update_rule'] == "ByGARS++":
            config['meta_update_type'] = 'stochavg'
            config['use_lr_inmeta'] = False

            if config['dataset'] == 'synth':
                config['learning_rate'] = 0.05
                config['lr_decay'] = 0.9
                config['regularization'] = 0
                config['meta_learning_rate'] = 0.05
                config['meta_lr_decay'] = 0.7
                config['meta_regularization'] = 0
                config['lr_schedule'] = 'new'

            elif config['dataset'] == 'MNIST':
                config['learning_rate'] = 0.05
                config['lr_decay'] = 0.9
                config['regularization'] = 0.005
                config['meta_learning_rate'] = 0.0005
                config['meta_lr_decay'] = 0.2
                config['meta_regularization'] = 0
                config['lr_schedule'] = 'new'

            elif config['dataset'] == 'CIFAR10':
                config['learning_rate'] = 0.05
                config['lr_decay'] = 0.9
                config['regularization'] = 0.005
                config['meta_learning_rate'] = 0.01
                config['meta_lr_decay'] = 0.2
                config['meta_regularization'] = 0
                config['lr_schedule'] = 'new'

            else:
                raise NotImplementedError

        else:
            pass

        if config['manual_seed'] == -1:
            seed_ = torch.get_rng_state()
            prng = np.random.RandomState()
            np_seed_ = prng.get_state()
        else:
            random.seed(a=config['manual_seed'], version=2)
            torch.manual_seed(config['manual_seed'])
            seed_ = torch.get_rng_state()
            np.random.seed(config['manual_seed'])
            np_seed_ = np.random.get_state()

        pickle.dump(seed_, open(exp_folder + "seed.p", 'wb'))
        pickle.dump(np_seed_, open(exp_folder + 'npseed.p', 'wb'))

        with open(exp_folder + "config.json", 'w') as fp:
            json.dump(args.__dict__, fp, indent=2)

        for trial in range(config['num_trials']):

            folder = exp_folder + 'exp_' + str(trial) + '/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            config["folder"] = folder

            stor = Server_torch_auto(**config)
            print(config)

            try:
                stor.train()
                os.makedirs(config['folder'] + "stat_plots/")
                os.makedirs(config['folder'] + "plots/")
                os.makedirs(config['folder'] + "meta_log_plots/")

                print("plotting")
                df = pd.read_csv(config["folder"] + "stats.csv")
                run_stats(df, config['folder'] + "stat_plots/")

                df = pd.read_csv(config["folder"] + "log.csv")
                run_diagnosis(df, config['folder'] + "plots/", config['num_workers'], config['update_rule'])

                df = pd.read_csv(config["folder"] + "meta_log.csv")
                run_meta_update_logs(df, config['folder'] + "meta_log_plots/", config['num_workers'])

            except KeyboardInterrupt:
                stor.logger.save_log(stor.folder + stor.filename)
                print("exiting")
                sys.exit(0)
            except Exception as e:
                print("Error in train")
                print(e, type(e))

    generate_plots(super_meta_folder, meta_config, is_individual_plots=False)
