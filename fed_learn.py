import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F

import multiprocessing as mp
import random
import warnings


to_tensor = transforms.ToTensor()
# normalize = transforms.Normalize()

TRAIN_SET = MNIST(root='./data', train=True, download=True, transform=to_tensor)
TEST_SET = MNIST(root='./data', train=False, download=True, transform=to_tensor)

TRAIN_SET = data_utils.TensorDataset(TRAIN_SET.data.flatten(start_dim=1) / 255.0,
                                     TRAIN_SET.targets)
TEST_SET = data_utils.TensorDataset(TEST_SET.data.flatten(start_dim=1) / 255.0,
                                    TEST_SET.targets)


DEFAULT_SERVER = {
    'num_workers': 2,
    'worker_lr': 0.001,
    'worker_steps': 10,
}

DEFAULT_WORKER = {
    'lr': 0.001,
    'slowness': 0
}


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.log_softmax(x, 1)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(784, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        return torch.log_softmax(self.output(x), 1)


class BaseServer(object):

    def __init__(self, **kwargs):

        self._config = kwargs.pop('config', DEFAULT_SERVER)
        assert isinstance(self._config, dict)

        self._all_workers = dict()
        self._train_data = kwargs.pop('train_data', TRAIN_SET)
        self._test_data = kwargs.pop('test_data', TEST_SET)
        self._sampler = kwargs.pop('sampler', None)

        self._working_workers = None
        self._data_distribution = dict()

        self.num_workers = self._config['num_workers']
        self._worker_steps = self._config['worker_steps']
        self._worker_lr = self._config['worker_lr']

        if self._train_data is None:
            raise ValueError('No training data provided')

        self._model = MLP()
        self._test_loss = list()
        self._test_accuracy = list()
        self._round_sgd_steps = list()

    def _spawn_new_worker(self, worker_id, dataset):
        self._all_workers[worker_id] = BaseWorker(id=worker_id,
                                                  dataset=dataset,
                                                  lr=self._worker_lr,
                                                  steps=self._worker_steps)
        return

    def _split_dataset(self, num_workers):
        assert isinstance(num_workers, int)

        if self._sampler is None:
            split_lst = [int(len(self._train_data)//num_workers)]*num_workers
            for _ in range(int(len(self._train_data) % num_workers)):
                split_lst[_] += 1
            set_lst = data_utils.random_split(self._train_data, split_lst)
        else:
            raise NotImplementedError

        return set_lst

    def assign_data_to_workers(self, *args, **kwargs):

        data_dist = self._split_dataset(self.num_workers)

        for _ in range(self.num_workers):
            self._spawn_new_worker(worker_id=_, dataset=data_dist[_])
            self._data_distribution[_] = len(data_dist[_])

    def _collect_results(self):
        results = dict()
        if self._working_workers is None:
            self._working_workers = list(self._all_workers.keys())
        for worker_id in self._working_workers:
            results[worker_id] = self._all_workers[worker_id].training()  # (param, completed steps)

        return results

    def _process_results(self, results):
        """Implement FedAvg"""
        assert isinstance(results, dict)
        state_dict = dict()
        round_sgd_steps = 0
        total_num_data = 0
        with torch.no_grad():
            for w_id in results:
                total_num_data += self._data_distribution[w_id]
                for item in results[w_id][0]:
                    if item not in state_dict:
                        state_dict[item] = 0
                    if results[w_id][1] == self._worker_steps:
                        state_dict[item] += self._data_distribution[w_id] * results[w_id][0][item]
                        round_sgd_steps += results[w_id][1]

            for item in state_dict:
                state_dict[item] /= total_num_data
        self._model.load_state_dict(state_dict)
        self._round_sgd_steps.append(round_sgd_steps)

        return state_dict

    def send_instruction(self, *args, **kwargs):
        assert 'state_dict' in kwargs or len(args) == 1

        state_dict = kwargs.pop('state_dict', args[0] if len(args) == 1 else None)
        for w_id in self._working_workers:
            self._all_workers[w_id].receive_instruction(state_dict=state_dict)

    def _add_workers(self, work_ids=None, num_workers=0, **kwargs):
        assert isinstance(work_ids, (list, tuple, type(None)))
        if work_ids:
            if num_workers > 0:
                warnings.warn('\'num_workers\' has no effect when \'worker_ids\' specified')
            count_dup = dict()
            for w_id in work_ids:
                assert w_id in self._all_workers.keys()
                if w_id not in count_dup:
                    count_dup[w_id] = 1
                else:
                    count_dup += 1
            for w_id in count_dup:
                while count_dup[w_id] > 1:
                    work_ids.remove(w_id)
                    count_dup[w_id] -= 1
            self._working_workers += list(work_ids)
        else:
            assert isinstance(num_workers, int)
            if num_workers > 0:
                if num_workers > self.num_workers:
                    self._working_workers += list(self._all_workers.keys())
                else:
                    self._working_workers += random.sample(self._all_workers.keys(),
                                                           k=num_workers)

    def _drop_worker(self, work_ids=None, num_workers=0, **kwargs):
        assert isinstance(work_ids, (list, tuple, type(None)))
        if work_ids:
            if num_workers > 0:
                warnings.warn('\'num_workers\' has no effect when \'worker_ids\' specified')
            for w_id in work_ids:
                try:
                    self._working_workers.remove(w_id)
                except ValueError:
                    pass
        else:
            assert isinstance(num_workers, int)
            if num_workers > 0:
                if num_workers >= self.num_workers:
                    self._working_workers = list()
                else:
                    drop_lst = random.sample(self._all_workers.keys(),
                                             k=num_workers)
                    for w_id in drop_lst:
                        try:
                            self._working_workers.remove(w_id)
                        except ValueError:
                            pass

    def evaluate(self, curr_round):
        self._model.eval()
        test_loss = 0
        accuracy = 0
        # test_loader = data_utils.DataLoader(self._test_data, batch_size=1000, shuffle=True)
        with torch.no_grad():
            for data, target in self._test_data:
                output = self._model(data)
                test_loss += F.binary_cross_entropy_with_logits(output, target).item()
                pred = output.data.argmax()
                accuracy += pred.eq(target.data.argmax()).sum().item()
        test_loss /= len(self._test_data)
        self._test_loss.append((curr_round, test_loss))
        accuracy = 100 * accuracy / len(self._test_data)
        self._test_accuracy.append((curr_round, accuracy))
        print(f'Round {curr_round}: accuracy {accuracy:.2f}%')

    def routine(self, total_round):
        assert isinstance(total_round, int) and total_round > 0

        self.assign_data_to_workers()
        self._add_workers()

        self.evaluate(0)
        for rnd in range(1, total_round+1):
            res = self._collect_results()
            s_dict = self._process_results(res)
            self.send_instruction(state_dict=s_dict)
            if rnd % 50 == 0:
                self.evaluate(rnd)


class BaseWorker(object):

    def __init__(self, **kwargs):

        self._config = kwargs.pop('config', None)
        self._id = kwargs.pop('id', 0)
        self._train_set = kwargs.pop('dataset', TRAIN_SET)
        self._lr = kwargs.pop('lr', 1e-3)
        self._batch_size = kwargs.pop('batch_size', 128)
        self._steps = kwargs.pop('steps', 5)
        self.available = True

        self._model = kwargs.pop('model', MLP)()
        self._loss_func = kwargs.pop('loss', nn.NLLLoss)()
        self._optimizer = kwargs.pop('optimizer', torch.optim.SGD(self._model.parameters(), lr=self._lr))
        self._train_loader = data_utils.DataLoader(self._train_set,
                                                   batch_size=self._batch_size,
                                                   num_workers=0,
                                                   shuffle=True)
        self._model.train()
        self._train_loss = list()
        self._sdg_step_count = list()

    def _sdg_step(self, *args, **kwargs):
        assert 'batch' in kwargs
        inputs, labels = kwargs.pop('batch', None)
        self._optimizer.zero_grad()
        outputs = self._model(inputs)
        loss = self._loss_func(outputs, labels)
        loss.backward()
        self._optimizer.step()
        self._train_loss.append(loss.item())

    def _send_results(self):
        state_dict = self._model.state_dict()
        return state_dict

    def training(self):
        bat_idx = 0
        for minibatch in self._train_loader:
            self._sdg_step(batch=minibatch)
            bat_idx += 1
            if bat_idx >= self._steps:
                break
        self._sdg_step_count.append(bat_idx)
        return self._send_results(), bat_idx

    def receive_instruction(self, *args, **kwargs):
        assert 'state_dict' in kwargs or len(args) == 1

        state_dict = kwargs.pop('state_dict', args[0] if len(args) == 1 else None)
        self._model.load_state_dict(state_dict)


def test_training(epoch=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ts = MNIST('./data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))]))

    te = MNIST('./data', train=False, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))]))

    train_loader = torch.utils.data.DataLoader(ts, batch_size=128, shuffle=True)

    test_loader = data_utils.DataLoader(te, batch_size=1000, shuffle=True)

    net = MLP().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_func = nn.NLLLoss().to(device)
    train_loss = []
    train_step = []
    steps = 0
    net.train()
    for ep in range(1, epoch + 1):
        for bat_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            steps += 1
            opt.zero_grad()
            outputs = net(inputs.flatten(start_dim=1))
            loss = loss_func(outputs, labels)
            loss.backward()
            opt.step()

            if steps % 100 == 0:
                print(f'Epoch {ep} Step {steps}: Train Loss: {loss.item(): .3f}')
                train_loss.append(loss.item())
                train_step.append(steps * 128)
                # print(outputs)
                # print(labels)

    test_loss = 0
    accuracy = 0
    net.eval()
    with torch.no_grad():
        for d, t in test_loader:
            d = d.to(device)
            t = t.to(device)
            ot = net(d.flatten(start_dim=1))
            test_loss += nn.NLLLoss()(ot, t).item()
            pred = ot.argmax(1)
            accuracy += pred.eq(t).sum().item()
    test_loss /= len(TEST_SET)
    print(f'Test loss: {test_loss: .3f}')
    accuracy /= len(TEST_SET)
    print(f'Accuracy: {accuracy * 100: .2f}%')


if __name__ == '__main__':
    # ser = BaseServer()
    # ser.routine(10)
    epoch = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ts = MNIST('./data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))]))

    te = MNIST('./data', train=False, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))]))

    train_loader = torch.utils.data.DataLoader(ts, batch_size=128, shuffle=True)

    test_loader = data_utils.DataLoader(te, batch_size=1000, shuffle=True)

    net = MLP().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_func = nn.NLLLoss().to(device)
    train_loss = []
    train_step = []
    steps = 0
    net.train()
    for ep in range(1, epoch + 1):
        for bat_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            steps += 1
            opt.zero_grad()
            outputs = net(inputs.flatten(start_dim=1))
            loss = loss_func(outputs, labels)
            loss.backward()
            opt.step()

            if steps % 100 == 0:
                print(f'Epoch {ep} Step {steps}: Train Loss: {loss.item(): .3f}')
                train_loss.append(loss.item())
                train_step.append(steps * 128)
                # print(outputs)
                # print(labels)

    test_loss = 0
    accuracy = 0
    net.eval()
    with torch.no_grad():
        for d, t in test_loader:
            d = d.to(device)
            t = t.to(device)
            ot = net(d.flatten(start_dim=1))
            test_loss += nn.NLLLoss()(ot, t).item()
            pred = ot.argmax(1)
            accuracy += pred.eq(t).sum().item()
    test_loss /= len(TEST_SET)
    print(f'Test loss: {test_loss: .3f}')
    accuracy /= len(TEST_SET)
    print(f'Accuracy: {accuracy * 100: .2f}%')