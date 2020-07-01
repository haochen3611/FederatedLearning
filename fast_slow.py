from byz import Server_torch_auto, Worker_torch
import torch


class FedServer(Server_torch_auto):

    def __init__(self, **kwargs):

        super(FedServer).__init__(**kwargs)


class FedWorker(Worker_torch):

    def __init__(self, **kwargs):
        self._worker_iterations = kwargs.pop('worker_speed', 1)
        self._learning_rate = kwargs.pop('worker_lr', 0.001)
        self._regularization = kwargs.pop('worker_regularization', 1)
        super(FedWorker).__init__(**kwargs)

    def _update_parameters(self, updated_grads):
        self.model.cpu()
        updates = []
        with torch.no_grad():
            for param1, param2 in zip(self.model.parameters(), updated_grads):
                update = self._learning_rate * (param2.data + self._regularization * param1.data)
                param1.data -= update
                updates.append(update)
        self.model.to(self.device)
        return updates

    def compute_gradient_federated(self, iterations=None):

        if iterations is None:
            iterations = self._worker_iterations
        gradient, last_loss, last_correct, last_batch_size = [None]*4
        for _ in iterations:
            grad, last_loss, last_correct, last_batch_size = self.compute_gradient()
            updates = self._update_parameters(grad)
            if gradient is None:
                gradient = updates
            else:
                for p_1, p_2 in zip(gradient, updates):
                    p_1 += p_2

        return gradient, last_loss, last_correct, last_batch_size



