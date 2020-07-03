import multiprocessing as mp
import random as rd
import time

rd.seed(100)


class Server:

    def __init__(self):

        self.workers = []
        self.lock = mp.Lock()
        self.tasks = []
        self.task_queue = mp.Queue()
        self.done_queue = mp.Queue()

    def assign_workers(self, num_workers):

        for idx in range(num_workers):
            self.workers.append(Worker(parent=self, worker_id=idx))
            self.tasks.append(rd.uniform(3, 6))

    def start_workers(self):

        process = []

        for tk in self.tasks:
            self.task_queue.put(tk)

        for worker in self.workers:
            proc = mp.Process(target=worker.do_job, args=(self.lock, self.task_queue, self.done_queue))
            process.append(proc)
            proc.start()

        for idx in range(len(self.tasks)):
            res = self.done_queue.get()
            print(res)

        for idx in range(len(self.workers)):
            self.task_queue.put('STOP')

        for proc in process:
            proc.join()


class Worker:

    def __init__(self, parent, worker_id):

        self.parent = parent
        self.id = worker_id

    def do_job(self, lock, task_queue, done_queue):
        lock.acquire()
        print(f'Worker {self.id} start working...')
        lock.release()
        for sleep_time in iter(task_queue.get, 'STOP'):
            time.sleep(sleep_time)
            done_queue.put(f'worker {self.id} sleeping time {sleep_time}')
        lock.acquire()
        print(f'Worker {self.id} is done!')
        lock.release()


if __name__ == '__main__':
    server = Server()

    server.assign_workers(5)
    server.start_workers()






