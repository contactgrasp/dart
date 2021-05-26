import sys
sys.path.append('')
from libpydart import GraspAnalyser
import multiprocessing as mp
from multiprocessing.queues import SimpleQueue
from itertools import product
import numpy as np


def worker_fn(object_name, session_name, task_q, result_q):
    ga = GraspAnalyser(object_name, session_name)
    while not task_q.empty():
        params = task_q.get()
        ga.set_params(*params)
        r = ga.analyze_grasps(5, 1, 1, True)
        result = [p for p in params]
        result.append(r)
        result_q.put(tuple(result))
    ga.close()


def create_task_q():
    aw =  [50, 100]
    rw =  [50, 100]
    tw =  [25]
    ad =  [2]
    rd =  [2]
    iw =  [10]
    lmd = [5, 10]
    reg = [-2, -4]
    q = SimpleQueue()
    for p in product(aw, rw, tw, ad, rd, iw, lmd, reg):
        q.put(p)
    return q


class GridSearcher(object):
    def __init__(self):
        sessions = [['camera-0', 'full19_use']]
                    # ['binoculars', 'full19_handoff']]
        # n_total_workers = mp.cpu_count()
        n_total_workers = 2
        n_workers = [n_total_workers / len(sessions)] * len(sessions)
        n_workers[-1] = n_total_workers - sum(n_workers[:-1])
        task_qs = [create_task_q()] * len(sessions)
        self.result_qs = [SimpleQueue()] * len(sessions)

        self.processes = []
        for session, n_w, task_q, result_q in \
            zip(sessions, n_workers, task_qs, self.result_qs):
            for _ in xrange(n_w):
                p = mp.Process(target=worker_fn, args=(session[0], session[1],
                                                       task_q, result_q))
                self.processes.append(p)

    def run(self):
        for p in self.processes:
            p.start()

        for p in self.processes:
            p.join()

        results = {}
        for q in self.result_qs:
            while not q.empty():
                r = q.get()
                if r[:-1] not in results:
                    results[r[:-1]] = [r[-1]]
                else:
                    results[r[:-1]].append(r[-1])
        return results


if __name__ == '__main__':
    gs = GridSearcher()
    results = gs.run()
    print(results)
    avg_errors = {k: np.mean(v) for k,v in results.items()}
    best_params = min(avg_errors, key=avg_errors.get)
    print('Best params = ', best_params)
