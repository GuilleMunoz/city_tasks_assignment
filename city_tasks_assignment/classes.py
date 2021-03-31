import numpy as np
import networkit as nk
from itertools import combinations
from cvxopt.glpk import ilp
from cvxopt import matrix
from math import comb
from os.path import isfile


class Problem:

    def __init__(self, file_name=None, dists_given=False):
        # task_loc[0] = location of the base
        self.tasks_loc = []
        # task_times[0] = 0
        self.tasks_times = []

        self.tasks_dists = []

        self.days = 0
        self.shifts = 2

        if file_name is not None:
            self.load(file_name, dists_given=dists_given)

    def compute_dists(self, graph):
        spsp = nk.distance.SPSP(graph, self.tasks_loc)
        spsp.run()
        self.tasks_dists = np.array([[ls[task] for task in self.tasks_loc] for ls in spsp.getDistances()])

    def create_random(self, n_tasks, n_nodes, n_edges):
        """
        Creates random problem.

        Args:
            n_tasks (int):
            n_nodes (int):
            n_edges (int):
        """
        from random import randint, sample, uniform

        self.days = randint(1, 20)
        self.tasks_loc = sample(range(n_nodes), n_tasks)
        self.tasks_times = [.0] + [uniform(1, 50) for _ in range(n_tasks - 1)]

        graph = nk.graph.Graph(n_nodes, weighted=True, directed=True)
        edges = set()
        while len(edges) < n_edges:
            edges.add((randint(0, n_nodes - 1), randint(0, n_nodes - 1)))
        for from_, to in edges:
            graph.addEdge(from_, to, uniform(1, 50))

        self.compute_dists(graph)

    def load(self, file_name, dists_given=False):
        """
        *************** "file_name.tasks" ***************

        <Number of days (int)> D

        <base location (int)> n

        <tasks locations on the graph (ints)> n1 n2 ... nj
        <tasks times (floats)> t1 t2 ... tj

        % OPTIONAL
        <distance from 0 to the others (floats)> d00 d01 ... d0j
        <distance from 1 to the others (floats)> d10 d11 ... d1j
        ...
        <distance from j to the others (floats)> dj0 dj1 ... djj


        If the distances are given it doesn't load the graph
        *************** "file_name.graph" ***************

        <edge1 (ints)> n1 n2 w1
        <edge2 (ints)> n1 n3 w2
        ...
        <edgei (ints)> nj nk wi

        Args:
            file_name (str):
            dists_given (bool): True if the distances are given in "file_name.tasks"
        """

        with open(file_name, 'r') as tasks_file:

            self.days = int(tasks_file.readline()[:-1])
            self.tasks_loc = [int(tasks_file.readline()[:-1])] + list(map(int, tasks_file.readline()[:-1].split(' ')))
            self.tasks_times = list(map(float, tasks_file.readline()[:-1].split(' ')))

            self.tasks_loc = np.array(self.tasks_loc)
            self.tasks_times = np.array(self.tasks_times)

            if dists_given:
                self.tasks_dists = np.zeros((self.tasks_times.shape[0], self.tasks_times.shape[0]))
                for i, line in enumerate(tasks_file.readlines()):
                    dists = list(map(float, line[:-1].split(' ')))
                    self.tasks_dists[0] = np.array(dists)

        if not dists_given and isfile(file_name + '.graph'):
            graph = nk.readGraph(file_name + '.graph', nk.Format.EdgeList, separator=' ', firstNode=0, directed=True)
            self.compute_dists(graph)

    def show_sol(self, ls):
        """
        Displays the solution (ls) obtained by optimize.

        Args:
            ls (list):

        """
        T = int(self.tasks_dists.shape[0])

        print('\n\n************************ SOLUTION ************************\n')

        def format_(i):
            if i == 0:
                return '\tBase'
            return 'Tarea {}'.format(i)

        total_time = 0
        to_str = lambda x: '0{}'.format(x) if x < 10 else str(x)

        for l in range(self.days):
            for k in range(self.shifts):
                print('Dia {} Turno {}:'.format(l, k))
                i = 0
                time = 0
                last = -1
                ss = set()

                for _ in range(T):
                    if ls[(2 * l + k) * T * T] or i in ss:
                        break
                    print(format_(i), end=' -> ')
                    ss.add(i)
                    last = i
                    i = ls[(2 * l + k) * T * T + i * T: (2 * l + k) * T * T + T * (i + 1)].index(1)
                    time += self.tasks_dists[last, i] + self.tasks_times[i]

                total_time += time

                if last != -1:
                    print('Base')
                    print('\tDuración = {}:{} horas'.format(int(time),
                                                            to_str(int((time - int(time)) * 60))))

        print('\nDuración total = {}:{} horas'.format(int(total_time),
                                                      to_str(int((total_time - int(total_time)) * 60))))

    def optimize(self):
        """
        Optimizes the given problem.
        Returns:
            (list)
        """
        # T = number of tasks
        # D = number of days
        # c = [x_0001, x_0101 , ..., x_1001, ..., x_0011, ..., x_0002, ..., x_TT1D]
        T = int(self.tasks_dists.shape[0]) - 1
        dists = self.tasks_dists.flatten()

        max_tasks_per_day = int(
            8 / (np.min(self.tasks_times[1:]) + np.min(dists[np.arange(dists.shape[0]) % (T + 2) > 0])))
        max_tasks_per_day = min(max_tasks_per_day, T) + 1
        times = np.repeat(self.tasks_times, T + 1)
        c = np.tile(dists, self.days * self.shifts)

        n_b = T  # (1)
        n_b += self.days * self.shifts * T  # (2)
        n_b += 2 * self.days * self.shifts  # (3)
        n_b += 1  # (4)
        A = np.zeros((n_b, c.shape[0]))
        b = np.zeros((n_b,))

        n_h = self.days * self.shifts  # (4)
        n_h += sum([comb(T + 1, i) for i in range(2, max_tasks_per_day)]) * T * 2  # (5)
        G = np.zeros((n_h, c.shape[0]))
        h = np.zeros((n_h,))

        indices = np.arange(c.shape[0]) % ((T + 1) * (T + 1))

        def restr_1(shift_):
            # (1): Se realizan todas las tareas una vez.

            for i in range(1, T + 1):
                # A[shift_, (((T+1) * i) < indices) & (indices < ((T+1) * (i+1)))] = 1
                A[shift_, indices % (T + 1) == i] = 1
                b[shift_] = 1
                shift_ += 1

            return shift_

        def restr_2(shift_):
            # (2): Para cada turno, despues de una tarea siempre va otra

            for l in range(0, self.days):
                for k in range(0, self.shifts):
                    row_shift = (T + 1) * (T + 1) * (l * 2 + k)
                    for i in range(1, T + 1):
                        A[shift_, row_shift + i * (T + 1): row_shift + (i + 1) * (T + 1)] = 1
                        A[shift_, row_shift + i: row_shift + i + (T + 1) * (T + 1): (T + 1)] = -1
                        A[shift_, row_shift + i * (T + 1) + i] = 0

                        shift_ += 1

            return shift_

        def restr_3(shift_):
            # (3): En cada turno hay que salir y llegar de la base (la tarea 0)
            for l in range(self.days):
                for k in range(self.shifts):
                    # x_{0jkl}
                    A[shift_, (T + 1) * (T + 1) * (l * 2 + k): (T + 1) * (T + 1) * (l * 2 + k) + T + 1] = 1
                    b[shift_] = 1

                    # x_{i0kl}
                    A[shift_ + self.days * self.shifts,
                    (T + 1) * (T + 1) * (l * 2 + k): (T + 1) * (T + 1) * (l * 2 + k + 1): T + 1] = 1
                    b[shift_ + self.days * self.shifts] = 1

                    # x_{00kl}
                    # A[shift_, (T + 1) * (T + 1) * (l * 2 + k)] = 0
                    # A[shift_ + self.days * self.shifts, (T+1)*(T+1)*(l * 2 + k)] = 0

                    shift_ += 1

            return shift_

        def restr_4(shift_):
            # (4): No se puede ir desde una tarea a la misma (menos desde la base)

            A[shift_, (indices > 0) & (indices % (T + 2) == 0)] = 1
            return shift_ + 1

        def restr_5(shift_):
            # (5): Cada turno dura 8 horas
            for l in range(0, self.days):
                for k in range(0, self.shifts):
                    G[shift_, (T + 1) * (T + 1) * (l * 2 + k): (T + 1) * (T + 1) * (l * 2 + k + 1)] = dists + times
                    h[shift_] = 8
                    shift_ += 1

            return shift_

        def restr_6(shift_):
            # (6): Hay que tener en cuenta que no se hagan ciclos.
            combs = [tup for n in range(2, max_tasks_per_day) for tup in combinations(range(1, T + 1), n)]

            for l in range(0, self.days):
                for k in range(0, self.shifts):
                    row_shift = (T + 1) * (T + 1) * (l * 2 + k)
                    for S in combs:
                        for i in S:
                            for j in S:
                                G[shift_, row_shift + i * (T + 1) + j] = 1
                        h[shift_] = len(S) - 1
                        shift_ += 1

            return shift_

        shift = restr_1(0)
        shift = restr_2(shift)
        restr_4(restr_3(shift))
        shift = 0
        shift = restr_5(shift)
        shift = restr_6(shift)

        c = matrix(c)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        return list(ilp(c=c, A=A, b=b, G=G, h=h, B=set(range(c.size[0])))[1])
