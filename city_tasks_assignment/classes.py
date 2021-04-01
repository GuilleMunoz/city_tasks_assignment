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
        self.teams = 1

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
        <Number of teams (int)> E

        <base location (int)> n

        <tasks locations on the graph (ints)> n1 n2 ... nj
        <tasks times of team 0 (floats)> t1 t2 ... -1 % -1 = inf
        <tasks times of team 1 (floats)> t1 -1 ... tj
        ...
        <tasks times of team E (floats)> -1 t2 ... -1

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
            self.teams = int(tasks_file.readline()[:-1])
            self.tasks_loc = [int(tasks_file.readline()[:-1])] + list(map(int, tasks_file.readline()[:-1].split(' ')))
            self.tasks_times = list(map(float, tasks_file.readline()[:-1].split(' ')))

            self.tasks_loc = np.array(self.tasks_loc)
            self.tasks_times = np.zeros((self.teams, self.tasks_loc.shape[0]))

            for i in range(self.teams):
                line = tasks_file.readline()
                times = list(map(float, line[:-1].split(' ')))
                self.tasks_times[i] = np.array(times)
                self.tasks_times[i, self.tasks_times[i] < 0] = np.inf

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

        arr = np.array(ls).reshape(self.days, self.shifts, self.teams, T, T)

        print('\n\n************************ SOLUTION ************************\n')

        def format_(i):
            if i == 0:
                return '\tBase'
            return 'Tarea {}'.format(i)

        total_time = 0
        to_str = lambda x: '0{}'.format(x) if x < 10 else str(x)

        for i in range(self.days):
            for j in range(self.shifts):
                print('Dia {} Turno {}:'.format(i, j))
                for k in range(self.teams):
                    print('  Equipo {}:'.format(k + 1))
                    index = 0
                    time = 0
                    last = -1
                    ss = set()

                    for _ in range(T):
                        if arr[i, j, k, 0, 0] or index in ss:
                            break
                        print(format_(index), end=' -> ')
                        ss.add(index)
                        last = index
                        index = np.where(arr[i, j, k, index] == 1)[0][0]
                        time += self.tasks_dists[last, index] + self.tasks_times[k, index]

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

        times = np.zeros((self.teams, T+1, T+1))
        for team in range(self.teams):
            times[team] = np.repeat(self.tasks_times[team], T + 1).reshape((T+1, T+1))

        max_tasks_per_day = int(8 / (np.min(self.tasks_times[:, 1:]) + np.min(self.tasks_dists[1:, 1:])))
        max_tasks_per_day = min(max_tasks_per_day, T) + 1

        c = np.tile(self.tasks_dists.flatten(), self.days * self.shifts * self.teams)

        n_b = T  # (1)
        n_b += self.days * self.shifts * self.teams * T  # (2)
        n_b += self.days * self.shifts * self.teams  # (3.1)
        n_b += self.days * self.shifts * self.teams  # (3.2)
        n_b += 1  # (4)
        A = np.zeros((n_b, self.days, self.shifts, self.teams, T+1, T+1))
        b = np.zeros((n_b,))

        n_h = self.days * self.shifts * self.teams  # (4)
        n_h += self.days * self.shifts * self.teams * sum([comb(T + 1, i) for i in range(2, max_tasks_per_day)])  # (5)
        G = np.zeros((n_h, self.days, self.shifts, self.teams, T+1, T+1))
        h = np.zeros((n_h,))

        def restr_1(shift_):
            # (1): Se realizan todas las tareas una vez.

            for l in range(1, T + 1):
                A[shift_, :, :, :, l, :] = 1
                b[shift_] = 1
                shift_ += 1

            return shift_

        def restr_2(shift_):
            # (2): Para cada turno, despues de una tarea siempre va otra

            for i in range(self.days):
                for j in range(self.shifts):
                    for k in range(self.teams):
                        for l in range(1, T + 1):
                            A[shift_, i, j, k, l, :] = 1
                            A[shift_, i, j, k, :, l] = -1
                            A[shift_, i, j, k, l, l] = 0

                            shift_ += 1

            return shift_

        def restr_3(shift_):
            # (3): En cada turno hay que salir y llegar de la base (la tarea 0)
            for i in range(self.days):
                for j in range(self.shifts):
                    for k in range(self.teams):
                        #x_{xijk0m}
                        A[shift_, i, j, k, 0, :] = 1
                        b[shift_] = 1

                        # x_{xijkl0}
                        A[shift_ + self.days * self.shifts * self.teams, i, j, k, :, 0] = 1
                        b[shift_ + self.days * self.shifts * self.teams] = 1

                        shift_ += 1

            return shift_

        def restr_4(shift_):
            # (4): No se puede ir desde una tarea a la misma (menos desde la base)

            for l in range(1, T+1):
                A[shift_, :, :, :, l, l] = 1

            return shift_ + 1

        def restr_5(shift_):
            # (5): Cada turno dura 8 horas
            for i in range(0, self.days):
                for j in range(0, self.shifts):
                    for k in range(0, self.teams):
                        G[shift_, i, j, k] = self.tasks_dists + times[k]
                        h[shift_] = 8
                        shift_ += 1

            return shift_

        def restr_6(shift_):
            # (6): Hay que tener en cuenta que no se hagan ciclos.
            combs = [np.array(tup) for n in range(2, max_tasks_per_day) for tup in combinations(range(1, T + 1), n)]

            for i in range(self.days):
                for j in range(self.shifts):
                    for k in range(self.teams):
                        for S in combs:
                            for l in S:
                                for m in S:
                                    G[shift_, i, j, k, l, m] = 1
                            h[shift_] = len(S) - 1
                            shift_ += 1

            return shift_

        shift = restr_1(0)
        shift = restr_2(shift)
        restr_4(restr_3(shift))
        shift = 0
        shift = restr_5(shift)
        restr_6(shift)

        A = A.reshape((n_b, c.shape[0]))
        # return
        A = matrix(A)
        G = matrix(G.reshape((n_h, c.shape[0])))
        b = matrix(b)
        h = matrix(h)
        c = matrix(c)

        return list(ilp(c=c, A=A, b=b, G=G, h=h, B=set(range(c.size[0])))[1])
