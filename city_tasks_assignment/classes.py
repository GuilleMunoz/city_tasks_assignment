import numpy as np
import networkit as nk
from itertools import combinations
from cvxopt.glpk import ilp
from cvxopt import matrix
from math import comb
from os.path import isfile
from matplotlib import pyplot as plt
from city_tasks_assignment.salib import simulated_annealing as sa

class Problem:

    def __init__(self, file_name=None, dists_given=False):
        # task_loc[0] = location of the base
        self.tasks_loc = np.array([])
        # task_times[0] = 0
        self.tasks_times = np.array([])

        self.tasks_dists = np.array([])

        self.days = 0
        self.shifts = 2
        self.teams = 1

        if file_name is not None:
            self.load(file_name, dists_given=dists_given)

    def compute_dists(self, graph):
        """
        Computes distances using Dijkstra algorithm (from the tasks locations).

        Args:
            graph (networkit.graph.Graph): the graph of the problem
        """
        # import networkit as nk

        spsp = nk.distance.SPSP(graph, self.tasks_loc)
        spsp.run()
        self.tasks_dists = np.array([[ls[task] for task in self.tasks_loc] for ls in spsp.getDistances()])

    def create_random(self, n_tasks, n_teams, n_shifts=2, n_days=-1, n_nodes=-1, n_edges=-1):
        """
        Creates a random problem.

        Args:
            n_tasks (int): Number of tasks (counting with the base = task 0)
            n_teams (int): Number of teams
            n_shifts (int, default): Number of shifts in a day
            n_days (int, default=-1): If not given n_days is a random int between (1 and 20)
            n_nodes (int, default=-1): If n_nodes and n_edges given, creates a graph and uses Problem.computes_dist
            n_edges (int, default=-1): If n_nodes and n_edges given, creates a graph and uses Problem.computes_dist
        """
        from random import randint, sample, uniform
        # import networkit as nk

        self.days = randint(1, 20) if n_days < 0 else n_days
        self.shifts = n_shifts
        self.teams = n_teams
        self.tasks_times = np.random.rand(n_teams, n_tasks)
        self.tasks_times *= 2
        self.tasks_times[:, 0] = 0

        self.tasks_costs = np.random.rand(n_tasks - 1, n_days * n_shifts + 1)
        self.teams_costs = np.random.rand(n_teams, n_days * n_shifts + 1)

        if n_nodes > 0 and n_edges > 0:
            self.tasks_loc = sample(range(n_nodes), n_tasks)
            graph = nk.graph.Graph(n_nodes, weighted=True, directed=True)
            edges = set()
            while len(edges) < n_edges:
                edges.add((randint(0, n_nodes - 1), randint(0, n_nodes - 1)))
            for from_, to in edges:
                graph.addEdge(from_, to, uniform(1, 50))

            self.compute_dists(graph)
        else:
            self.tasks_loc = np.arange(n_tasks)
            self.tasks_dists = np.random.rand(n_tasks, n_tasks)  # # tiempos entre 0 y 1 hora.
            np.fill_diagonal(self.tasks_dists, 0)
            self.tasks_dists *= 2

    def load(self, file_name, dists_given=False):
        """
        Loads a problem from one or two (if not dists_given) files. The first one <file_name>.tasks
        gives general info of the problem number of days, ... and the second contains the graph edges.
        The format is the following:

        *************** "<file_name>.tasks" ***************

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


        % If the distances are given it doesn't load the graph
        *************** "<file_name>.graph" ***************

        <edge1 (ints)> n1 n2 w1
        <edge2 (ints)> n1 n3 w2
        ...
        <edgei (ints)> nj nk wi

        Args:
            file_name (str): the name of the file (without the extension)
            dists_given (bool): True if the distances are given in "<file_name>.tasks"
        """
        import networkit as nk

        with open(file_name + '.tasks', 'r') as tasks_file:

            self.days = int(tasks_file.readline()[:-1])
            self.teams = int(tasks_file.readline()[:-1])
            self.tasks_loc = [int(tasks_file.readline()[:-1])] + list(map(int, tasks_file.readline()[:-1].split(' ')))
            # self.tasks_times = list(map(float, tasks_file.readline()[:-1].split(' ')))

            self.tasks_loc = np.array(self.tasks_loc)
            self.tasks_times = np.zeros((self.teams, self.tasks_loc.shape[0]))

            for i in range(self.teams):
                line = tasks_file.readline()
                times = [0.] + list(map(float, line[:-1].split(' ')))
                self.tasks_times[i] = np.array(times)
                self.tasks_times[i, self.tasks_times[i] < 0] = np.inf

            if dists_given:
                self.tasks_dists = np.empty((self.tasks_times.shape[0], self.tasks_times.shape[0]))
                for i, line in enumerate(tasks_file.readlines()):
                    self.tasks_dists[i] = np.array(map(float, line[:-1].split(' ')))

        if not dists_given and isfile(file_name + '.graph'):
            graph = nk.readGraph(file_name + '.graph', nk.Format.EdgeList, separator=' ', firstNode=0, directed=True)
            self.compute_dists(graph)

    def to_std_ilp(self, ls):
        """
        Transform the solution obtained by ilp_optimize to an standard form ie. list where the i-th element of
        it is the order of tasks the team i will execute. The first element is always 0 (Starts in the base).

        Args:
            ls (list(int)): Solution obtained by ilp_optimize

        Returns:
            (list(int)) The solution in the standard form.
        """

        T = int(self.tasks_dists.shape[0])
        arr = np.array(ls).reshape(self.days, self.shifts, self.teams, T, T)

        std = [list() for i in range(self.teams)]

        for i in range(self.days):
            for j in range(self.shifts):
                for k in range(self.teams):
                    index = 0
                    last = -1
                    ss = set()
                    for _ in range(T):
                        if arr[i, j, k, 0, 0] or index in ss:
                            break
                        std[k].append(index)
                        ss.add(index)
                        last = index
                        index = np.where(arr[i, j, k, index] == 1)[0][0]

                    if last != -1:
                        std[k].append(0)

        return std

    def to_std_sa(self, ls):
        """
        Transform the solution obtained by sa_optimize to an standard form ie. list where the i-th element of
        it is the order of tasks the team i will execute. The first element is always 0 (Starts in the base).

        Args:
            ls (list(int)): Solution obtained by sa_optimize

        Returns:
            (list(int)) The solution in the standard form.
        """

        n_tasks = self.tasks_loc.shape[0] - 1

        std = [list() for i in range(self.teams)]

        for team in range(self.teams):
            from_ = 0
            time = 0
            std[team].append(from_)

            for task in range(n_tasks):
                to = ls[team * n_tasks + task]
                if to < 0:
                    continue
                temp = self.tasks_dists[from_, to] + self.tasks_times[team, to] + self.tasks_dists[to, 0]

                if time + temp > 8:
                    time = self.tasks_dists[0, to] + self.tasks_times[team, to]
                    std[team].append(0)
                else:
                    time += temp - self.tasks_dists[to, 0]

                std[team].append(to)
                from_ = to

            std[team].append(0)

        return std

    def show_sol(self, ls, is_ilp=True):
        """
        Displays the solution (ls) obtained by ilp_optimize if is_ilp or by sa_optimize.

        Args:
            ls (list):
        """
        std = self.to_std_ilp(ls) if is_ilp else self.to_std_sa(ls)

        def to_str(x):
            return '0{}'.format(x) if x < 10 else str(x)

        for team in range(self.teams):
            shift = 0
            from_ = 0
            time = 0
            print(f"Team {team}:")
            print(f"\tDay 0 Shift 0: Base", end="")
            for i, to in enumerate(std[team][1:]):
                time += self.tasks_dists[from_][to] + self.tasks_times[team][to]
                if to == 0:
                    if from_ != 0:
                        print(" -> Base")
                        print("\tTIME: {}:{} horas".format(int(time), to_str(int((time - int(time)) * 60))))
                    else:
                        print()
                    if i < len(std[team]) - 1:
                        print()
                        print(f"\tDay {shift // self.days} Shift {shift % self.shifts}: Base", end="")
                    time = 0
                else:
                    print(f" -> Task {to}", end="")
                from_ = to
            print("\n")

    def ilp_optimize(self):
        """
        Optimizes the given problem using integer linear programming. Works only with one objective programing:
        the time.

        Returns:
            fitness, (np.array)
        """
        # T = number of tasks
        # D = number of days


        # from cvxopt.glpk import ilp
        # from cvxopt import matrix

        T = int(self.tasks_dists.shape[0]) - 1

        times = np.zeros((self.teams, T+1, T+1))
        for team in range(self.teams):
            times[team] = np.repeat(self.tasks_times[team], T+1).reshape((T+1, T+1))

        max_tasks_per_day = int(8 / (np.min(self.tasks_times[:, 1:]) + np.min(self.tasks_dists[1:, 1:])))
        max_tasks_per_day = min(max_tasks_per_day, T) + 1

        c_ = times + self.tasks_dists  # TODO: Añadir tiempos a distancias a latex
        c = np.tile(c_.flatten(), self.days * self.shifts)

        n_b = T  # (1)
        n_b += self.days * self.shifts * self.teams * T  # (2)
        n_b += self.days * self.shifts * self.teams  # (3.1)
        n_b += self.days * self.shifts * self.teams  # (3.2)
        n_b += 1  # (4)
        A = np.zeros((n_b, self.days, self.shifts, self.teams, T+1, T+1))
        b = np.zeros((n_b,))

        n_h = self.days * self.shifts * self.teams  # (4)
        n_h += self.days * self.shifts * self.teams * sum([comb(T+1, i) for i in range(2, max_tasks_per_day)])  # (5)
        G = np.zeros((n_h, self.days, self.shifts, self.teams, T+1, T+1))
        h = np.zeros((n_h,))

        def restr_1(shift_):
            # (1): Se realizan todas las tareas una vez.

            for l in range(1, T+1):
                A[shift_, :, :, :, l, :] = 1
                b[shift_] = 1
                shift_ += 1

            return shift_


        def restr_2(shift_):
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

        def restr_31(shift_):
            # (2): Para cada turno, despues de una tarea siempre va otra

            for i in range(self.days):
                for j in range(self.shifts):
                    for k in range(self.teams):
                        for l in range(1, T+1):
                            A[shift_, i, j, k, l, :] = 1
                            A[shift_, i, j, k, :, l] = -1
                            A[shift_, i, j, k, l, l] = 0

                            shift_ += 1

            return shift_

        def restr_32(shift_):
            # (4): No se puede ir desde una tarea a la misma (menos desde la base)

            for l in range(1, T+1):
                A[shift_, :, :, :, l, l] = 1

            return shift_ + 1

        def restr_4(shift_):
            # (5): Cada turno dura 8 horas
            for i in range(0, self.days):
                for j in range(0, self.shifts):
                    for k in range(0, self.teams):
                        G[shift_, i, j, k] = self.tasks_dists + times[k]
                        h[shift_] = 8
                        shift_ += 1

            return shift_

        def restr_5(shift_):
            # (6): Hay que tener en cuenta que no se hagan ciclos.
            combs = [np.array(tup) for n in range(2, max_tasks_per_day) for tup in combinations(range(1, T+1), n)]

            for i in range(self.days):
                for j in range(self.shifts):
                    for k in range(self.teams):
                        for S in combs:
                            for l in S:
                                G[shift_, i, j, k, l, S] = 1
                            h[shift_] = len(S) - 1
                            shift_ += 1

            return shift_

        # A
        shift = restr_1(0)
        shift = restr_2(shift)
        restr_32(restr_31(shift))

        # G
        shift = restr_4(0)
        restr_5(shift)

        A = matrix(A.reshape((n_b, c.shape[0])))
        G = matrix(G.reshape((n_h, c.shape[0])))

        b = matrix(b)
        h = matrix(h)
        c = matrix(c)

        arr = np.array(ilp(c=c, A=A, b=b, G=G, h=h, B=set(range(c.size[0])))[1])
        return np.sum(np.multiply(arr, c)), arr

    def sa_optimize(self, t, c, Nt, Nc, coef=1.5, rearrange_opt=3, max_space=10, hamming_dist_perc=.5, temp_steps=300,
                    tries_per_temp=10000, ini_tasks_to_rearrange=10, ini_temperature=200, cooling_rate=.9):
        """
        Simulated annealing implementetion for finding a solution. Uses C extension.
        To compile the extension: "python3 salib/setup.py build_ext --inplace"

        Args:
            t (float between (0, 1)): The time coefficient for multiobjective optimization.
            c (float between (0, 1)): The cost coefficient for multiobjective optimization.
            Nt (float): Divider to normalize the time objective.
            Nc (float): Divider to normalize the cost objective.
            coef (float, default=1.5): How bad the solution gets.
            rearrange_opt (int, default=0): if 1 -> opposite
                                       if 2 -> permute
                                       if 3 -> replace
                                       else -> swap
            max_space (int, default=10): max space beetween tasks in swap...
            hamming_dist_perc (float, default=.1): maximum hamming distance accepted for permutation.
            temp_steps (int, default=100): number of temperature steps.
            tries_per_temp (int, default=100000): number of tries per temperature step.
            ini_tasks_to_rearrange (int, default=100): number of tasks to rearrange at first.
            ini_temperature (float, default=20.): initial temperature.
            cooling_rate (float, default=1.5): cooling rate.

        Returns:
            (float, list(int), list(float)): fitness, conf and fitness on each temperature step.

        """

        # Prepare data for the algorithm
        n_tasks = self.tasks_loc.shape[0]

        dists = list(np.reshape(self.tasks_dists, (n_tasks * n_tasks, )))
        times = list(np.reshape(self.tasks_times, (self.teams * n_tasks, )))

        n_tasks -= 1

        tasks_costs = list(np.reshape(self.tasks_costs, (n_tasks * (self.days * self.shifts + 1), )))
        teams_costs = list(np.reshape(self.teams_costs, (self.teams * (self.days * self.shifts + 1),)))

        # Run the simulated annealing algorithm in C
        fitness, conf, ls_fitness = sa.run(t, c, Nt, Nc, self.days, self.shifts, self.teams, n_tasks, times,
                                           dists, tasks_costs, teams_costs, coef, rearrange_opt, max_space,
                                           hamming_dist_perc, temp_steps, tries_per_temp, ini_tasks_to_rearrange,
                                           ini_temperature, cooling_rate)

        return fitness, conf, ls_fitness

    @staticmethod
    def plot_fs_MC(fs, its, hist=True):
        """
        Plots the fitness of the Monte Carlo simulation. If hist plots a histogram else box and whiskers.

        Args:
            fs: fitness of the Monte Carlo simulation.
            its: Number of iterations of the Monte Carlo simulation.
            hist (boolean, default=True):

        """
        if hist:
            m, M = min(fs), max(fs)
            # Plots the fitness
            plt.hist(fs, np.arange(m, M, (M - m) / (its / 5)))  # Divide the interval in its / 5 chunks
        else:
            plt.boxplot(fs, vert=False)
            plt.yticks([])
        # Shows the plot
        plt.show()

    def monte_carlo_simulation(self, fname, var_dists, var_times, var_cost_tasks, var_cost_teams, t, c, Nt, Nc,
                               its=1000, print_conf=True, coef=1.5, rearrange_opt=3, max_space=10, hamming_dist_perc=.5,
                               temp_steps=300, tries_per_temp=10000, ini_tasks_to_rearrange=10, ini_temperature=200,
                               cooling_rate=.9):
        """
        Monte Carlo simulation. Uses C extension. To compile the extension: "python3 salib/setup.py build_ext --inplace"
        Writes every solution (fitness and if print_conf configuration) in a file (fname) and returns the fitness.

        File format:

        fitness_1 time cost
        conf_1 % If print conf
        ...
        fitness_its
        conf_its % If print conf

        Args:
            fname (str): File name to write results to.
            va_dists (list(float)): Variance for the distance between tasks.
            va_times (list(float)): Variance for tasks times.
            var_cost_tasks (list(float)): Variance for tasks costs.
            var_cost_teams (list(float)): Variance for temas costs.
            t (float between (0, 1)): The time coefficient for multiobjective optimization.
            c (float between (0, 1)): The cost coefficient for multiobjective optimization.
            Nt (float): Divider to normalize the time objective.
            Nc (float): Divider to normalize the cost objective.
            its (int, default=1000): Number of iterations for the Monte Carlo simulation.
            print_conf (boolean, default=True): True if write configuration to fname.
            coef (float, default=1.5):
            rearrange_opt (int, default=0): if 1 -> opposite
                                       if 2 -> permute
                                       if 3 -> replace
                                       else -> swap
            max_space (int, default=10): max space beetween tasks in swap
            hamming_dist_perc (float, default=.1): maximum hamming distance accepted for permutation
            temp_steps (int, default=100): number of temperature steps
            tries_per_temp (int, default=100000): number of tries per temperature step
            ini_tasks_to_rearrange (int, default=100): number of tasks to rearrange at first
            ini_temperature (float, default=20.): initial temperature
            cooling_rate (float, default=1.5): cooling rate

        Returns:
            list(double): The fitness of the Monte Carlo simulation

        """
        n_tasks = self.tasks_loc.shape[0]

        # Join mean and variance arrays
        dists = np.reshape(self.tasks_dists, (n_tasks * n_tasks,))
        dists = [var_dists[i // 2] if i % 2 else dists[i // 2] for i in range(2 * dists.shape[0])]

        times = np.reshape(self.tasks_times, (self.teams * n_tasks,))
        times = [var_times[i // 2] if i % 2 else times[i // 2] for i in range(2 * times.shape[0])]

        n_tasks -= 1

        costs_tasks = np.reshape(self.tasks_costs, (n_tasks * (self.days * self.shifts + 1),))
        costs_tasks = [var_cost_tasks[i // 2] if i % 2 else costs_tasks[i // 2]
                       for i in range(2 * costs_tasks.shape[0])]

        costs_teams = np.reshape(self.teams_costs, (self.teams * (self.days * self.shifts + 1),))
        costs_teams = [var_cost_teams[i // 2] if i % 2 else costs_teams[i // 2]
                       for i in range(2 * costs_teams.shape[0])]

        # Run Monte Carlo simulation in C
        sa.run_monte_carlo(fname, its, int(print_conf), t, c, Nt, Nc, self.days, self.shifts, self.teams, n_tasks,
                           times, dists, costs_tasks, costs_teams, coef, rearrange_opt, max_space, hamming_dist_perc,
                           temp_steps, tries_per_temp, ini_tasks_to_rearrange, ini_temperature, cooling_rate)

        fs = []
        # Read results of the Monte Carlo simulation
        with open(fname, 'r') as file:
            fs = [float(line[:-1].split(' ')[0]) for i, line in enumerate(file.readlines()) if i % (print_conf + 1) == 0]

        return fs

    @staticmethod
    def plot_MO(ls, pixels):
        """
        Plots the Monte Carlo simulation for multi-objectives.

        Args:
            ls (list(tuples(float) of len 3)): [(x1, y1, f1, time, cost), ...]
            pixels (int): Number of pixels per dimension

        """

        def mean_bins(xs_, ys_, fs_, bins):

            sum_ = np.zeros((len(bins), len(bins)))
            count = np.zeros(sum_.shape, dtype=int)

            for i in range(len(xs_)):
                for j in range(len(bins)):
                    for k in range(len(bins)):
                        if xs_[i] < bins[j] and ys_[i] < bins[k]:
                            count[j, k] += 1
                            sum_[j, k] += fs_[i]
                            break
                    else:
                        continue
                    break

            count[count == 0] = 1
            return sum_ / count

        xs, ys, _, times, costs = tuple(zip(*ls))

        bins = [i / pixels for i in range(1, pixels + 1)]
        means_t = mean_bins(xs, ys, times, bins)
        means_c = mean_bins(xs, ys, costs, bins)

        ticks = ["{:.2f}".format(i) for i in np.arange(.1, 1.1, .1)]
        
        plt.subplot(1, 2, 1)
        plt.imshow(means_t, interpolation='nearest', cmap=plt.get_cmap('gray'), origin='lower')
        plt.xticks(np.arange(pixels / 10 - .5, pixels, pixels / 10), ticks, rotation=90)
        plt.yticks(np.arange(pixels / 10 - .5, pixels, pixels / 10), ticks)
        plt.title('Tiempos')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(means_c, interpolation='nearest', cmap=plt.get_cmap('gray'), origin='lower')
        plt.xticks(np.arange(pixels / 10 - .5, pixels, pixels / 10), ticks, rotation=90)
        plt.title('Costes')

        plt.colorbar()
        plt.show()

    def monte_carlo_simulation_MO(self, fname, Nt, Nc, its=1000, print_conf=True, coef=1.5, rearrange_opt=3,
                                  max_space=10, hamming_dist_perc=.5, temp_steps=300, tries_per_temp=10000,
                                  ini_tasks_to_rearrange=10, ini_temperature=200, cooling_rate=.9):
        """
        Monte Carlo simulation using simulated annealing.
        The uncertainty factors will be the coefficients of time and cost.
        Writes every solution (fitness and if print_conf configuration) in a file (fname) and returns the fitness.
        Writes every solution (t, c, fitness and if print_conf configuration) in a file (fname)
        
        File format:
        
        t1 c1 f1 time1 cost1
        conf1 %if print_sol
        ...

        Args:
            fname (str): File name to write results to.
            Nt (float): Divider to normalize the time objective.
            Nc (float): Divider to normalize the cost objective.
            its (int, default=1000): Number of iterations for the Monte Carlo simulation.
            print_conf (boolean, default=True): True if write configuration to fname.
            coef (float, default=1.5):
            rearrange_opt (int, default=0): if 1 -> opposite
                                       if 2 -> permute
                                       if 3 -> replace
                                       else -> swap
            max_space (int, default=10): max space beetween tasks in swap
            hamming_dist_perc (float, default=.1): maximum hamming distance accepted for permutation
            temp_steps (int, default=100): number of temperature steps
            tries_per_temp (int, default=100000): number of tries per temperature step
            ini_tasks_to_rearrange (int, default=100): number of tasks to rearrange at first
            ini_temperature (float, default=20.): initial temperature
            cooling_rate (float, default=1.5): cooling rate

        Returns:
            list(double): The fitness of the Monte Carlo simulation

        """
        n_tasks = self.tasks_loc.shape[0]

        dists = list(np.reshape(self.tasks_dists, (n_tasks * n_tasks,)))
        times = list(np.reshape(self.tasks_times, (self.teams * n_tasks,)))

        n_tasks -= 1

        tasks_costs = list(np.reshape(self.tasks_costs, (n_tasks * (self.days * self.shifts + 1),)))
        teams_costs = list(np.reshape(self.teams_costs, (self.teams * (self.days * self.shifts + 1),)))

        # Run Monte Carlo simulation in C
        sa.run_monte_carlo_MO(fname, its, int(print_conf), Nt, Nc, self.days, self.shifts, self.teams, n_tasks,
                              times, dists, tasks_costs, teams_costs, coef, rearrange_opt, max_space, hamming_dist_perc,
                              temp_steps, tries_per_temp, ini_tasks_to_rearrange, ini_temperature, cooling_rate)

        ls = []
        # Read results of the Monte Carlo simulation
        with open(fname, 'r') as file:
            ls = [tuple(map(float, line[:-1].split(' ')))
                  for i, line in enumerate(file.readlines()) if i % (print_conf + 1) == 0]

        return ls
