from classes import Problem
import numpy as np

if __name__ == '__main__':

    N_TASKS = 4
    DAYS = 2
    SHIFTS = 2

    N_TASKS += 1  # Base

    problem = Problem()

    problem.days = DAYS
    problem.shifts = SHIFTS

    tasks_times = np.random.rand(N_TASKS)  # tiempos entre 0 y 1 horas.
    tasks_times[0] = 0  # La tarea 0 es la base se puede poner un tiempo (tiempo de preparar el equipo p.j.)
    problem.tasks_times = tasks_times

    problem.tasks_loc = np.arange((N_TASKS, ))

    tasks_dists = np.random.rand(N_TASKS, N_TASKS)  # # tiempos entre 0 y 1 horas.
    np.fill_diagonal(tasks_dists, 0)
    problem.tasks_dists = tasks_dists

    problem.show_sol(problem.optimize())
