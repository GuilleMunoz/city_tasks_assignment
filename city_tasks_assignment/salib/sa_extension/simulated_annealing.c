#include <Python.h>
#include "../src/simulated_annealing_.h"


double* to_arr(PyObject *list, int len)
{
    double *arr;
    int i;

    if (!(arr = (double *)malloc(sizeof(double)*len)))
    {
        printf("ERROR: Unable to allocate memory: %lu\n", sizeof(int) * len);
		exit(-1);
    }

    for (i = 0; i < len; i++)
    {
        arr[i] = PyFloat_AsDouble(PyList_GetItem(list, i));
    }
    
    return arr;
}


PyObject* to_pylist_conf(int *arr, int len)
{
    int i;
    PyObject* list = PyList_New(len);
    
    for(i = 0; i < len; i++)
    {
        PyList_SetItem(list, i, PyLong_FromLong((long) arr[i]));
    }

    return list;
}

PyObject* to_pylist_fitness(double *arr, int len)
{
    int i;
    PyObject* list;
    for (i = len - 1; i >= 0; i--)
    {
        if (arr[i] > 0)
            break; 
    }
    len = i + 1;
    list = PyList_New(len);
    for(i = 0; i < len; i++)
    {
        PyList_SetItem(list, i, PyFloat_FromDouble((double) arr[i]));
    }

    return list;
}


static PyObject *py_run(PyObject *self, PyObject *args) 
{
    int n_days, n_shifts, n_teams, n_tasks, rearrange_opt, max_space, temp_steps, ini_tasks_to_rearrange;
    double coef, ini_temperature, cooling_rate, hamming_dist_perc, tries_per_temp;
    double * arr_fitness;

	PyObject* tasks_times;
    PyObject* tasks_dists;
    PyObject* ret = NULL;
    struct Info info;
    struct Solution sol;

    
    if(!PyArg_ParseTuple(args, "iiiiOOdiidididd", &n_days, &n_shifts, &n_teams, &n_tasks, 
                         &tasks_times, &tasks_dists, &coef, 
                         &rearrange_opt, &max_space, &hamming_dist_perc,
                         &temp_steps, &tries_per_temp, &ini_tasks_to_rearrange,
                         &ini_temperature, &cooling_rate)) {
        return NULL;
    }

    info = (struct Info) {n_days, n_shifts, n_teams, n_tasks, max_space, coef, hamming_dist_perc,
                            to_arr(tasks_times, n_teams * (n_tasks + 1)), to_arr(tasks_dists, (n_tasks + 1) * (n_tasks + 1))};

                        
	sol = (struct Solution) {&info, -1, create_conf(n_tasks, n_teams)};

    arr_fitness = run(&sol, rearrange_opt, temp_steps, tries_per_temp, ini_tasks_to_rearrange, ini_temperature, cooling_rate);

    ret = PyTuple_New(3);

    PyTuple_SetItem(ret, 0, PyFloat_FromDouble(sol.fitness));
    PyTuple_SetItem(ret, 1, to_pylist_conf(sol.configuration, n_teams * n_tasks));
    PyTuple_SetItem(ret, 2, to_pylist_fitness(arr_fitness, temp_steps + 1));

    free(info.tasks_times); free(info.tasks_dists); free(sol.configuration); free(arr_fitness);
    return ret;
}


PyDoc_STRVAR(py_run_doc, 
                "Python interface for a C implementation of simulated anneling for city tasks assigment\n"
                "\nArguments:\n"
                "   ndays (int): Number of days\n"
                "   nshifts (int):\n"
                "   nteams (int):\n"
                "   ntasks (int):\n"
                "   tasks_times (1d list(double)): List of times: t_{ij} = time that team i spends on task j t_{i0} = 0\n"
                "   tasks_dists (1d list(double)): List of dists: distance between tasks\n"
                "   coef (double): How\n"
                "   rearrange_opt (int): 1 -> opposite, 2 -> permute, 3 -> replace if other int swap\n"
                "   max_space (int): max space beetween tasks in swap\n"
                "   hamming_dist_perc (double): hamming distance for permutation\n"
                "   temp_steps (int): number of temeprature steps\n"
                "   tries_per_temp (double):"
                "   ini_tasks_to_rearrange (int): number of tasks to rearrange at first\n"
                "   ini_temperature (double): initial temperature\n"
                "   cooling_rate (double):\n"
                "\nReturns:\n"
                "   (int, list): fitness and conf");

static PyMethodDef SAMethods[] = {
    {"run", py_run, METH_VARARGS, py_run_doc},
    {NULL, NULL, 0, NULL} 
};


static struct PyModuleDef SAmodule = {
    PyModuleDef_HEAD_INIT,
    "simulated_annealing",
    "Python interface for a C implementation of simulated anneling for city tasks assigment",
    -1,
    SAMethods
};


PyMODINIT_FUNC PyInit_simulated_annealing(void) {
    return PyModule_Create(&SAmodule);
}
