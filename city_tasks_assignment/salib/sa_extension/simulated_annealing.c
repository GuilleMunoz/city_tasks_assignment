#include <Python.h>
#include "../src/simulated_annealing_.h"
#include "../src/norm.h"


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
    int n_days, n_shifts, n_teams, n_tasks, rearrange_opt, max_space, temp_steps, ini_tasks_to_rearrange, steps;
    double coef, ini_temperature, cooling_rate, hamming_dist_perc, tries_per_temp;
    double * arr_fitness;

	PyObject* tasks_times;
    PyObject* tasks_dists;
    PyObject* ret = NULL;
    struct Info info;
    struct Solution sol;

    // Pass args
    if(!PyArg_ParseTuple(args, "iiiiOOdiidididd", &n_days, &n_shifts, &n_teams, &n_tasks, 
                         &tasks_times, &tasks_dists, &coef, 
                         &rearrange_opt, &max_space, &hamming_dist_perc,
                         &temp_steps, &tries_per_temp, &ini_tasks_to_rearrange,
                         &ini_temperature, &cooling_rate)) {
        return NULL;
    }

    // Seed sdtlib random generator with current time
    srand(time(NULL));

    // Initialize info structure 
    info = (struct Info) {n_days, n_shifts, n_teams, n_tasks, max_space, coef, hamming_dist_perc,
                            to_arr(tasks_times, n_teams * (n_tasks + 1)), to_arr(tasks_dists, (n_tasks + 1) * (n_tasks + 1))};

    // Initialize sol structure              
	sol = (struct Solution) {&info, -1, create_conf(n_tasks, n_teams)};

    arr_fitness = run(&sol, rearrange_opt, temp_steps, tries_per_temp, ini_tasks_to_rearrange, ini_temperature,
                        cooling_rate, &steps, "");

    ret = PyTuple_New(3);

    PyTuple_SetItem(ret, 0, PyFloat_FromDouble(sol.fitness));
    PyTuple_SetItem(ret, 1, to_pylist_conf(sol.configuration, n_teams * n_tasks));
    PyTuple_SetItem(ret, 2, to_pylist_fitness(arr_fitness, steps));

    free(info.tasks_times); free(info.tasks_dists); free(sol.configuration); free(arr_fitness);
    return ret;
}


/**
 * 
 * Defines a random array (arr) has to be allocated with length len.
 * The mean and variance of each element is defined in mean_var_arr:
 *      The mean of the i-th element is mean_var_arr[2i]
 *      The variance of the ith element is mean_var_arr[2i + 1]
*/
static void arr_norm(double *mean_var_arr, double *arr, int len)
{

    for (int i = 0; i < len; i++)
    {
        arr[i] = mean_var_arr[2 * i] + norm((double) rand() / RAND_MAX) * mean_var_arr[2*i + 1];
    }
}


/**
 * 
 * Prints a solution (sol) to an opened file pointed by fp.
*/
static void print_sol(FILE *fp, struct Solution *sol)
{
    if (!fp)
    {
        exit(-1);
    }
    
    fprintf(fp, "%f\n", sol->fitness);
    for (int i = 0; i < sol->info->TEAMS * sol->info->TASKS; i++)
    {
        fprintf(fp, "%d ", sol->configuration[i]);
    }
    fprintf(fp, "\n");
}


static PyObject *py_run_montecarlo(PyObject *self, PyObject *args)
{
    int n_days, n_shifts, n_teams, n_tasks, rearrange_opt, max_space, temp_steps, ini_tasks_to_rearrange, steps;
    double coef, ini_temperature, cooling_rate, hamming_dist_perc, tries_per_temp;
    int its;
    FILE *fp;

	PyObject* mean_var_times; double *mean_var_t;
    PyObject* mean_var_dists; double *mean_var_d;
    char *fname;
    char *message;

    struct Info info;
    struct Solution sol;

    // Pass args
    if(!PyArg_ParseTuple(args, "siiiiiOOdiidididd", &fname, &its, &n_days, &n_shifts, &n_teams, &n_tasks, 
                         &mean_var_times, &mean_var_dists, &coef, 
                         &rearrange_opt, &max_space, &hamming_dist_perc,
                         &temp_steps, &tries_per_temp, &ini_tasks_to_rearrange,
                         &ini_temperature, &cooling_rate)) {
        return NULL;
    }
    message = (char*) malloc(((int)log10(its) + 1)*sizeof(char));

    mean_var_t = to_arr(mean_var_times, 2 * n_teams * (n_tasks + 1));
    mean_var_d = to_arr(mean_var_dists, 2 * (n_tasks + 1) * (n_tasks + 1));

    // Seed sdtlib random generator with current time
    srand(time(NULL));

    // Initialize info structure
    info = (struct Info) {n_days, n_shifts, n_teams, n_tasks, max_space, coef, hamming_dist_perc,
                            (double*) malloc(sizeof(double) * n_teams * (n_tasks + 1)),
                            (double*) malloc(sizeof(double) * (n_tasks + 1) * (n_tasks + 1))};

    // Initialize sol structure 
	sol = (struct Solution) {&info, -1, create_conf(n_tasks, n_teams)};

    if (!(fp = fopen(fname, "w")))
    {
        printf("Unable to open file: %s", fname);
        exit(-1);
    }
    for (int i = 0; i < its; i++)
    {
        sprintf(message, "%d: ", i);

        // Defines random times (normal distribution)
        arr_norm(mean_var_t, sol.info->tasks_times, n_teams * (n_tasks + 1));
        arr_norm(mean_var_d, sol.info->tasks_dists, (n_tasks + 1) * (n_tasks + 1));

        // Run simulatted anneling
        run(&sol, rearrange_opt, temp_steps, tries_per_temp, ini_tasks_to_rearrange, ini_temperature, cooling_rate,
            &steps, message);
        // Print the solution
        print_sol(fp, &sol);
    }
    fclose(fp);

    free(info.tasks_times); free(info.tasks_dists); free(sol.configuration); 
    return PyLong_FromLong((long) its);
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

PyDoc_STRVAR(py_run_montecarlo_doc,
                "Montecarlo simulation\n"
                "Writes every solution (fitness and conffiguration) in a file (fname) and the plots a histogram.\n"
                "\nArguments:\n"
                "   fname (str): File name"
                "   its (int): Number of tries"
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
                "   int");

static PyMethodDef SAMethods[] = {
    {"run", py_run, METH_VARARGS, py_run_doc},
    {"run_montecarlo", py_run_montecarlo, METH_VARARGS, py_run_montecarlo_doc},
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
