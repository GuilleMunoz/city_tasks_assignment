#include <Python.h>
#include "../src/simulated_annealing_.h"
#include "../src/norm.h"


/**
 * Transforms a Python list of floats to a C array (of doubles)
 * 
 * @param list PyObject, Python list
 * @param len int, length of the list
 * @return The array
 */
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


/**
 * Transforms a array of ints to a Python list of ints (longs)
 * 
 * @param arr pointer to an array of int, The array to transform.
 * @param len int, length of the array.
 * @return PyObject, the Python list.
 */
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


/**
 * Transforms a array of doubles to a Python list of floats
 * 
 * @param arr pointer to an array of double, The array to transform.
 * @param len int, length of the array.
 * @return PyObject, the Python list.
 */
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


PyDoc_STRVAR(py_run_doc, 
                "Python interface for a C implementation of simulated annealing for city tasks assigment\n"

                "\nArguments:\n"
                "   t (float): Coefficient of the time\n"
                "   c (float): Coefficient of the cost\n"
                "   Nt (float): How to normalize the time\n"
                "   Nc (float): How to normalize the cost\n"
                "   ndays (int): Number of days\n"
                "   nshifts (int):\n"
                "   nteams (int):\n"
                "   ntasks (int): Number of tasks (WITHOUT THE BASE)\n"
                "   tasks_times (1d list(double)): List of times: t_{ij} = time that team i spends on task j t_{i0} = 0\n"
                "   tasks_dists (1d list(double)): List of dists: distance between tasks\n"
                "   task_cost (list(double)): task_cost[i][j] cost of doing task i on shift j,  0 <= i < TASKS, 0 <= j <= DAYS * SHIFTS\n"
                "   team_cost (list(double): team_cost[i]([j]) team_cost[i] cost of hiring team i (on shift j), 0 <= i < TEAMS (0 <= j <= DAYS * SHIFTS)\n"
                "   coef (double): Coefficient that will be multiplied to the shift time if the team has to work more\n"
                "   rearrange_opt (int): 1 -> opposite,\n"
                "                        2 -> permute,\n"
                "                        3 -> replace,\n"
                "                        4 -> replace or permute,\n"
                "                        if other int swap\n"
                "   max_space (int): max space beetween tasks in swap\n"
                "   hamming_dist_perc (double): hamming distance for permutation\n"
                "   temp_steps (int): number of temeprature steps\n"
                "   tries_per_temp (double):"
                "   ini_tasks_to_rearrange (int): number of tasks to rearrange at first\n"
                "   ini_temperature (double): initial temperature\n"
                "   cooling_rate (double):\n"
                "\nReturns:\n"
                "   (int, list, list): fitness, configuration and fitness on each temperatur steps");

/**
 * Extension to run method of simulated_anneling_. See py_run_doc.
 * 
 */
static PyObject *py_run(PyObject *self, PyObject *args) 
{
    int n_days, n_shifts, n_teams, n_tasks, rearrange_opt, max_space, temp_steps, ini_tasks_to_rearrange, steps;
    double coef, ini_temperature, cooling_rate, hamming_dist_perc, tries_per_temp;
    double * arr_fitness;

    double t, c, Nt, Nc;

	PyObject* tasks_times;
    PyObject* tasks_dists;
    PyObject* tasks_costs;
    PyObject* teams_costs;

    PyObject* ret = NULL;
    struct Info info;
    struct Solution sol;

    // Pass args
    if(!PyArg_ParseTuple(args, "ddddiiiiOOOOdiidididd", &t, &c, &Nt, &Nc, 
                            &n_days, &n_shifts, &n_teams, &n_tasks, 
                            &tasks_times, &tasks_dists, &tasks_costs, &teams_costs,
                            &coef, &rearrange_opt, &max_space, &hamming_dist_perc,
                            &temp_steps, &tries_per_temp, &ini_tasks_to_rearrange,
                            &ini_temperature, &cooling_rate))
    {
        return NULL;
    }

    // Seed sdtlib random generator with current time
    srand(time(NULL));

    // Initialize info structure 
    info = (struct Info) {n_days, n_shifts, n_teams, n_tasks, max_space, coef, hamming_dist_perc,
                            to_arr(tasks_times, n_teams * (n_tasks + 1)), 
                            to_arr(tasks_dists, (n_tasks + 1) * (n_tasks + 1)),
                            t, c, Nt, Nc,
                            to_arr(tasks_costs, n_tasks * (n_days * n_shifts + 1)), 
                            to_arr(teams_costs, n_teams * (n_days * n_shifts + 1))};

    // Initialize sol structure              
	sol = (struct Solution) {&info, -1, -1, -1, create_conf(n_tasks, n_teams)};

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
 * 
 * @param mean_var_arr pointer to array of doubles, the mean and variance. The length has to be >= 2 * len.
 * @param arr pointer to array of doubles, the random array
 * @param len int, length of arr
*/
static void arr_norm(double *mean_var_arr, double *arr, int len)
{

    for (int i = 0; i < len; i++)
    {
        arr[i] = mean_var_arr[2 * i] + norm((double) rand() / RAND_MAX) * mean_var_arr[2*i + 1];
    }
}


/**
 * Prints a solution (sol) to an opened file pointed by fp.
 * If fp is null exist with -1.
 * 
 * @param fp pointer to a FILE, file to print
 * @param sol pointer to a struct Solution, the solution to be printed
 * @param print_conf int, 1 if print configuration to fp, 0 otherwise
*/
static void print_sol(FILE *fp, struct Solution *sol, int print_conf)
{
    if (!fp)
    {
        exit(-1);
    }
    
    fprintf(fp, "%f %f %f\n", sol->fitness, sol->time, sol->cost);
    if (print_conf)
    {
        for (int i = 0; i < sol->info->TEAMS * sol->info->TASKS; i++)
        {
            fprintf(fp, "%d ", sol->configuration[i]);
        }
        fprintf(fp, "\n");
    }
}


PyDoc_STRVAR(py_run_monte_carlo_doc,
                "Monte Carlo simulation using simulated annealing.\n"
                "The uncertainty factors will be the times between tasks and the tasks times.\n"
                "Writes every solution (fitness and if print_conf configuration) in a file (fname)\n"
                "\nArguments:\n"
                "   fname (str): File name to write each solution.\n"
                "   its (int): Number of tries.\n"
                "   print_conf (int):  1 if print configuration to fname, 0 otherwise\n"
                "   t (float): Coefficient of the time\n"
                "   c (float): Coefficient of the cost\n"
                "   Nt (float): How to normalize the time\n"
                "   Nc (float): How to normalize the cost\n"
                "   ndays (int): Number of days.\n"
                "   nshifts (int): Number of shifts.\n"
                "   nteams (int): Number of teams.\n"
                "   ntasks (int): Number of tasks (WITHOUT THE BASE).\n"
                "   mean_var_times (1d list(double)): List of times of length 2 * nteams * (ntasks + 1)\n"
                "   mean_var_dists (1d list(double)): List of dists of length 2 * (ntasks + 1)^2\n"
                "   mean_var_tasks_costs (1d list(double)): List of dists of length ntasks * (ndays * nshifts + 1)\n"
                "   mean_var_teams_costs (1d list(double)): List of dists of length nteams * (ndays * nshifts + 1)\n"
                "   coef (double): Coefficient that will be multiplied to the shift time if the team has to work more\n"
                "   rearrange_opt (int): 1 -> opposite,\n"
                "                        2 -> permute,\n"
                "                        3 -> replace,\n"
                "                        4 -> replace or permute,\n"
                "                        if other int swap\n"
                "   max_space (int): max space beetween tasks in swap\n"
                "   hamming_dist_perc (double): hamming distance for permutation\n"
                "   temp_steps (int): number of temeprature steps\n"
                "   tries_per_temp (double):\n"
                "   ini_tasks_to_rearrange (int): number of tasks to rearrange at first\n"
                "   ini_temperature (double): initial temperature\n"
                "   cooling_rate (double):\n");

/**
 * Runs a Monte Carlo simulation of the problem. See py_run_monte_carlo_doc for doc.
 * 
 */
static PyObject *py_run_monte_carlo(PyObject *self, PyObject *args)
{
    int n_days, n_shifts, n_teams, n_tasks, rearrange_opt, max_space, temp_steps, ini_tasks_to_rearrange, steps;
    double coef, ini_temperature, cooling_rate, hamming_dist_perc, tries_per_temp;
    int its, print_conf;
    FILE *fp;

	PyObject* mean_var_times; double *mean_var_t;
    PyObject* mean_var_dists; double *mean_var_d;
    PyObject* mean_var_task_cost; double *mean_var_task_c;
    PyObject* mean_var_team_cost; double *mean_var_team_c;
    char *fname;
    char *message;

    double t, c, Nt, Nc;

    struct Info info;
    struct Solution sol;

    // Pass args
    if(!PyArg_ParseTuple(args, "siiddddiiiiOOOOdiidididd", &fname, &its, &print_conf, 
                         &t, &c, &Nt, &Nc, &n_days, &n_shifts, &n_teams, 
                         &n_tasks, &mean_var_times, &mean_var_dists, 
                         &mean_var_task_cost, &mean_var_team_cost, &coef, 
                         &rearrange_opt, &max_space, &hamming_dist_perc,
                         &temp_steps, &tries_per_temp, &ini_tasks_to_rearrange,
                         &ini_temperature, &cooling_rate))
    {
        return NULL;
    }
    message = (char*) malloc(((int)log10(its) + 1)*sizeof(char));

    mean_var_t = to_arr(mean_var_times, 2 * n_teams * (n_tasks + 1));
    mean_var_d = to_arr(mean_var_dists, 2 * (n_tasks + 1) * (n_tasks + 1));
    mean_var_task_c = to_arr(mean_var_task_cost, 2 * n_tasks * (n_days * n_shifts + 1));
    mean_var_team_c = to_arr(mean_var_task_cost, 2 * n_teams * (n_days * n_shifts + 1));

    // Seed sdtlib random generator with current time
    srand(time(NULL));

    // Initialize info structure
    info = (struct Info) {n_days, n_shifts, n_teams, n_tasks, max_space, coef, hamming_dist_perc,
                            (double*) malloc(sizeof(double) * n_teams * (n_tasks + 1)),
                            (double*) malloc(sizeof(double) * (n_tasks + 1) * (n_tasks + 1)),
                            t, c, Nt, Nc, 
                            (double*) malloc(sizeof(double) * n_tasks * (n_days * n_shifts + 1)), 
                            (double*) malloc(sizeof(double) * n_teams * (n_days * n_shifts + 1))};

    // Initialize sol structure 
	sol = (struct Solution) {&info, -1, -1, -1, create_conf(n_tasks, n_teams)};

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
        arr_norm(mean_var_task_c, sol.info->task_cost, n_tasks * (n_days * n_shifts + 1));
        arr_norm(mean_var_team_c, sol.info->team_cost, n_teams * (n_days * n_shifts + 1));

        // Run simulatted annealing
        run(&sol, rearrange_opt, temp_steps, tries_per_temp, ini_tasks_to_rearrange, ini_temperature, cooling_rate,
            &steps, message);
        // Print the solution
        print_sol(fp, &sol, print_conf);
    }
    fclose(fp);

    free(info.tasks_times); free(info.tasks_dists); free(info.task_cost); free(info.team_cost);
    free(sol.configuration); 
    free(mean_var_t); free(mean_var_d); free(mean_var_task_c); free(mean_var_team_c);
    return Py_BuildValue("");
}


PyDoc_STRVAR(py_run_monte_carlo_MO_doc, 
                "Monte Carlo simulation using simulated annealing.\n"
                "The uncertainty factors will be the coefficients of time and cost.\n"
                "Writes every solution (t, c, fitness and if print_conf configuration) in a file (fname)\n"
                "File format:\n"
                "t c fitness time cost\n"
                "conf %if print_sol\n"
                "...\n"
                "\nArguments:\n"
                "   fname (str): File name to write each solution.\n"
                "   its (int): Number of tries.\n"
                "   print_conf (int):  1 if print configuration to fname, 0 otherwise\n"
                "   Nt (float): How to normalize the time\n"
                "   Nc (float): How to normalize the cost\n"
                "   ndays (int): Number of days\n"
                "   nshifts (int):\n"
                "   nteams (int):\n"
                "   ntasks (int): Number of tasks (WITHOUT THE BASE)\n"
                "   tasks_times (1d list(double)): List of times: t_{ij} = time that team i spends on task j t_{i0} = 0\n"
                "   tasks_dists (1d list(double)): List of dists: distance between tasks\n"
                "   task_cost (list(double)): task_cost[i][j] cost of doing task i on shift j,  0 <= i < TASKS, 0 <= j <= DAYS * SHIFTS\n"
                "   team_cost (list(double): team_cost[i]([j]) team_cost[i] cost of hiring team i (on shift j), 0 <= i < TEAMS (0 <= j <= DAYS * SHIFTS)\n"
                "   coef (double): Coefficient that will be multiplied to the shift time if the team has to work more\n"
                "   rearrange_opt (int): 1 -> opposite,\n"
                "                        2 -> permute,\n"
                "                        3 -> replace,\n"
                "                        4 -> replace or permute,\n"
                "                        if other int swap\n"
                "   max_space (int): max space beetween tasks in swap\n"
                "   hamming_dist_perc (double): hamming distance for permutation\n"
                "   temp_steps (int): number of temeprature steps\n"
                "   tries_per_temp (double):"
                "   ini_tasks_to_rearrange (int): number of tasks to rearrange at first\n"
                "   ini_temperature (double): initial temperature\n"
                "   cooling_rate (double):\n");


/**
 * Runs a Monte Carlo simulation of the problem for multiobjective. 
 * See py_run_monte_carlo_MO_doc for doc.
 * 
 */
static PyObject *py_run_monte_carlo_MO(PyObject *self, PyObject *args)
{
    int n_days, n_shifts, n_teams, n_tasks, rearrange_opt, max_space, temp_steps, ini_tasks_to_rearrange, steps;
    double coef, ini_temperature, cooling_rate, hamming_dist_perc, tries_per_temp;
    int its, print_conf;
    FILE *fp;
    double t, c, Nt, Nc;

	PyObject* tasks_times;
    PyObject* tasks_dists;
    PyObject* tasks_cost;
    PyObject* teams_cost;

    char *fname;
    char *message;

    struct Info info;
    struct Solution sol;

    // Pass args
    if(!PyArg_ParseTuple(args, "siiddiiiiOOOOdiidididd", &fname, &its, &print_conf,
                            &Nt, &Nc, &n_days, &n_shifts, &n_teams, &n_tasks, 
                            &tasks_times, &tasks_dists, &tasks_cost, &teams_cost,
                            &coef, &rearrange_opt, &max_space, &hamming_dist_perc,
                            &temp_steps, &tries_per_temp, &ini_tasks_to_rearrange,
                            &ini_temperature, &cooling_rate))
    {
        return NULL;
    }

    // Seed sdtlib random generator with current time
    srand(time(NULL));

    // Initialize info structure 
    info = (struct Info) {n_days, n_shifts, n_teams, n_tasks, max_space, coef, hamming_dist_perc,
                            to_arr(tasks_times, n_teams * (n_tasks + 1)), 
                            to_arr(tasks_dists, (n_tasks + 1) * (n_tasks + 1)),
                            0, 0, Nt, Nc, 
                            to_arr(tasks_cost, n_tasks * (n_days * n_shifts + 1)), 
                            to_arr(teams_cost, n_teams * (n_days * n_shifts + 1))};

    // Initialize sol structure              
	sol = (struct Solution) {&info, -1, -1, -1, create_conf(n_tasks, n_teams)};

    message = (char*) malloc(((int)log10(its) + 1)*sizeof(char));

    if (!(fp = fopen(fname, "w")))
    {
        printf("Unable to open file: %s", fname);
        exit(-1);
    }
    for (int i = 0; i < its; i++)
    {
        sprintf(message, "%d: ", i);

        t = (double) rand() / RAND_MAX;
        c = (double) rand() / RAND_MAX;
        fprintf(fp, "%f %f ", t, c);
        sol.info->t = t;
        sol.info->c = c;

        // Run simulatted annealing
        run(&sol, rearrange_opt, temp_steps, tries_per_temp, ini_tasks_to_rearrange, ini_temperature, cooling_rate,
            &steps, message);
        // Print the solution
        print_sol(fp, &sol, print_conf);
    }
    fclose(fp);

    free(info.tasks_times); free(info.tasks_dists); free(info.task_cost); free(info.team_cost);
    free(sol.configuration);
    return Py_BuildValue("");
}



static PyMethodDef SAMethods[] = {
    {"run", py_run, METH_VARARGS, py_run_doc},
    {"run_monte_carlo", py_run_monte_carlo, METH_VARARGS, py_run_monte_carlo_doc},
    {"run_monte_carlo_MO", py_run_monte_carlo_MO, METH_VARARGS, py_run_monte_carlo_MO_doc},
    {NULL, NULL, 0, NULL} 
};


static struct PyModuleDef SAmodule = {
    PyModuleDef_HEAD_INIT,
    "simulated_annealing",
    "Python interface for a C implementation of simulated annealing for city tasks assigment",
    -1,
    SAMethods
};


PyMODINIT_FUNC PyInit_simulated_annealing(void) {
    return PyModule_Create(&SAmodule);
}
