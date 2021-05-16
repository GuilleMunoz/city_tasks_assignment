#include "simulated_annealing_.h"
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>


/**
 * Swaps to integers
 * 
 * @param a pointer to an integer
 * @param b pointer to an integer
 */ 
static void swap(int *a, int *b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

/**
 * Creates a random configuration for the problem as a continous array.
 * 
 * @param n_tasks number of tasks of the problem
 * @param n_teams number of teams of the problem
 * @return array of length n_tasks * n_teams of integers, the configuration
 */
int* create_conf(int n_tasks, int n_teams)
{	
	int i;
	int *configuration;

	if (!(configuration = (int *)malloc(sizeof(int) * n_tasks * n_teams)))
	{
		printf("CREATE CONF: ERROR, unable to allocate memory: %lu\n", sizeof(int) * n_tasks * n_teams);
		exit(-1);
	}

	for (i = 0; i < n_tasks; i++)
		configuration[i] = i + 1;

	for (i = n_tasks; i < n_teams * n_tasks; i++)
		configuration[i] = -1;
	
	for(i = n_teams * n_tasks - 1; i>0; i--)
		swap(&configuration[i], &configuration[rand() % (i+1)]);

	return configuration;
}


/**
 * Evaluates a given solution. 
 * Sets the fitness of the solution to the correct fitness and also returns de fitness.
 * 
 * @param sol pointer to a struct Solution
 * @return double, the fitness
 */
static double evaluate(struct Solution* sol)
{	
	double total_time = 0., time, temp, coef;
	double cost = 0.;
	int team, j, from, to, shifts, offset = sol->info->TASKS + 1;

	for(team = 0; team < sol->info->TEAMS; team++)
	{
		time = 0;
		from = 0;
		shifts = 1;
		coef = sol->info->coef;

		for (j = 0; j < sol->info->TASKS; ++j)
		{
			to = sol->configuration[team * sol->info->TASKS + j];
			if (to < 0){continue;}
			
			temp = sol->info->tasks_dists[from * offset + to] + 
					sol->info->tasks_times[team * offset + to] +
					sol->info->tasks_dists[to * offset];

			if (time + temp > 8)
			{
				time += sol->info->tasks_dists[from * offset];
				cost += sol->info->teams_costs[team * (sol->info->DAYS * sol->info->SHIFTS + 1) + shifts];

				if (shifts > sol->info->DAYS * sol->info->SHIFTS)
				{
					total_time += coef * time;
					coef += 1;
				}
				else
				{
					total_time += time;
					shifts++;
				}
				time = sol->info->tasks_dists[to] + sol->info->tasks_times[team * offset + to];
			}
			else
			{
				time += temp - sol->info->tasks_dists[to * offset];
			}
			cost += sol->info->tasks_costs[(to - 1) * sol->info->DAYS * sol->info->SHIFTS + shifts];

			from = to;
		}
		if (from != 0)
		{
			time += sol->info->tasks_dists[from * offset];

			cost += sol->info->teams_costs[team * (sol->info->DAYS * sol->info->SHIFTS + 1) + shifts];
			if (shifts > sol->info->DAYS * sol->info->SHIFTS)
			{
				total_time += coef * time;
			}
			else
			{
				total_time += time;
			}
		}
		
	}

	sol->time = total_time;
	sol->cost = cost;
	sol->fitness = sol->info->t * sol->time / sol->info->Nt + sol->info->c * sol->cost / sol->info->Nc;
	return sol->fitness;
}


/**
 * Swaps n tasks with a maximum distance between them.
 * 
 * @param sol pointer to a struct Solution 
 * @param n int, number of tasks to swap
 */
static void swap_tasks(struct Solution *sol, int n)
{
	int j, offset, len = sol->info->TEAMS * sol->info->TASKS;
	//srand(time(NULL));

	for (int i = 0; i < n; i++)
	{
		j = rand() % len;
		offset = rand() % (2 * sol->info->max_space) - sol->info->max_space;
		if (0 > j + offset)
			offset = -j;
		if (j + offset >= len)
			offset = len - j - 1;
		
		swap(&sol->configuration[j], &sol->configuration[j + offset]);
	}
}


/**
 * Puts a section of the configuration in the opposite order.
 * 
 * @param sol pointer to a struct Solution
 * @param n int, length of the section
 */
static void opposite_order(struct Solution *sol, int n)
{
	int i, start, len = sol->info->TEAMS * sol->info->TASKS;
	//srand(time(NULL));
	start = rand() % (len - n);

	for (i = 0; i < n / 2; i++)
	{
		swap(&sol->configuration[start + i], &sol->configuration[start + n - i - 1]);
	}
}


/**
 * Permutes a section of the configuration. The maximum Hamming distance of the resulting permutation is
 * given in by sol->info->hamming_dist_perc.
 * 
 * @param sol pointer to a struct Solution
 * @param n int, length of the section
 */
static void permute(struct Solution *sol, int n)
{
	int a, b, hamming_dist, start, len = sol->info->TEAMS * sol->info->TASKS;

	//srand(time(NULL));
	start = rand() % (len - n);

	hamming_dist = round(n * sol->info->hamming_dist_perc);
	if (hamming_dist <= 0)
		hamming_dist = 1;

	for (int i = 0; i < hamming_dist; i++)
	{
		a = rand() % n;
		b = rand() % (n - 1);
		if (a <= b)
			b += 1;
		swap(&sol->configuration[start + a], &sol->configuration[start + b]);
	}
	
}


/**
 * Removes a section of the configuration and places in another randomly slected part of the configuration.
 * 
 * @param sol pointer to a struct Solution
 * @param n int, length of the section
 */
static void replace(struct Solution* sol, int n)
{
	int i, start, new_pos, len = sol->info->TEAMS * sol->info->TASKS;
	int *temp;

	if (!(temp = (int *) malloc(sizeof(int) * n)))
	{
		printf("SA REPLACE: ERROR, unable to allocate memory: %lu\n", sizeof(int) * n);
		exit(-1);
	}

	start = rand() % (len - n);
	new_pos = rand() % (len - n - 1);
	
	if (start <= new_pos)
		new_pos += 1;
	
	for (i = 0; i < n; i++)
		temp[i] = sol->configuration[start + i];

	if (start <= new_pos)
	{
		for(i = start; i < new_pos; i++)
			sol->configuration[i] = sol->configuration[i + n];
	}
	else
	{
		for(i = start + n - 1; i >= new_pos + n; i--)
			sol->configuration[i] = sol->configuration[i - n];
	}
	
	for(i = 0; i < n; i++)
		sol->configuration[new_pos + i] = temp[i];

	free(temp);
}


/**
 * Throws a coin and decides whether to do the replace method or the permute one 
 * (the two most effectives rearrangements).
 * 
 * @param sol pointer to a struct Solution
 * @param n int, length of the section
 */
static void remute(struct Solution *sol, int n)
{
	if (rand()/RAND_MAX < .5)
	{
		replace(sol, n);
	}
	else
	{
		permute(sol, n);
	}
}


/**
 * Runs the simulated annealing algorithm.
 * 
 * @param sol pointer to a struct Solution, everything has to be initialize .
 * @param rearrange_opt int, the rearrange method that will be used:
* 								1 -> opposite_order
* 								2 -> permute
* 								3 -> replace
* 								4 -> replace or permute
* 								other int -> swap_tasks
 * @param temp_steps int, number of temperature steps ie. number of times the initial temperature will be 
 * 						  multiplied by the cooling rate.
 * @param tries_per_temp int, number of rearrangements to try for every temperature.
 * @param ini_tasks_to_rearrange int, number of tasks to rearrange. If less or equal to 0, the number of tasks 
 * 									  to rearrange will be 4.
 * @param ini_temperature double, initial temperature.
 * @param cooling_rate double, cooling rate (<1).
 * @param steps pointer to an integer, at the end the integer will be the number of temperature steps in made.
 * @param message string, message to print on each step.
 * @return array of double of size temp_steps, the final fitness on all the steps. If the algorithm finishes early,
 * 		   the last part of the array is meaningless. 
 */
double * run(struct Solution *sol, int rearrange_opt, int temp_steps, double tries_per_temp,
					 int ini_tasks_to_rearrange, double ini_temperature, double cooling_rate,
					 int* steps, char* message)
{

	void (*rearrange)(struct Solution *, int);  // Rearrange function that will be used
	int *temp_conf = NULL;

	char *chrs = "/-\\|/-\\|";
	int conf_len = sol->info->TEAMS * sol->info->TASKS;
	int tasks_to_rearrange = ini_tasks_to_rearrange >= conf_len ? ini_tasks_to_rearrange : round(conf_len / 10);
	if (tasks_to_rearrange <= 0)
	{
		tasks_to_rearrange = 4;
	}
	
	double fitness, diff, temperature = ini_temperature, max_succ_per_temp = tries_per_temp /10;
	double *arr_fitness;
	int succ_per_temp, i;

	//srand(time(NULL));
	if (!(arr_fitness = (double *) malloc(sizeof(double) * (temp_steps + 1))))
	{
		printf("SA RUN: ERROR, unable to allocate memory: %lu\n", sizeof(double) * (temp_steps + 1));
		exit(-1);
	}
	
	if (!(temp_conf = (int *)malloc(sizeof(int) * conf_len)))
	{
		printf("SA RUN: ERROR, unable to allocate memory: %lu\n", sizeof(int) * conf_len);
		exit(-1);
	}

	switch (rearrange_opt)
	{
	case 1:
		rearrange = &opposite_order;
		break;
	case 2:
		rearrange = &permute;
		break;
	case 3:
		rearrange = &replace;
		break;
	case 4:
		rearrange = &remute;
		break;
	default:
		rearrange = &swap_tasks;
	}

	fitness = evaluate(sol);
	memcpy(temp_conf, sol->configuration, sizeof(int) * conf_len);

	arr_fitness[0] = fitness;
	for (i = 0; i < temp_steps; i++)
	{	
		printf("\r%s%c Searching for a solution: %.2d%%",message, chrs[i%8], (100 * i / temp_steps));
		fflush(stdout);

		succ_per_temp = 0;

		for (int j = 0; j < tries_per_temp; j++)
		{	
			rearrange(sol, tasks_to_rearrange);
			evaluate(sol); 
			diff = fitness - sol->fitness;
			if (0 < diff || rand()/RAND_MAX < exp(diff / temperature)) // if better score or accept (metrop)
			{ 
				memcpy(temp_conf, sol->configuration, sizeof(int) * conf_len);
				fitness = sol->fitness;
				succ_per_temp++;
			}else
			{
				memcpy(sol->configuration, temp_conf, sizeof(int) * conf_len);
			}

			if (succ_per_temp > max_succ_per_temp)
				break;
		}
		arr_fitness[i + 1] = fitness;
		if (succ_per_temp == 0) {
		    *steps = i;
		    break; // No succes
		}
		temperature *= cooling_rate;
	}
	evaluate(sol);
	printf("\r                                     \r");
	fflush(stdout);

	free(temp_conf);
	*steps = i;
	return arr_fitness;
}
