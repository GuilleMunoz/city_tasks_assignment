#include "simulated_annealing_.h"
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>



static void swap(int *a, int *b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}


int* create_conf(int n_tasks, int n_teams)
{	
	int i;
	int *configuration;

	srand(time(NULL));

	if (!(configuration = (int *)malloc(sizeof(int) * n_tasks * n_teams)))
	{
		printf("ERROR: Unable to allocate memory: %lu\n", sizeof(int) * n_tasks * n_teams);
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


static double evaluate(struct Solution* sol)
{	
	double total_time = 0, time, temp, coef;
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

			from = to;
		}
		if (from != 0)
		{
			time += sol->info->tasks_dists[from * offset];
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
	sol->fitness = total_time;
	return total_time;
}


static void swap_cities(struct Solution *sol, int n)
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


static void replace(struct Solution* sol, int n)
{
	int i, start, new_pos, len = sol->info->TEAMS * sol->info->TASKS;
	int *temp;

	if (!(temp = (int *) malloc(sizeof(int) * n)))
	{
		printf("ERROR: Unable to allocate memory: %lu\n", sizeof(int) * n);
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

double * run(struct Solution *sol, int rearrange_opt, int temp_steps, double tries_per_temp,
					 int ini_tasks_to_rearrange, double ini_temperature, double cooling_rate, int* steps)
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
	double *arr_fitness = (double *) malloc(sizeof(double) * (temp_steps + 1));
	int succ_per_temp, i;

	srand(time(NULL));

	if (!(temp_conf = (int *)malloc(sizeof(int) * conf_len)))
	{
		printf("ERROR: Unable to allocate memory: %lu\n", sizeof(int) * conf_len);
		exit(-1);
	}

	switch (rearrange_opt)
	{
	case 0:
		rearrange = &swap_cities;
		break;
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
		rearrange = &swap_cities;
	}

	fitness = evaluate(sol);
	memcpy(temp_conf, sol->configuration, sizeof(int) * conf_len);

	arr_fitness[0] = fitness;
	for (i = 0; i < temp_steps; i++)
	{	
		printf("\r%c Searching for a solution: %.2d%%", chrs[i%8], (100 * i / temp_steps));
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
