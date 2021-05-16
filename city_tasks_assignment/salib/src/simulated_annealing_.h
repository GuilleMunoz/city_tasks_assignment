#ifndef HEURISTICS_SA_H
#define HEURISTICS_SA_H

struct Info
{
	int DAYS, SHIFTS, TEAMS, TASKS;
	int max_space;
	double coef, hamming_dist_perc;
	double* tasks_times; // [TEAMS][TASKS + 1]  = [:][BASE, TASK1, ...]
	double* tasks_dists; // [TASKS + 1][TASKS + 1] = [BASE, TASK1, ...][BASE, TASK1, ...]

	double t, c, Nt, Nc; // coefficients for multiobjective

	double* tasks_costs; // tasks_costs[i][j] cost of doing task i on shift j,  0 <= i < TASKS, 0 <= j <= DAYS * SHIFTS

	double* teams_costs; // teams_costs[i]([j]) teams_costs[i] cost of hiring team i (on shift j), 0 <= i < TEAMS (0 <= j <= DAYS * SHIFTS)
};

struct Solution
{
	struct Info* info; // Information about the problem
	double fitness; // Fitness of the solution
	double time; // Time of the solution (not normalized)
	double cost; // Cost of the solution (not normalized)
	int* configuration;  // Configuration of the solution: -1 or n where 1 <= n <= T
};

int * create_conf(int n_tasks, int n_teams);
double * run(struct Solution *sol, int rearrange_opt, int temp_steps, double tries_per_temp,
                int ini_tasks_to_rearrange, double ini_temperature, double cooling_rate,
                int *steps, char *message);

#endif // HEURISTICS_SA_H