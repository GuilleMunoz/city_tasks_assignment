# city tasks assignment

## The problem

The problem is to assign some tasks in a city, represented as a graph, to some teams given D days and S shifts. The assigment has to follow some rules:

* Every shift last no more than 8 hours.
* Every shift starts and ends in the base.

## Requirements

You will need the following software to use the package:

- Python 3 (Tested with 3.9)
    - [Development libraries](https://devguide.python.org) for Python3
- [NetworKit](https://networkit.github.io/)    
- [numpy](https://numpy.org)
- [cvxopt](http://cvxopt.org)
- [matplotlib](https://matplotlib.org)

## Software

All the heavy computational tasks are implemented using compiled languages such as C or C++. 

## Usage

To use the package, first, you need to compile the C extension:
 ```bash
 $ python3 city_tasks_assigment/city_tasks_assigment/salib/setup.py build_ext --inplace
 ```

Then use it as a normal package, for example:

```Python
>>> from city_tasks_assignment.classes import Problem
>>> p = Problem()
>>> p.create_random(11, 2, 2, 2)
>>> f, sol, fs = p.sa_optimize(1, 0, 40, 20)
```
