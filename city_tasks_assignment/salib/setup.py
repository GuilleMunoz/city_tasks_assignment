from distutils.core import setup, Extension

def main():
    setup(name="simulated_annealing",
          version="1.0.0",
          description="Python interface for a C implementation of simulated anneling for city tasks assigment",
          author="Guillermo Muñoz",
          ext_modules=[Extension("simulated_annealing",
                                 ["sa_extension/simulated_annealing.c", "src/simulated_annealing_.c"],
                                 include_dirs=["src"])]
        )
#python3 setup.py install 
if __name__ == "__main__":
    main()