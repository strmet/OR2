import cplex
from cplex.callbacks import LazyConstraintCallback
import time
import multiprocessing


class LazyCallback(LazyConstraintCallback):


    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)

    def __call__(self):

        # Set sequential code
        self.model.parameters.threads.set(1)

        start = time.time()

        if self.sum_time > self.inital_wait_time and self.initial_wait_time is not None:
            # Get solution to build cuts
            sol = [self.EdgeSol(self.ypos(i, j), i, j)
                       for i in range(self.n_nodes)
                       for j in range(self.n_nodes)
                       if self.get_values(self.ypos(i, j)) > 0.5]

            violations = self.get_violated_edges(sol)

            if len(violations) > 0:
                for violation in violations:
                    self.add(constraint=cplex.SparsePair(
                        [el.idx for el in violation],
                        [1.0] * len(violation)),
                        sense='L',
                        rhs=1.0
                    )

        # Restore full parallelism
        self.model.parameters.threads.set(multiprocessing.cpu_count())

        # Print time spent on callbacks
        end = time.time()
        self.sum_time += end - start
