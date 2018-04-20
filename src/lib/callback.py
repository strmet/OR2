import cplex
from cplex.callbacks import LazyConstraintCallback

class LazyCallback(LazyConstraintCallback):
    """Lazy constraint callback to enforce the capacity constraints.

    If used then the callback is invoked for every integer feasible
    solution CPLEX finds. For each location j it checks whether
    constraint

    sum(c in C) supply[c][j] <= (|C| - 1) * used[j]

    is satisfied. If not then it adds the violated constraint as lazy
    constraint.
    """

    # Callback constructor. Fields 'locations', 'clients', 'used', 'supply'
    # are set externally after registering the callback.
    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)

    def add_violating_constraint(self, crossing):
        print(crossing)

        #### In questa parte c'è l'errore
        if len(crossing) > 0:
            coefficients = [1] * len(crossing)
            self.add(
                 constraint=cplex.SparsePair(
                    ind=crossing,
                    val=coefficients
                ),
                sense=["L"],
                rhs=[1],
            )



    def __call__(self):

        sol = [self.EdgeSol(self.ypos(i, j), i, j)
                   for i in range(self.n_nodes)
                   for j in range(self.n_nodes)
                   if self.get_values(self.ypos(i, j)) > 0.5]

        violations = self.get_violated_edges(sol)

        print(violations)
        if len(violations) > 0:
            for violation in violations:
                self.add_violating_constraint(violation)

