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
    '''
    def __call__(self):
        for j in self.locations:
            isused = self.get_values(self.used[j])
            served = sum(self.get_values(
                [self.supply[c][j] for c in self.clients]))
            if served > (len(self.clients) - 1.0) * isused + EPS:
                print('Adding lazy constraint %s <= %d*used(%d)' %
                      (' + '.join(['supply(%d)(%d)' % (x, j) for x in self.clients]),
                       len(self.clients) - 1, j))
                self.add(constraint=cplex.SparsePair(
                    [self.supply[c][j] for c in self.clients] + [self.used[j]],
                    [1.0] * len(self.clients) + [-(len(self.clients) - 1)]),
                    sense='L',
                    rhs=0.0)
    '''
    def __call__(self):

        sol = [self.EdgeSol(self.ypos(i, j), i + 1, j + 1)
                   for i in range(self.n_nodes)
                   for j in range(self.n_nodes)
                   if self.get_values(self.ypos(i, j)) > 0.5]
        print(len(sol))

        input()
