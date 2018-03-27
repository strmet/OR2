from networkx import *
import numpy


class AttributeWarning(UserWarning):
    pass


class OurClass:
    def __init__(self, a=0.0, s="test"):
        self._probability = a
        self._name = s

    @property
    def probability(self):
        return self._probability

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = str(val)

    @probability.setter
    def probability(self, val):
        if not type(val) == float:
            raise TypeError("Expecting a floating point number (it's a probability!); " +
                            "we've been given, instead a: " + str(type(val)))
        if val < 0:
            raise AttributeError("Received a number below 0, which makes no sense")
        elif val > 1:
            raise AttributeError("Received a number above 1, which makes no sense")
        else:
            self._probability = val


list_of_genes = [OurClass(1/(i+1), "Gene#"+str(i)) for i in range(100)]

G = Graph()
G.add_nodes_from(["Gene#"+str(i) for i in range(100)], probability=0.5)
print(G.nodes.data())
L = Graph()
L.add_nodes_from(list_of_genes)
print(L.nodes.data())
a = numpy.reshape(numpy.random.random_integers(0,1,size=100),(10,10))
D = nx.DiGraph(a)
print(D.edges.data())

print("Test eseguito!")
