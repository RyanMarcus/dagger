from graph import DAG
import numpy as np
from collections import defaultdict


class Scheduler:
    def __init__(self, dag, deadline, num_machine_types):
        self.dag = dag
        self.n_types = num_machine_types
        self.deadline = deadline
        self.reset()
        
    def __eligible(self):
        # find all the vertices in self.not_done whose
        # parents are scheduled (in self.done)
        for v in self.not_done:
            parents = self.dag.parents_of(v)
            if all(x in self.done for x in parents):
                yield v

    def __add(self, vertex):
        # adds the vertex to the queue of clusters
        self.clusters[-1].append(vertex)
        self.done.add(vertex)
        self.not_done.remove(vertex)
    
    def reset(self):
        self.clusters = [[0]]
        self.cluster_types = [0]
        self.done = set([0])
        self.not_done = set(range(self.dag.num_vertices())) - set([0])
        self.current_vertex = 0
        
    def actions(self):
        return [self.lowest_vertex_weight_child,
                self.highest_edge_weight_child,
                self.least_slack,
                self.most_slack,
                self.highest_vertex_weight,
                self.lowest_vertex_weight,
                self.split,
                self.promote]
                

    def lowest_vertex_weight_child(self):
        # find the child of the last added vertex with the lowest
        # vertex weight and add it. If there is no such child,
        # do nothing.
        if len(self.clusters[-1]) == 0:
            return
        
        last = self.clusters[-1][-1]
        eligible = set(self.__eligible()) & set(self.dag.children_of(last))
        min_child = min(eligible,
                        key=lambda x: self.dag.edge_weight(last, x),
                        default=None)

        if min_child == None:
            return

        self.__add(min_child)

    def highest_edge_weight_child(self):
        # find the child of the last added vertex with the lowest
        # vertex weight and add it. If there is no such child,
        # do nothing.
        if len(self.clusters[-1]) == 0:
            return
        
        last = self.clusters[-1][-1]
        eligible = set(self.__eligible()) & set(self.dag.children_of(last))

        max_child = max(eligible,
                        key=lambda x: self.dag.vertex_weight(x,self.cluster_types[-1]),
                        default=None)

        if max_child == None:
            return

        self.__add(max_child)

    def least_slack(self):
        """ find the child with the least slack and add it """
        slack = self.dag.slack(self.deadline)
        v = min(self.__eligible(), key=lambda x: slack[x])
        self.__add(v)

    def most_slack(self):
        """ find the child with the most slack and add it """
        slack = self.dag.slack(self.deadline)
        v = max(self.__eligible(), key=lambda x: slack[x])
        self.__add(v)
        
    def highest_vertex_weight(self):
        # find the vertex with the highest weight and add it
        v = max(self.__eligible(),
                key=lambda x: self.dag.vertex_weight(x, self.cluster_types[-1]))
        self.__add(v)
        

    def lowest_vertex_weight(self):
        # find the vertex with the lowest weight and add it
        v = max(self.__eligible(),
                key=lambda x: self.dag.vertex_weight(x, self.cluster_types[-1]))
        self.__add(v)

    def promote(self):
        self.cluster_types[-1] = min(self.cluster_types[-1] + 1, self.n_types-1)
        
    def split(self):
        """ creates a new cluster assuming that the last cluster has
        at least one vertex assigned to it (if not, does nothing) """
        if len(self.clusters[-1]) == 0:
            return
        
        self.clusters.append([])
        self.cluster_types.append(0)

    def cost(self):
        return self.dag.cost_of(self.clusters, self.cluster_types, self.deadline)

    def is_done(self):
        return len(self.not_done) == 0

    def state_vector(self):
        """ returns the state vector for the current context """
        vec_to_cluster = defaultdict(list)
        vec_to_mt = defaultdict(lambda: 0)
        for cluster, mt in zip(self.clusters, self.cluster_types):
            for vertex in cluster:
                vec_to_cluster[vertex] = cluster
                vec_to_mt[vertex] = mt
        
        
        vec = []
        for i in range(self.dag.num_vertices()):
            for j in range(i+1,self.dag.num_vertices()):
                if i in vec_to_cluster[j]:
                    vec.append(0)
                else:
                    vec.append(self.dag.edge_weight(i, j))

        for i in range(self.dag.num_vertices()):
            for mt in range(self.n_types):
                vec.append(self.dag.vertex_weight(i, mt))

        for i in range(self.dag.num_vertices()):
            vec.append(vec_to_mt[i])
                
        return np.array(vec)
                    
    def __repr__(self):
        return str(self.clusters)


if __name__ == "__main__":
    adj = np.array([[0, 1, 2, 0], [0, 0, 0, 5], [0, 0, 0, 3], [0, 0, 0, 0]])
    w = np.array([[3, 2], [4, 1], [1, 1], [8, 7]])

    d = DAG(w, adj, (1, 5))
    s = Scheduler(d, 20, 2)
    s.least_slack()
    s.split()
    s.promote()
    s.least_slack()
    s.least_slack()

    print(s)
