# < begin copyright > 
# Copyright Ryan Marcus 2017
# 
# This file is part of dagger.
# 
# dagger is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# dagger is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with dagger.  If not, see <http://www.gnu.org/licenses/>.
# 
# < end copyright > 
from graph import DAG
import numpy as np
from collections import defaultdict

class Scheduler:
    def __init__(self, dag, deadline, num_machine_types):
        self.dag = dag
        self.n_types = num_machine_types
        self.deadline = deadline
        self.reset()
        self.__vertex_weight_info = []
        for i in range(self.dag.num_vertices()):
            for mt in range(self.n_types):
                self.__vertex_weight_info.append(self.dag.vertex_weight(i, mt))


    def __complex_action(f):
        def wrapper(self):
            curr = str(self)
            current_cost = self.cost()            
            if not f(self):
                # action was a no-op
                return None
            
            new_cost = self.cost()
            return current_cost - new_cost
        return wrapper
        
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
                

    @__complex_action
    def lowest_vertex_weight_child(self):
        # find the child of the last added vertex with the lowest
        # vertex weight and add it. If there is no such child,
        # do nothing.
        if len(self.clusters[-1]) == 0:
            return None
        
        last = self.clusters[-1][-1]
        eligible = set(self.__eligible()) & set(self.dag.children_of(last))
        min_child = min(eligible,
                        key=lambda x: self.dag.edge_weight(last, x),
                        default=None)

        if min_child == None:
            return None

        self.__add(min_child)
        return True

    @__complex_action
    def highest_edge_weight_child(self):
        # find the child of the last added vertex with the lowest
        # vertex weight and add it. If there is no such child,
        # do nothing.
        if len(self.clusters[-1]) == 0:
            return None
        
        last = self.clusters[-1][-1]
        eligible = set(self.__eligible()) & set(self.dag.children_of(last))

        max_child = max(eligible,
                        key=lambda x: self.dag.vertex_weight(x,self.cluster_types[-1]),
                        default=None)

        if max_child == None:
            return None

        self.__add(max_child)
        return True

    @__complex_action
    def least_slack(self):
        """ find the child with the least slack and add it """
        slack = self.dag.slack(self.deadline)
        v = min(self.__eligible(), key=lambda x: slack[x])
        self.__add(v)
        return True
        
    @__complex_action
    def most_slack(self):
        """ find the child with the most slack and add it """
        slack = self.dag.slack(self.deadline)
        v = max(self.__eligible(), key=lambda x: slack[x])
        self.__add(v)
        return True

    @__complex_action        
    def highest_vertex_weight(self):
        # find the vertex with the highest weight and add it
        v = max(self.__eligible(),
                key=lambda x: self.dag.vertex_weight(x, self.cluster_types[-1]))
        self.__add(v)
        return True
        
    @__complex_action
    def lowest_vertex_weight(self):
        # find the vertex with the lowest weight and add it
        v = max(self.__eligible(),
                key=lambda x: self.dag.vertex_weight(x, self.cluster_types[-1]))
        self.__add(v)
        return True

    @__complex_action
    def promote(self):
        if self.cluster_types[-1] == self.n_types - 1:
            return None
        self.cluster_types[-1] = self.cluster_types[-1] + 1
        return True

    def split(self):
        """ creates a new cluster assuming that the last cluster has
        at least one vertex assigned to it (if not, does nothing) """
        if len(self.clusters[-1]) == 0:
            return None
        
        self.clusters.append([])
        self.cluster_types.append(0)
        return - self.dag.machine_costs[0] * 60

    def cost(self):
        return self.dag.cost_of(self.clusters, self.cluster_types, self.deadline)

    def latency(self):
        return self.dag.latency_of(self.clusters, self.cluster_types);
    
    def default_cost(self):
        return self.dag.cost_of([], [], self.deadline)
    
    def is_done(self):
        return len(self.not_done) == 0

    def state_vector(self):
        """ returns the state vector for the current context """
        vec_to_cluster = defaultdict(list)
        vec_to_mt = defaultdict(lambda: 0)
        for cluster, mt in zip(self.clusters, self.cluster_types):
            for vertex in cluster:
                vec_to_cluster[vertex] = np.array(cluster)
                vec_to_mt[vertex] = mt
        
        
        vec = []
        for i in range(self.dag.num_vertices()):
            cluster = vec_to_cluster[i]
            edge_sum = np.array(self.dag.adj_matrix[i])
            edge_sum[cluster] = 0
            edge_sum[edge_sum == -1] = 0
            edge_sum = np.sum(edge_sum)
            vec.append(edge_sum)

        vec.extend(self.__vertex_weight_info)

        for i in range(self.dag.num_vertices()):
            vec.append(vec_to_mt[i])
                
        return np.array(vec)
                    
    def __repr__(self):
        return str(self.clusters) + ", " + str(self.cluster_types)


if __name__ == "__main__":
    adj = np.array([[-1, 10, 20, -1], [-1, -1, -1, 50],
                    [-1, -1, -1, 30], [-1, -1, -1, -1]])
    w = np.array([[30, 20], [40, 10], [10, 10], [80, 70]])

    d = DAG(w, adj, (1, 5))
    
    s = Scheduler(d, 20, 2)
    actions = [s.split, s.least_slack, s.least_slack, s.split, s.least_slack]
    actions = [x() for x in actions]
    print(actions, sum(actions))
    print("default:", s.default_cost(), "final:", s.cost())
