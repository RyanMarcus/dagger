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
import numpy as np
from collections import defaultdict
from functools import lru_cache

def flatten(l):
    return (item for sublist in l for item in sublist)

def dag_from_file(from_file):
    weights = []
    adj_matrix = []
    with open(from_file, "r") as f:
        num_vertices = int(next(f).strip())
        for i in range(num_vertices):
            l = next(f)
            weights.append([int(x) for x in l.strip().split(",")])

        for l in f:
            adj_matrix.append([int(x) for x in l.strip().split(",")])

    return DAG(weights, adj_matrix, [1, 5])


class DAG:
    def __init__(self, weights, adj_matrix, machine_costs):
        self.weights =np.array(weights)
        self.adj_matrix = np.array(adj_matrix)
        self.machine_costs = np.array(machine_costs)
        self.parents = dict()
        for i in range(self.num_vertices()):
            self.parents[i] = list(self.__find_parents(i))

        
    def get_root(self):
        return 0
    
    def num_vertices(self):
        return self.weights.shape[0]
    
    def edge_weight(self, a, b):
        return self.adj_matrix[a][b]

    def vertex_weight(self, a, machine_type):
        return self.weights[a][machine_type]

    def children_of(self, a):
        return np.where(self.adj_matrix[a] > -1)[0]

    def sinks(self):
        return np.where(np.all(self.adj_matrix == -1, axis=1))[0]

    def parents_of(self, a):
        return self.parents[a]
    
    def __find_parents(self, a):
        for i in range(self.weights.shape[0]):
            if a in self.children_of(i):
                yield i
    
    def __resolve(self, clusters, cluster_types):
        clusters = [[{"start": None, "end": None, "vertex": v, "type": mt}
                     for v in x]
                    for x, mt in zip(clusters, cluster_types)]
        
        vertices = { x["vertex"]: x for x in flatten(clusters) }
        
        def is_resolved(x):
            return x["start"] != None

        def parents(x):
            for p in self.parents_of(x["vertex"]):
                yield vertices[p]
        
        while any(not is_resolved(x) for x in flatten(clusters)):
            # try and resolve the next task on each cluster
            resolved_one = False
            for cluster, mt in zip(clusters, cluster_types):
                # find the first unresolved vertex, if it exists
                nxt = None
                try:
                    nxt = next(x for x in cluster if not is_resolved(x))
                except StopIteration:
                    continue # no unresolved vertices left on this one
                
                # see if all of this vertex's parents are resolved
                if all(is_resolved(p) for p in parents(nxt)):
                    # all my parents are resolved, which means
                    # I should be resolved!
                        
                    # my start time will be the latest end time of my
                    # parents                        
                    nxt["start"] = max((p["end"] for p in parents(nxt)),
                                       default=0)

                    # my end time will be my vertex weight for this machine
                    # type...
                    nxt["end"] = nxt["start"] + self.vertex_weight(nxt["vertex"], mt)

                    # ... plus my largest outgoing edge to a vertex that
                    # is not in my cluster
                    outgoing_edges = ((x, self.edge_weight(nxt["vertex"], x))
                                      for x in self.children_of(nxt["vertex"]))

                    outgoing_edges = (e for e in outgoing_edges
                                      if e[0] not in [x["vertex"] for x in cluster])

                    outgoing_weights = (e[1] for e in outgoing_edges)
                    max_outgoing = max(outgoing_weights, default=0)
                    nxt["end"] += max_outgoing

                    # set resolved one to true so the while loop continues
                    resolved_one = True
                        
            if not resolved_one:
                # we couldn't resolve the schedule. must be a back edge.
                raise ValueError("Clusters could not be resolved, verify that schedule is valid")

        return clusters

    def latency_of(self, clusters, machine_types):
        resolved = self.__resolve(clusters, machine_types)
        return max(x["end"] for x in flatten(resolved))

    def cost_of(self, clusters, machine_types, deadline):
        machine_types = np.array(machine_types)
        latency = self.latency_of(clusters, machine_types)

        penalty = 0 if latency < deadline else (latency - deadline) * 5

        # create tuples that are (runtime, machine_type) pairs
        resolved = self.__resolve(clusters, machine_types)
        totals = np.array([0 for _ in self.machine_costs])

        for cluster in resolved:
            mt = cluster[-1]["type"]
            time = cluster[-1]["end"]
            totals[mt] += time

        runtime_cost = sum(totals * self.machine_costs)
        startup_cost = sum(self.machine_costs[machine_types] * 60)
        return runtime_cost + startup_cost + penalty
            
    def t_levels(self, clusters=[], cluster_types=[]):
        """ returns a map from each vertex to its t-level (the earliest
        possible starting time for that vertex) """
        vertex_to_mt = defaultdict(lambda: 0)
        vertex_to_cluster = defaultdict(list)
        for cluster, mt in zip(clusters, cluster_types):
            for vertex in cluster:
                vertex_to_mt[vertex] = mt
                vertex_to_cluster[vertex] = cluster

        toR = {0: 0}
        for i in range(1, self.num_vertices()):
            # my t-level is the minimum of, for each parent p:
            # p's t-level + p's weight + the edge weight if we are different
            # clusters.
            toR[i] = min(toR[p]
                         + self.vertex_weight(p, vertex_to_mt[p])
                         + (0 if p in vertex_to_cluster[i] else
                            self.edge_weight(p, i))
                          for p in self.parents_of(i))

        return toR

    def b_levels(self, deadline, clusters=[], cluster_types=[]):
        """ returns a map from each vertex to its b-level (the latest
        possible starting time for a vertex so that the deadline can still
        be met) """

        sinks = list(self.sinks())
        vertex_to_mt = defaultdict(lambda: 0)
        vertex_to_cluster = defaultdict(list)
        for cluster, mt in zip(clusters, cluster_types):
            for vertex in cluster:
                vertex_to_mt[vertex] = mt
                vertex_to_cluster[vertex] = cluster

        # we need a reverse topological ordering of the DAG
        rtopo = list(sinks)
        remaining = set(range(self.num_vertices()))
        remaining -= set(rtopo)

        while len(remaining) != 0:
            # add every vertex v from remaining for which:
            # all of v's children are already in rtopo
            added = False
            srtopo = set(rtopo)
            match = list(v for v in remaining
                         if set(self.children_of(v)).issubset(srtopo))
            added = len(match) != 0
            rtopo.extend(match)
            remaining -= set(match)
                
            if not added:
                raise ValueError("Backedge found when building rtopo list")
        
        # now we have the reverse topological sort.
        # for each vertex v in rtopo, it's b-level is:
        # if v is a sink, the b-level is deadline - weight(v)
        # otherwise, b-level is the maximum of (child's b-level + edge weight
        # to the child if not in same cluster) - weight

        toR = dict()
        for sink in sinks:
            toR[sink] = deadline - self.vertex_weight(sink,
                                                      vertex_to_mt[sink])

        rtopo = rtopo[len(sinks):] # cut out the sinks, already have
        for v in rtopo:
            max_child = max(toR[c] + (0 if c in vertex_to_cluster[v]
                                      else self.edge_weight(v, c))
                            for c in self.children_of(v))
            toR[v] = max_child - self.vertex_weight(v, vertex_to_mt[v])

        return toR

    def slack(self, deadline, clusters=[], cluster_types=[]):
        t_levels = self.t_levels(clusters=clusters, cluster_types=cluster_types)
        b_levels = self.b_levels(deadline, clusters=clusters,
                                 cluster_types=cluster_types)
        t = []
        b = []
        for i in range(self.num_vertices()):
            t.append(t_levels[i])
            b.append(b_levels[i])

        return np.array(b) - np.array(t)
                    

if __name__ == "__main__":
    adj = np.array([[-1, 1, 2, -1], [-1, -1, -1, 5],
                    [-1, -1, -1, 3], [-1, -1, -1, -1]])
    w = np.array([[3, 2], [4, 1], [1, 1], [8, 7]])

    d = DAG(w, adj, (1, 5))
    
    #print(d.cost_of([(0, 1), (2, 3)], (1, 0), 100))
    print(d.t_levels())
    print(d.b_levels(20))
    print(d.slack(20))

    d = dag_from_file("sparselu1.txt")
    #print(d.slack(2000))
