import numpy as np

def flatten(l):
    return (item for sublist in l for item in sublist)

class DAG:
    def __init__(self, weights, adj_matrix, machine_costs):
        self.weights = weights
        self.adj_matrix = adj_matrix
        self.machine_costs = np.array(machine_costs)

    def get_root(self):
        return 0
    
    def num_vertices(self):
        return self.weights.shape[0]
    
    def edge_weight(self, a, b):
        return self.adj_matrix[a][b]

    def vertex_weight(self, a, machine_type):
        return self.weights[a][machine_type]

    def children_of(self, a):
        return np.where(self.adj_matrix[a] > 0)[0]

    def parents_of(self, a):
        for i in range(self.weights.shape[0]):
            if a in self.children_of(i):
                yield i
    
    def __resolve(self, clusters, cluster_types):
        clusters = [[{"start": None, "end": None, "vertex": v, "type": mt} for v in x]
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
                try:
                    nxt = next(x for x in cluster if not is_resolved(x))
                    # see if all of this vertex's parents are resolved
                    if all(is_resolved(p) for p in parents(nxt)):
                        # all my parents are resolved, which means I should be resolved!
                        # my start time will be the latest end time of my parents                        
                        nxt["start"] = max((p["end"] for p in parents(nxt)), default=0)

                        # my end time will be my vertex weight for this machine type...
                        nxt["end"] = nxt["start"] + self.vertex_weight(nxt["vertex"], mt)

                        # ... plus my largest outgoing edge to a vertex that is not in my cluster
                        outgoing_edges = ((x, self.edge_weight(nxt["vertex"], x))
                                          for x in self.children_of(nxt["vertex"]))

                        outgoing_edges = (e for e in outgoing_edges
                                          if e[0] not in [x["vertex"] for x in cluster])

                        outgoing_weights = (e[1] for e in outgoing_edges)
                        max_outgoing = max(outgoing_weights, default=0)
                        nxt["end"] += max_outgoing

                        # set resolved one to true so the while loop continues
                        resolved_one = True
                        
                except StopIteration:
                    continue # no unresolved left in this cluster

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
        totals = np.array([0 for _ in machine_types])

        for cluster in resolved:
            mt = cluster[-1]["type"]
            time = cluster[-1]["end"]
            totals[mt] += time

        runtime_cost = sum(totals * self.machine_costs)
        startup_cost = sum(self.machine_costs[machine_types] * 60)
        return runtime_cost + startup_cost + penalty
            

        
        

adj = np.array([[0, 1, 2, 0], [0, 0, 0, 5], [0, 0, 0, 3], [0, 0, 0, 0]])
w = np.array([[3, 2], [4, 1], [1, 1], [8, 7]])

d = DAG(w, adj, (1, 5))

print(d.cost_of([(0, 1), (2, 3)], (1, 0), 100))
            
