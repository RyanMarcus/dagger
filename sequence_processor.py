from graph import DAG


class Scheduler:
    def __init__(self, dag, num_machine_types):
        self.dag = dag
        self.n_types = num_machine_types
        self.clusters = [{"tasks": [x], "type": 0} for x in range(dag.num_vertices())]
        self.vertex_map = { i: self.clusters[i] for i in range(dag.num_vertices()) }
        self.done = set()
        self.not_done = set(range(dag.num_vertices()))
        self.current_vertex = 0
        
    
    def lowest_vertex_weight_child(self):
        pass

    def highest_edge_weight_child(self):
        pass

    def least_slack(self):
        pass

    def highest_vertex_weight(self):
        pass

    def lowest_vertex_weight(self):
        pass

    def promote(self):
        pass

    def split(self):
        pass
