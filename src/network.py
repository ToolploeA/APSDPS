from itertools import islice
from math import isclose
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy

import utils

class DPS_Graph(nx.DiGraph):
    def __init__(self):
        super().__init__()
    
    # def __repr__(self):
    #     return f"DPS_Graph with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges"
    
    def top_k_shortest_path(self, source_region: list, target_region: list, k: int, num_zero: int | None = 0, inf_step: list | None = None) -> tuple[list, list]:
        '''
        Find the top k shortest paths between source and target regions in the graph.

        Parameters
        ----------
        source_region : list
            The list of source node ID.
        target_region : list
            The list of target node ID.
        k : int
            The max number of shortest paths to find.
        # zero: bool
        #     If True, include paths with zero length.
        num_zero: int | None
            the number of zero-length paths
        inf_step: list | None
            The list of the steps whose weight is +inf in searching. [(opt_id1, opt_id2), (), ...]
        
        Returns
        -------
        tuple[list, list]
            A tuple containing two lists:
            - The first list contains the shortest paths.
            - The second list contains the lengths of the corresponding paths.
        '''
        # if inf_step:
        #     for step in inf_step:
        #         self.set_edge_weight(step[0], step[1], np.inf)
        graph = copy.deepcopy(self)
        if inf_step:
            for step in inf_step:
                graph.set_edge_weight(step[0], step[1], np.inf)
        k = k + num_zero
        paths = [[] for _ in range(k)]
        lengths = [float('inf') for _ in range(k)]
        for source in source_region:
            for target in target_region:
                _graph = copy.deepcopy(graph)
                _graph.remove_nodes_from(filter(lambda x: x != source, source_region))
                _graph.remove_nodes_from(filter(lambda x: x != target, target_region))
                local_paths = list(islice(nx.shortest_simple_paths(_graph, source, target, weight='weight'), k)) # sum of weight
                local_lengths = [nx.path_weight(_graph, path, 'weight') for path in local_paths]
                for path, length in zip(local_paths, local_lengths):
                    if length < lengths[-1]:
                        lengths_copy = copy.deepcopy(lengths)
                        lengths_copy[-1] = length
                        lengths_copy.sort()
                        i = lengths_copy.index(length)
                        lengths.pop()
                        paths.pop()
                        lengths.insert(i, length)
                        paths.insert(i, path)
        paths = [x for x, y in zip(paths, lengths) if not np.isinf(y)]
        lengths = [y for y in lengths if not np.isinf(y)]
        # if not zero:
        #     paths = [x for x, y in zip(paths, lengths) if y != 0]
        #     lengths = [y for y in lengths if y != 0]
        # if inf_step:
        #     for step in inf_step:
        #         self.set_edge_weight(step[0], step[1], 0)

        return paths, lengths
    
    def new_top_k_shortest_path(self, source_region: list, target_region: list, k: int, inf_step: list | None = None) -> tuple[list, list]:
        graph = copy.deepcopy(self)
        if inf_step:
            for step in inf_step:
                graph.set_edge_weight(step[0], step[1], np.inf)
        paths, lengths = utils.top_k(graph, source_region, target_region, k)
        return paths, lengths

    def new_s_t_shortest_path(self, source_region: list, target_region: list, k: int, inf_step: list | None = None) -> tuple[list, list]:
        '''
        返回每一组(s, t)的最短路径，一共有len(source_region) * len(target_region)个结果。
        如果一组(s, t)中有多条0路径，只返回其中一条，和其它(s, t)组中可能的非0路径一起。
        传参k应该为len(source_region) * len(target_region)。
        '''
        graph = copy.deepcopy(self)
        if inf_step:
            for step in inf_step:
                graph.set_edge_weight(step[0], step[1], np.inf)
        paths, lengths = utils.top_k_yen(graph, source_region, target_region, k)
        return paths, lengths

    def new_node(self, node_id: int, node_list: list[int], weight_list: list[float]) -> None:
        '''
        Add a node to the graph.

        Parameters
        ----------
        node_id : int
            The ID of the node to be added.
        node_list : list[int]
            A list of node IDs that this node is connected to.
        weight_list : list[float]
            A list of weights associated with the node.
        '''
        self.add_node(node_id)

        edges = [(node_id, node, {'weight': weight}) for node, weight in zip(node_list, weight_list)] + [(node, node_id, {'weight': weight}) for node, weight in zip(node_list, weight_list)]
        self.add_edges_from(edges)

        # for node, weight in zip(node_list, weight_list):
        #     self.add_edge(node_id, node, weight=weight)
    
    def set_edge_weight(self, node1: int, node2: int, weight: float, neb_id: str = None) -> None:
        '''
        Set the weight of an edge between two nodes.

        Parameters
        ----------
        node1 : int
            The ID of the first node.
        node2 : int
            The ID of the second node.
        weight : float
            The weight to be set for the edge.
        '''
        self[node1][node2]['weight'] = weight
        if np.isclose(weight, np.inf):
            self[node2][node1]['weight'] = weight
        if neb_id is not None:
            self[node1][node2]['neb_id'] = neb_id
