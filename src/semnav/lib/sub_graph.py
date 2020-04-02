from __future__ import print_function

import networkx as nx

from semnav.lib.categories import room2category, behavior_id2category
from semnav.lib.sem_graph import Node, Edge


class SubGraph(object):
    """Sub-graph of SemGraph used for graph networks.
    """

    def __init__(self, sem_graph, cur_position, n_neighbor_dist, cur_behavior_id=None):
        """Build a local subgraph from sem_graph centered around the cur_position node or
        cur_behavior_id edge (which is an outgoing edge of the cur_position node). The subgraph
        extends to a max of n_neighbor_dist nodes or edges away from the current node/edge.

        Args:
            sem_graph: Semantic graph.
            cur_position: Node or Edge representing the current position in the graph, or a str
                    representing the node name of a Node.
            n_neighbor_dist: Int representing the max distance from the current position that the subgraph
                    will contain.
            cur_behavior_id: Current edge behavior ID as a string (e.g. 'tr'). If this is None, then
                    the subgraph will be cropped centered around the current node (cur_position). If
                    cur_behavior_id is NOT None, then the subgraph will be centered around the
                    current edge (which is computed based on cur_position and cur_behavior_id).
        """
        self.nxG, self.nodes, self.edges, self.nxG_compact = self._build_subgraph(
            sem_graph,
            cur_position,
            n_neighbor_dist,
            cur_behavior_id=cur_behavior_id
            )

    @classmethod
    def get_successors(cls, cur_node, depth, forward):
        if depth <= 0:
            return []

        if forward is True:  # Find successors
            neighboring_edges = cur_node.outgoing_edges
            start_or_end_node = 'end_node'
        elif forward is False:  # Find predecessors
            neighboring_edges = cur_node.incoming_edges
            start_or_end_node = 'start_node'
        else:
            raise ValueError('The forward parameter must be a boolean.')

        successor_nodes = []
        for edge in neighboring_edges:
            successor_nodes.append(getattr(edge, start_or_end_node))

        all_future_edges = []
        if depth > 1:
            all_future_successors = []
            for successor_node in successor_nodes:
                future_successors, future_edges = cls.get_successors(successor_node, depth - 1, forward=forward)
                all_future_successors.extend(future_successors)
                all_future_edges.extend(future_edges)
            successor_nodes.extend(all_future_successors)

        successor_edges = neighboring_edges + all_future_edges
        return successor_nodes, successor_edges

    def _build_subgraph(self, sem_graph, cur_position, n_neighbor_dist, cur_behavior_id=None):
        """Build a list of nodes (nbunch) and let networkx extract the subgraph.
        """
        if isinstance(cur_position, str):
            cur_position = sem_graph.nodes[cur_position]

        if cur_behavior_id is not None:  # Set cur_position to be an Edge
            assert isinstance(cur_position, Node)
            for edge in cur_position.outgoing_edges:
                if edge.behavior_id == cur_behavior_id:
                    cur_position = edge
                    break

        if isinstance(cur_position, Node):  # Build local graph centered around node
            cur_node = cur_position
            forward_dist = n_neighbor_dist
        elif isinstance(cur_position, Edge):  # Build local graph centered around edge
            cur_node = cur_position.start_node
            forward_dist = n_neighbor_dist + 1
        else:
            raise ValueError('Invalid current position input type.')
        nbunch = [cur_node]

        # Look at successors and successors of successors
        successor_nodes, successor_edges = self.get_successors(cur_node, depth=forward_dist, forward=True)
        nbunch += successor_nodes

        # Look at predecessors and predecessors of predecessors
        predecessor_nodes, predecessor_edges = self.get_successors(cur_node, depth=n_neighbor_dist, forward=False)
        nbunch += predecessor_nodes

        nbunch_edges = successor_edges + predecessor_edges

        nx_subgraph = nx.OrderedDiGraph()
        nx_subgraph.add_nodes_from(nbunch)
        for u, v, cur_obj in sem_graph.nxG.edges(data='object'):
            if (u in nx_subgraph) and (v in nx_subgraph) and True:
                nx_subgraph.add_edge(u, v, object=cur_obj)

        nodes = {node.name: node for node in nx_subgraph.nodes}
        edges = [edge_tuple[2] for edge_tuple in nx_subgraph.edges(data='object')]

        nxG_compact = self.convert_to_compact(nx_subgraph, store_features=False)
        return nx_subgraph, nodes, edges, nxG_compact

    def convert_to_compact(self, G, store_features):
        """Generates a compact networkx graph for the input graph. The features of each node and
        edge are converted to room categories and behavior categories, which may result in a loss
        of information (e.g. s_r -> s category) in the networkx compact graph.
        """
        nxG_compact = nx.OrderedDiGraph()

        # Create nodes
        for node in G.nodes:
            # Create node
            if store_features is True:  # Store a features attribute
                nxG_compact.add_node(node.name, room_category=room2category(node.name),
                                     features=node['features'])
            else:
                nxG_compact.add_node(node.name, room_category=room2category(node.name))

        # Create edges
        for u, v, edge in G.edges(data='object'):
            if store_features is True:
                nxG_compact.add_edge(u.name, v.name,
                                     behavior_category=behavior_id2category(edge.behavior_id),
                                     features=edge['features'])
            else:
                nxG_compact.add_edge(u.name, v.name,
                                     behavior_category=behavior_id2category(edge.behavior_id))

        assert len(nxG_compact.nodes) == len(G.nodes)
        assert len(nxG_compact.edges) == len(G.edges)

        return nxG_compact
