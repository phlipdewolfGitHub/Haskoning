"""
Create and manage a graph structure G=(N,A) for MIP optimization.

In graph theory notation:
- G: Graph
- N: Set of nodes/vertices
- A: Set of arcs/edges
"""

from typing import Set, Dict, List, Tuple, Optional


class Graph:
    """
    Represents a directed graph G=(N,A) for MIP optimization problems.
    
    This graph structure is suitable for network flow problems, shortest path,
    traveling salesman, and other optimization problems.
    """
    
    def __init__(self):
        """
        Initialize an empty graph with no nodes or arcs.
        """
        self.nodes: Set[int] = set()  # N: Set of nodes
        self.arcs: Set[Tuple[int, int]] = set()  # A: Set of arcs (from_node, to_node)
        self.arc_costs: Dict[Tuple[int, int], float] = {}  # Cost/distance for each arc
        self.arc_capacities: Dict[Tuple[int, int], float] = {}  # Capacity for each arc
    
    def add_node(self, node: int) -> None:
        """
        Add a node to the graph.
        
        Args:
            node (int): Node identifier to add
        """
        self.nodes.add(node)
    
    def add_nodes(self, nodes: List[int]) -> None:
        """
        Add multiple nodes to the graph.
        
        Args:
            nodes (List[int]): List of node identifiers to add
        """
        self.nodes.update(nodes)
    
    def add_arc(self, from_node: int, to_node: int, 
                cost: Optional[float] = None, 
                capacity: Optional[float] = None) -> None:
        """
        Add an arc from from_node to to_node.
        
        Args:
            from_node (int): Source node
            to_node (int): Destination node
            cost (Optional[float]): Cost/distance associated with the arc
            capacity (Optional[float]): Capacity constraint for the arc
        """
        # Ensure nodes exist
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        
        # Add the arc
        arc = (from_node, to_node)
        self.arcs.add(arc)
        
        # Store arc attributes
        if cost is not None:
            self.arc_costs[arc] = cost
        if capacity is not None:
            self.arc_capacities[arc] = capacity
    
    def get_arc_cost(self, from_node: int, to_node: int) -> Optional[float]:
        """
        Get the cost of an arc.
        
        Args:
            from_node (int): Source node
            to_node (int): Destination node
            
        Returns:
            Optional[float]: Cost of the arc, or None if not set
        """
        return self.arc_costs.get((from_node, to_node))
    
    def get_arc_capacity(self, from_node: int, to_node: int) -> Optional[float]:
        """
        Get the capacity of an arc.
        
        Args:
            from_node (int): Source node
            to_node (int): Destination node
            
        Returns:
            Optional[float]: Capacity of the arc, or None if not set
        """
        return self.arc_capacities.get((from_node, to_node))
    
    def get_outgoing_arcs(self, node: int) -> List[Tuple[int, int]]:
        """
        Get all arcs originating from a node.
        
        Args:
            node (int): Node identifier
            
        Returns:
            List[Tuple[int, int]]: List of arcs (from_node, to_node)
        """
        return [arc for arc in self.arcs if arc[0] == node]
    
    def get_incoming_arcs(self, node: int) -> List[Tuple[int, int]]:
        """
        Get all arcs arriving at a node.
        
        Args:
            node (int): Node identifier
            
        Returns:
            List[Tuple[int, int]]: List of arcs (from_node, to_node)
        """
        return [arc for arc in self.arcs if arc[1] == node]
    
    def get_num_nodes(self) -> int:
        """
        Get the number of nodes in the graph.
        
        Returns:
            int: Number of nodes |N|
        """
        return len(self.nodes)
    
    def get_num_arcs(self) -> int:
        """
        Get the number of arcs in the graph.
        
        Returns:
            int: Number of arcs |A|
        """
        return len(self.arcs)
    
    def __repr__(self) -> str:
        """
        String representation of the graph.
        
        Returns:
            str: Graph representation
        """
        return f"Graph(N={self.get_num_nodes()}, A={self.get_num_arcs()})"


def create_example_graph() -> Graph:
    """
    Create an example graph for demonstration purposes.
    
    Returns:
        Graph: A sample graph with nodes and arcs
    """
    G = Graph()
    
    # Add nodes
    G.add_nodes([1, 2, 3, 4, 5])
    
    # Add arcs with costs
    G.add_arc(1, 2, cost=10.0)
    G.add_arc(1, 3, cost=15.0)
    G.add_arc(2, 3, cost=5.0)
    G.add_arc(2, 4, cost=12.0)
    G.add_arc(3, 4, cost=8.0)
    G.add_arc(3, 5, cost=9.0)
    G.add_arc(4, 5, cost=7.0)
    
    return G


if __name__ == "__main__":
    # Example usage
    G = create_example_graph()
    
    print("Graph G=(N,A) created:")
    print(f"  Nodes (N): {sorted(G.nodes)}")
    print(f"  Number of nodes: |N| = {G.get_num_nodes()}")
    print(f"\n  Arcs (A): {sorted(G.arcs)}")
    print(f"  Number of arcs: |A| = {G.get_num_arcs()}")
    
    print("\nArc costs:")
    for arc in sorted(G.arcs):
        cost = G.get_arc_cost(arc[0], arc[1])
        print(f"  Arc {arc}: cost = {cost}")
    
    print(f"\n{repr(G)}")

