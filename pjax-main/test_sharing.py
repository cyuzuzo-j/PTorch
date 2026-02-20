
import jax.numpy as jnp
from pjax.core.computation import Parameter, Operation, make_computation

# Create parameters with the same name
p1 = Parameter(jnp.array([1.]), name="w")
p2 = Parameter(jnp.array([2.]), name="w")

print(f"p1 hash: {hash(p1)}")
print(f"p2 hash: {hash(p2)}")
print(f"p1 == p2: {p1 == p2}")

# Create an operation that uses both
# Assuming some operation exists or creating a dummy one
def add_op(a, b):
    return a + b

def add_proj(a, b, z):
    return (a, b) # Dummy

add = make_computation("add", add_op, add_proj)

# Create op using both params
op = add(p1, p2)

print(f"Op parents: {op.parents}")
print(f"Parent 1 hash: {hash(op.parents[0])}")
print(f"Parent 2 hash: {hash(op.parents[1])}")
print(f"Parent 1 == Parent 2: {op.parents[0] == op.parents[1]}")

# Use networkx to see graph structure
import networkx as nx
def get_graph(computation):
    graph = nx.DiGraph()
    def add(node):
        if node in graph: return
        graph.add_node(node)
        for p in node.parents:
            add(p)
            graph.add_edge(p, node)
    add(computation)
    return graph

g = get_graph(op)
print(f"Number of nodes in graph: {len(g.nodes)}")
# If p1 == p2, there should be 2 nodes: op and p1 (which is same as p2)
# If p1 != p2, there should be 3 nodes: op, p1, p2

for node in g.nodes:
    print(f"Node: {node}, Name: {node.name}")
