graph_prompt = {
    "intro": '''You are given a task to generate an adjacency matrix of an undirected graph with $n$ nodes and $m$ edges. The graph also need to satisfy several properties, which may includes (1) the min among nodes (2) max degree among nodes (3) number of connected components (4) diameter of graph (5) have cycle or not. Diameter of a graph is the maximum distance between any two nodes. A connected component is a subgraph in which any two nodes are connected to each other by paths, and which is connected to no other nodes in the supergraph. A cycle is a path of length at least 3 that starts and ends at the same node. 

    The graph may be unconnected, but every node has AT LEAST ONE EDGE. If you don't know how to generate a graph that satisfy all properties, you can output a graph that at least have the correct number of nodes and edges, and match other properties at your best effort. Output the adjacency matrix of the graph only without any other information.''',
    "examples": [
        (
            '''Undirected graph with 9 nodes, 11 edges, 1 connected component, one or more cycle, min degree 1, max diameter 5, max degree 3.''',
            '''```
0 1 0 0 1 1 0 0 0
1 0 0 0 0 0 0 1 1
0 0 0 1 0 0 0 0 0
0 0 1 0 0 0 1 0 1
1 0 0 0 0 1 0 0 0
1 0 0 0 1 0 0 1 0
0 0 0 1 0 0 0 0 0
0 1 0 0 0 1 0 0 1
0 1 0 1 0 0 0 1 0
```end'''
        ),
        (
            '''Undirected graph with min degree 1, max diameter 4, 16 nodes, one or more cycle, max degree 11, 1 connected component, 22 edges.''',
            '''```
0 1 1 0 0 1 1 0 1 0 0 0 0 0 0 0
1 0 1 1 1 0 1 1 0 1 1 1 0 1 0 1
1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
1 1 1 0 0 1 0 0 0 1 0 1 0 0 1 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```end'''
        ),
        (
            '''Undirected graph with 6 edges, max degree 5, max diameter 2, 6 nodes.''',
            '''```
0 1 1 0 0 0
1 0 1 1 1 1
1 1 0 0 0 0
0 1 0 0 0 0
0 1 0 0 0 0
0 1 0 0 0 0
```end'''
        ),
    ],
    "prompt": '''{description}'''
}
