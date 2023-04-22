graph_prompt = {
    "intro": '''You are given a task to generate any adjacency matrix of an undirected graph that satisfy several properties, including node count, edge count. The graph may be unconnected, but every node has at least one edge. The graph may be a tree. Follow the following example exactly:''',
    "examples": [
        (
            '''Graph with 5 nodes, 10 edges.''',
            '''```
0 1 1 0 1
1 0 1 1 0
1 1 0 1 0
0 1 1 0 1
1 0 0 1 0
```end'''
        ),
        (
            '''Graph with 7 nodes, 21 edges.''',
            '''```
0 1 1 1 1 1 1
1 0 1 1 1 1 1
1 1 0 1 1 1 1
1 1 1 0 1 1 1
1 1 1 1 0 1 1
1 1 1 1 1 0 1
1 1 1 1 1 1 0
```end'''
        ),
        (
            '''Graph with 20 nodes, 19 edges.''',
            '''```
0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0
```end'''
        ),
    ],
    "prompt": '''{description}'''
}
