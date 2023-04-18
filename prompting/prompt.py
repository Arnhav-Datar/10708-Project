graph_prompt = {
    "intro": '''You are given a task to generate any adjacency matrix of an undirected graph that satisfy several properties, including node count, edge count, maximum and minimum node degrees. You are only allow to output the adjaency matrix. The graph may be unconnected. The graph may be a tree. Here are some examples:''',
    "examples": [
        (
            '''Graph with 5 nodes, 10 edges, max node degree 3, min node degree 2.''',
            '''```
0 1 1 0 1
1 0 1 1 0
1 1 0 1 0
0 1 1 0 1
1 0 0 1 0
```'''
        ),
        (
            '''Graph with 3 nodes, 3 edges.''',
            '''```
0 1 1
1 0 1
1 1 0
```'''
        ),
    ],
    "prompt": '''{description}'''
}
