import pdb
from collections import deque

def floyd_warshall(graph):
    """
    Finds the shortest paths between all pairs of vertices in a weighted graph using the Floyd-Warshall algorithm.
    
    Parameters:
        graph (list of lists): The graph representation as an adjacency matrix. The weight of edge (u, v) is stored
                               in graph[u][v]. If there's no edge between u and v, the weight should be set to infinity
                               (or a sufficiently large value).
                               
    Returns:
        list of lists: A 2D matrix containing the shortest distances between all pairs of vertices.
    """
    num_vertices = len(graph)
    
    # Initialize the distance matrix with the same values as the adjacency matrix.
    dist = [[1 if x == 1 else 0x777777 for x in row] for row in graph]
    for i in range(num_vertices):
        dist[i][i] = 0
    
    # Main algorithm
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                # If there's a shorter path from vertex i to vertex j through vertex k, update the distance.
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

def find_path(u, pre):
    path = []
    while (pre[u] != -1):
        v = u
        u = pre[u]
        path.append((u, v))
    return list(reversed(path))

def wrap(dist, pre, roottree, leaftree):
    paths1 = []
    paths2 = []
    for u in roottree:
        if dist[u] < float('inf'):
            paths1.append((dist[u], u, find_path(u, pre)))
    for u in leaftree:
        if dist[u] < float('inf'):
            paths2.append((dist[u], u, find_path(u, pre)))
    
    return paths1, paths2

def bfs(start, roottree, leaftree, graph):
    """
    Finds the shortest paths between the 'start' to all vertices in roottree and leaftree
    Parameters:
        graph (list of lists): The graph representation as an adjacency matrix. The weight of edge (u, v) is stored
                               in graph[u][v]. If there's no edge between u and v, the weight should be set to infinity
                               (or a sufficiently large value).
                               
    Returns:
        2 lists: each list is a list of tuple-3 (cost, destination, path_to_destination)
    """
    num_vertices = len(graph)
    dist = [float('inf') for _ in range(num_vertices)]
    dist[start] = 0
    pre = [-1 for _ in range(num_vertices)]
    visited = [False for _ in range(num_vertices)]
    visited[start] = True
    
    obstacles = roottree + leaftree
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in range(num_vertices):
            if graph[u][v] == 1:
                if not visited[v]:
                    dist[v] = dist[u] + 1
                    visited[v] = True
                    pre[v] = u
                    if v in leaftree:
                        return wrap(dist, pre, roottree, leaftree)
                    if not v in obstacles:
                        queue.append(v)
    
    return wrap(dist, pre, roottree, leaftree)
