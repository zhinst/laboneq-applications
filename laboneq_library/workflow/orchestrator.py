"""Module for workflow orchestration."""
from queue import Queue


def sort_task_graph(graph: dict[int, list[int]]) -> list[int]:
    """Sort task topology.

    The function sorts the graph in the correct order.
    Does not support branching.

    Arguments:
        graph: A mapping where keys depicts a task and values
            are a list of its' dependencies.

    Returns:
        A topological order the tasks.

    Raises:
        ValueError: Graph contains a cycle
    """
    # TODO: Branching support
    # Khan's algorithm
    in_degree = {vertex: 0 for vertex in graph}
    for neighbors in graph.values():
        for neighbor in neighbors:
            in_degree[neighbor] += 1

    queue = Queue()
    for vertex, degree in in_degree.items():
        if degree == 0:
            queue.put(vertex)

    topological_order = []
    while not queue.empty():
        current_vertex = queue.get()
        topological_order.append(current_vertex)
        for neighbor in graph[current_vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.put(neighbor)

    if len(topological_order) != len(graph):
        msg = "The graph contains a cycle."
        raise ValueError(msg)

    topological_order.reverse()
    return topological_order
