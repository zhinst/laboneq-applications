import pytest

from laboneq_library.workflow.orchestrator import sort_task_graph


class TestSortTaskGraph:
    def test_no_input(self):
        graph = {}
        assert sort_task_graph(graph) == []

    def test_valid_graph(self):
        graph = {
            0: [],
            1: [0],
            2: [0],
            3: [2, 1],
        }
        assert sort_task_graph(graph) == [0, 1, 2, 3]

    def test_has_cycle(self):
        graph = {
            0: [],
            1: [2],
            2: [1],
        }
        with pytest.raises(ValueError, match="The graph contains a cycle"):
            sort_task_graph(graph)
