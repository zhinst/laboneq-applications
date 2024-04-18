from unittest.mock import Mock

import pytest

from laboneq_applications.workflow import Workflow, exceptions, task


@task
def addition(x, y) -> float:
    return x + y

@task
def substraction(x, y) -> float:
    return x - y


class TestTaskFunctionKeepsNormalBehaviour:
    def test_task_no_context(self):
        assert addition(1, 2) == 3
        assert substraction(1, 2) == -1

    def test_task_within_task(self):
        @task
        def substraction(x, y) -> float:
            return x - y

        @task
        def addition(x, y) -> float:
            assert substraction(x, y) == -1

        with Workflow() as wf:
            addition(1, 2)
        wf.run()


def test_workflow_single():
    with Workflow() as wf:
        addition(1, 2)
    result = wf.run()
    assert result.tasklog["addition"][0] == 3


class TestWorkFlowInput:
    def test_valid_input(self):
        with Workflow() as wf:
            addition(wf.input.kwargs["add_me"], y=wf.input.kwargs["add_me"])
        result = wf.run(add_me=5)
        assert result.tasklog["addition"][0] == 10

    def test_missing_input_key(self):
        with Workflow() as wf:
            addition(wf.input.args[0]["add_me"], y=1)

        with pytest.raises(KeyError) as err:
            wf.run({})
        assert str(err.value) == "'add_me'"

    def test_mapping_input(self):
        @task
        def work_on_mapping(result):
            return result["sum"] + 5 + result["value"]

        with Workflow() as wf:
            work_on_mapping(wf.input.args[0])

        result = wf.run({"sum": 5, "value": 1.2})
        assert result.tasklog["work_on_mapping"] == [5 + 5 + 1.2]

    def test_args_input(self):
        @task
        def work_on_mapping(x, y):
            return x, y

        with Workflow() as wf:
            work_on_mapping(wf.input.args[0], wf.input.args[0])

        result = wf.run(5, 10.5)
        assert result.tasklog["work_on_mapping"] == [(5, 5)]


class TestWorkFlowRerunNoCache:
    def test_workflow_rerun_input_change(self):
        with Workflow() as wf:
            addition(1, wf.input.args[0]["add_me"])
        result = wf.run({"add_me": 5})
        assert result.tasklog["addition"][0] == 6
        result = wf.run({"add_me": 6})
        assert result.tasklog["addition"][0] == 7

    def test_task_graph_up_executed_on_input_change(self):
        with Workflow() as wf:
            x = addition(1, wf.input.args[0]["add_me"])
            addition(1, x)

        result = wf.run({"add_me": 1})
        assert result.tasklog["addition"] == [2, 3]
        result = wf.run({"add_me": 2})
        assert result.tasklog["addition"] == [3, 4]

    def test_task_graph_down_executed_on_input_change(self):
        with Workflow() as wf:
            x = addition(1, 1)
            addition(x, wf.input.args[0]["add_me"])

        result = wf.run({"add_me": 1})
        assert result.tasklog["addition"] == [2, 3]
        result = wf.run({"add_me": 2})
        assert result.tasklog["addition"] == [2, 4]

    def test_rerun(self):
        mock_obj = Mock()

        @task
        def create_subject():
            return mock_obj()

        with Workflow() as wf:
            create_subject()

        wf.run()
        wf.run()
        wf.run()
        assert mock_obj.call_count == 3


class TestMultipleTasks:
    def test_each_call_produces_task(self):
        n_tasks = 10
        with Workflow() as wf:
            for _ in range(n_tasks):
                addition(1, 1)
        assert len(wf.tasks) == n_tasks

    def test_independent_tasks(self):
        with Workflow() as wf:
            addition(1, 1)
            substraction(3, 2)
        result = wf.run()
        assert result.tasklog["addition"] == [2]
        assert result.tasklog["substraction"] == [1]

    def test_context_passing_args(self):
        with Workflow() as wf:
            x = addition(1, 1)
            substraction(3, x)
        result = wf.run()
        assert result.tasklog["addition"] == [2]
        assert result.tasklog["substraction"] == [1]

    def test_context_passing_kwargs(self):
        with Workflow() as wf:
            y = addition(1, 1)
            substraction(3, y=y)
        result = wf.run()
        assert result.tasklog["addition"] == [2]
        assert result.tasklog["substraction"] == [1]

    def test_multiple_dependecy(self):
        with Workflow() as wf:
            x = addition(1, 1)
            substraction(3, x)
            addition(x, 5)
        result = wf.run()
        assert result.tasklog["addition"] == [2, 7]
        assert result.tasklog["substraction"] == [1]

    def test_nested_calls(self):
        with Workflow() as wf:
            substraction(addition(1, addition(0, 1)), addition(1, substraction(2, 1)))
        result = wf.run()
        assert result.tasklog["addition"] == [1, 2, 2]
        assert result.tasklog["substraction"] == [1, 0]

    def test_single_call_task_multiple_child_tasks(self):
        mock_obj = Mock()

        @task
        def create_subject():
            return mock_obj()

        def foobar(_):
            ...

        with Workflow() as wf:
            res = create_subject()
            foobar(res)
            foobar(res["a"])
            foobar(res["a"][0])

        wf.run()
        assert mock_obj.call_count == 1


class TestNestedWorkflows:
    def test_nested_workflow_definition(self):
        with Workflow():
            addition(1, 1)
            with pytest.raises(exceptions.WorkflowError) as err:
                with Workflow():
                    addition(1, 1)
        assert str(err.value) == "Nesting Workflows is not allowed."

    def test_executing_workflow_within_workflow(self):
        with Workflow() as wf1:
            addition(1, 2)

        with Workflow():
            addition(1, 1)
            with pytest.raises(exceptions.WorkflowError) as err:
                wf1.run()

        assert str(err.value) == "Nesting Workflows is not allowed."


class TestWorkflowPromises:
    def test_task_promise_getattr(self):
        @task
        def return_mapping():
            return {"a": 123, "b": [1, 2]}

        @task
        def act_on_mapping_value(a, b):
            return a, b

        with Workflow() as wf:
            result = return_mapping()
            act_on_mapping_value(result["a"], result["b"][1])
        assert wf.run().tasklog == {
            "return_mapping": [{"a": 123, "b": [1, 2]}],
            "act_on_mapping_value": [(123, 2)],
        }
