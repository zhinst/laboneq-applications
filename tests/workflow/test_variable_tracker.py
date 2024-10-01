from laboneq_applications.workflow import reference, variable_tracker


def test_variable_tracker():
    @variable_tracker.track
    def return_1():
        return 1

    @variable_tracker.track
    def return_2():
        return 2

    def big_function():
        a = return_1()
        a = return_2()
        a = return_1()  # noqa: F841

    tracker = variable_tracker.WorkflowFunctionVariableTracker()
    with variable_tracker.WorkflowFunctionVariableTrackerContext.scoped(tracker):
        big_function()

    assert len(tracker._variables) == 3
    assert len(tracker._variables["a"]) == 2

    first_def = tracker._variables["a"][0]
    assert reference.get_ref(first_def) is None
    assert reference.get_default(first_def) == 1
    assert len(first_def._overwrites) == 0

    second_def = tracker._variables["a"][1]
    assert reference.get_ref(second_def) is None
    assert reference.get_default(second_def) == 2
    assert reference.are_equal(second_def._overwrites[0], first_def)
