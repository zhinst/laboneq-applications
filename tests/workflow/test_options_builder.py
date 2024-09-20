import pytest

from laboneq_applications.workflow.options import TaskOptions, WorkflowOptions
from laboneq_applications.workflow.options_builder import (
    OptionBuilder,
    OptionInfo,
    OptionInfoList,
    _get_all_fields,
)


class TOpt1(TaskOptions):
    shared: int = 1


class TOpt2(TaskOptions):
    shared: int = 2
    b: int = 2


class OuterOptions(WorkflowOptions):
    # What if shared options exist at the top layer too
    shared: int = 1
    outer_not_shared: int = 1


class NestedOptions(WorkflowOptions):
    nested_not_shared: int = 3
    shared: int = 3


class TestOptionBuilder:
    @pytest.fixture()
    def option_builder(self):
        """
        The options created here follow this workflow structure:
        def workflow():
            task1()
            task2()
            nested_wf()

        def nested_wf():
            task1()
        """

        option = OuterOptions()
        nested_o = NestedOptions()
        nested_o.task_options = {"task1": TOpt1()}
        option.task_options = {
            "task1": TOpt1(),
            "task2": TOpt2(),
            "nested_wf": nested_o,
        }
        return OptionBuilder(option)

    def test_create(self):
        option = OuterOptions()
        builder = OptionBuilder(option)
        assert builder.base == option

    def test_read_options(self, option_builder):
        opt_info = OptionInfo("base", "outer_not_shared", option_builder.base)
        assert option_builder.outer_not_shared[0] == opt_info

        opt_infos = [
            OptionInfo("base", "shared", option_builder.base),
            OptionInfo("task1", "shared", option_builder.base.task_options["task1"]),
            OptionInfo("task2", "shared", option_builder.base.task_options["task2"]),
            OptionInfo(
                "nested_wf.base",
                "shared",
                option_builder.base.task_options["nested_wf"],
            ),
            OptionInfo(
                "nested_wf.task1",
                "shared",
                option_builder.base.task_options["nested_wf"].task_options["task1"],
            ),
        ]
        assert option_builder.shared == opt_infos

    def test_set_options(self, option_builder):
        # Set a field that is shared
        option_builder.shared(1234)
        assert option_builder.base.task_options == {
            "task1": TOpt1(shared=1234),
            "task2": TOpt2(shared=1234),
            "nested_wf": NestedOptions(
                shared=1234, task_options={"task1": TOpt1(shared=1234)}
            ),
        }
        assert option_builder.base.shared == 1234

        # Set a field that is nested but not shared
        option_builder.b(1111)
        assert option_builder.base.task_options == {
            "task1": TOpt1(shared=1234),
            "task2": TOpt2(shared=1234, b=1111),
            "nested_wf": NestedOptions(
                shared=1234, task_options={"task1": TOpt1(shared=1234)}
            ),
        }

    def test_set_options_with_slice(self, option_builder):
        option_builder.shared[1:3](1234)
        assert option_builder.base.task_options == {
            "task1": TOpt1(shared=1234),
            "task2": TOpt2(shared=1234),
            "nested_wf": NestedOptions(task_options={"task1": TOpt1()}),
        }

    def test_set_top_not_shared(self, option_builder):
        # Set a top level field that is not shared

        option_builder.outer_not_shared(1234)

        assert option_builder.base.outer_not_shared == 1234

    def test_set_nested_not_shared(self, option_builder):
        # Set a field of a nested option that is not shared
        option_builder.nested_not_shared(1234)
        assert option_builder.base.task_options == {
            "task1": TOpt1(),
            "task2": TOpt2(),
            "nested_wf": NestedOptions(
                nested_not_shared=1234, task_options={"task1": TOpt1()}
            ),
        }

    def test_access_non_existing(self, option_builder):
        with pytest.raises(
            AttributeError,
            match="'OptionBuilder' object has no attribute 'non_existing'",
        ):
            _ = option_builder.non_existing

    def test_set_non_existing(self, option_builder):
        # Setting a new field that is not defined in the options is possible.
        option_builder.non_existing = 1234

    def test_error_raised_when_setting_by_assigning(self, option_builder):
        with pytest.raises(
            TypeError,
            match="Setting options by assignment is not allowed. "
            "Please use the method call.",
        ):
            option_builder.shared = 1234

    def test_str(self, option_builder):
        assert (
            str(option_builder.shared)
            == "[(base,1), (task1,1), (task2,2), (nested_wf.base,3), "
            "(nested_wf.task1,1)]"
        )

    def test_dir(self, option_builder):
        assert dir(option_builder) == [
            "b",
            "logstore",
            "nested_not_shared",
            "outer_not_shared",
            "shared",
            "task_options",
        ]


class TestOptionInfo:
    def test_create(self):
        info = OptionInfo("task1", "shared", TOpt1())
        assert info.name == "task1"
        assert info.field == "shared"
        assert info.opt == TOpt1()

    def test_call(self):
        info = OptionInfo("task1", "shared", TOpt1())
        info(1234)
        assert info.opt.shared == 1234

    def test_str(self):
        info = OptionInfo("task1", "shared", TOpt1())
        assert str(info) == "(task1,1)"

    def test_equal(self):
        info1 = OptionInfo("task1", "shared", TOpt1())
        info2 = OptionInfo("task1", "shared", TOpt1())
        assert info1 == info2

        info1 = OptionInfo("task1", "shared", TOpt1())
        info2 = OptionInfo("task1", "shared", TOpt2())
        assert info1 != info2

        info1 = OptionInfo("task1", "shared", TOpt1())
        info2 = OptionInfo("task1", "shared", TOpt1())
        info2(1234)
        assert info1 != info2


class TestOptionInfoList:
    def test_create(self):
        opts_list = OptionInfoList()
        assert opts_list == []

        opts_list = OptionInfoList([OptionInfo("task1", "shared", TOpt1())])
        assert opts_list == [OptionInfo("task1", "shared", TOpt1())]

    def test_str(self):
        opts_list = OptionInfoList(
            [
                OptionInfo("task1", "shared", TOpt1()),
                OptionInfo("task2", "shared", TOpt1()),
            ]
        )
        assert str(opts_list) == "[(task1,1), (task2,1)]"

    def test_call(self):
        opts_list = OptionInfoList(
            [
                OptionInfo("task1", "shared", TOpt1()),
                OptionInfo("task2", "shared", TOpt1()),
            ]
        )
        opts_list(1234)
        assert opts_list[0].opt.shared == 1234
        assert opts_list[1].opt.shared == 1234

    def test_get(self):
        opts_list = OptionInfoList(
            [
                OptionInfo("task1", "shared", TOpt1()),
                OptionInfo("task2", "shared", TOpt1()),
                OptionInfo("task3", "shared", TOpt1()),
            ]
        )
        opts_list[0] = OptionInfo("task1", "shared", TOpt2())
        opts_list[0:2] = OptionInfoList(
            [
                OptionInfo("task1", "shared", TOpt2()),
                OptionInfo("task2", "shared", TOpt2()),
            ]
        )


def test_get_all_fields():
    class T(TaskOptions):
        shared: int = 1
        not_shared: int = 2

    class A(WorkflowOptions):
        alice: int = 1
        shared: int = 2

    a = A()
    assert set(_get_all_fields(a)) == {
        "alice",
        "shared",
        "logstore",
        "task_options",
    }

    class B(WorkflowOptions):
        bob: int = 1

    b = B()
    b.task_options = {
        "task1": TaskOptions(),
        "nested_wf": A(task_options={"task1": T()}),
    }
    assert _get_all_fields(b) == {
        "bob",
        "task_options",
        "alice",
        "shared",
        "not_shared",
        "logstore",
    }
