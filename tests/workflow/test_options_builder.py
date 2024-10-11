import pytest
from pydantic import Field

from laboneq_applications.workflow.options import TaskOptions, WorkflowOptions
from laboneq_applications.workflow.options_builder import (
    OptionBuilder,
    OptionNode,
    OptionNodeList,
    _get_all_fields,
)

from tests.helpers.format import minify_string, strip_ansi_codes


class TOpt1(TaskOptions):
    shared: int = 1


class TOpt2(TaskOptions):
    shared: int = 2
    b: int = 2


class OuterOptions(WorkflowOptions):
    shared: int = 1
    outer_not_shared: int = 1


class NestedOptions(WorkflowOptions):
    nested_shared: int = 3
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
        nested_o._task_options = {"task1": TOpt1()}
        option._task_options = {
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
        opt_info = OptionNode("base", "outer_not_shared", option_builder.base)
        assert option_builder.outer_not_shared[0] == opt_info

        opt_infos = [
            OptionNode("base", "shared", option_builder.base),
            OptionNode(
                "base.task1", "shared", option_builder.base._task_options["task1"]
            ),
            OptionNode(
                "base.task2", "shared", option_builder.base._task_options["task2"]
            ),
            OptionNode(
                "base.nested_wf",
                "shared",
                option_builder.base._task_options["nested_wf"],
            ),
            OptionNode(
                "base.nested_wf.task1",
                "shared",
                option_builder.base._task_options["nested_wf"]._task_options["task1"],
            ),
        ]
        assert option_builder.shared == opt_infos

    def test_access_non_existing(self, option_builder):
        with pytest.raises(
            AttributeError,
            match="'OptionBuilder' object has no attribute 'non_existing'",
        ):
            _ = option_builder.non_existing

    def test_dir(self, option_builder):
        assert dir(option_builder) == [
            "b",
            "logstore",
            "nested_shared",
            "outer_not_shared",
            "shared",
        ]


class TestOptionBuilderPrinting:
    @pytest.fixture()
    def option_builder(self):
        class W(WorkflowOptions):
            a: int = 1

        class T(TaskOptions):
            b: str = "b"

        option_builder = W()
        option_builder._task_options = {"task1": T(), "wf1": W()}
        return option_builder

    def test_str(self, option_builder):
        expected_str = (
            "W(a=1,_task_options={'task1':T(b='b'),'wf1':W(a=1,_task_options={})})"
        )
        assert expected_str == (strip_ansi_codes(minify_string(str(option_builder))))


class TestOptionBuilderOverview:
    @pytest.fixture()
    def option_builder(self):
        class W(WorkflowOptions):
            a: int = Field(1, description="a")

        class T(TaskOptions):
            b: str = "b"
            a: int = Field(2, description="aa")

        option_builder = W()
        option_builder._task_options = {"task1": T(), "wf1": W()}
        return OptionBuilder(option_builder)

    def test_overview(self, option_builder):
        overview = option_builder._overview()
        expected_overview = (
            "OptionFields============="
            "a:\tDescription:a\tClassesandDefaults:[('W',1)],"
            "\tDescription:aa\tClassesandDefaults:[('T',2)],"
            "b:\tDescription:None\tClassesandDefaults:[('T','b')],"
            "logstore:\tDescription:None\tClassesandDefaults:[('W',None)],"
        )
        assert expected_overview == strip_ansi_codes(minify_string(overview))


class TestOptionBuilderSetOption:
    """Separate out from TestOptionBuilder due to complexities
    of setting options.
    """

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
        nested_o._task_options = {"task1": TOpt1(), "nested_twice_o": NestedOptions()}
        option._task_options = {
            "task1": TOpt1(),
            "task2": TOpt2(),
            "nested_wf": nested_o,
        }
        return OptionBuilder(option)

    def test_set_options_by_task_name(self, option_builder):
        option_builder.shared(1234, "task1")
        nested_wf_opt = NestedOptions()
        nested_wf_opt._task_options = {
            "task1": TOpt1(),
            "nested_twice_o": NestedOptions(),
        }
        assert option_builder.base._task_options == {
            "task1": TOpt1(shared=1234),
            "task2": TOpt2(),
            "nested_wf": nested_wf_opt,
        }

    def test_set_options_top_level(self, option_builder):
        option_builder.shared(1234, ".")
        nested_wf_opt = NestedOptions(shared=1234)
        nested_wf_opt._task_options = {
            "task1": TOpt1(),
            "nested_twice_o": NestedOptions(),
        }
        assert option_builder.base.shared == 1234
        assert option_builder.base._task_options == {
            "task1": TOpt1(shared=1234),
            "task2": TOpt2(shared=1234),
            "nested_wf": nested_wf_opt,
        }

    def test_set_options_nested_workflow(self, option_builder):
        option_builder.shared(1234, "nested_wf")
        nested_wf_opt = NestedOptions(shared=1234)
        nested_wf_opt._task_options = {
            "task1": TOpt1(shared=1234),
            "nested_twice_o": NestedOptions(shared=1234),
        }
        assert option_builder.base._task_options == {
            "task1": TOpt1(),
            "task2": TOpt2(),
            "nested_wf": nested_wf_opt,
        }

    def test_set_options_by_nonexisting_task(self, option_builder):
        with pytest.raises(
            ValueError,
            match="Task or workflow non_existing not found to have the option shared.",
        ):
            option_builder.shared(111, "non_existing")

    def test_error_raised_when_setting_by_assigning(self, option_builder):
        with pytest.raises(
            TypeError,
            match="Setting options by assignment is not allowed. "
            "Please use the method call.",
        ):
            option_builder.shared = 1234

    def test_set_options(self, option_builder):
        # Set a field that is shared
        option_builder.shared(1234)
        nested_wf_opt = NestedOptions(shared=1234)
        nested_wf_opt._task_options = {
            "task1": TOpt1(shared=1234),
            "nested_twice_o": NestedOptions(shared=1234),
        }
        assert option_builder.base._task_options == {
            "task1": TOpt1(shared=1234),
            "task2": TOpt2(shared=1234),
            "nested_wf": nested_wf_opt,
        }
        assert option_builder.base.shared == 1234

        # Set a field that is nested but not shared
        option_builder.b(1111)
        assert option_builder.base._task_options == {
            "task1": TOpt1(shared=1234),
            "task2": TOpt2(shared=1234, b=1111),
            "nested_wf": nested_wf_opt,
        }

    def test_set_options_with_slice(self, option_builder):
        option_builder.shared[1:3](1234)
        nest_wf_opt = NestedOptions()
        nest_wf_opt._task_options = {
            "task1": TOpt1(),
            "nested_twice_o": NestedOptions(),
        }
        assert option_builder.base._task_options == {
            "task1": TOpt1(shared=1234),
            "task2": TOpt2(shared=1234),
            "nested_wf": nest_wf_opt,
        }

    def test_set_top_not_shared(self, option_builder):
        # Set a top level field that is not shared

        option_builder.outer_not_shared(1234)

        assert option_builder.base.outer_not_shared == 1234

    def test_set_nested_shared(self, option_builder):
        # Set a field of a nested option that is not shared
        option_builder.nested_shared(1234)
        nest_wf_opt = NestedOptions(nested_shared=1234)
        nest_wf_opt._task_options = {
            "task1": TOpt1(),
            "nested_twice_o": NestedOptions(nested_shared=1234),
        }

        assert option_builder.base._task_options == {
            "task1": TOpt1(),
            "task2": TOpt2(),
            "nested_wf": nest_wf_opt,
        }


class TestOptionNode:
    def test_create(self):
        info = OptionNode("task1", "shared", TOpt1())
        assert info.name == "task1"
        assert info.field == "shared"
        assert info.option == TOpt1()

    def test_call(self):
        info = OptionNode("task1", "shared", TOpt1())
        info(1234)
        assert info.option.shared == 1234

    def test_str(self):
        info = OptionNode("task1", "shared", TOpt1())
        assert str(info) == "task1 : 1"

    def test_equal(self):
        info1 = OptionNode("task1", "shared", TOpt1())
        info2 = OptionNode("task1", "shared", TOpt1())
        assert info1 == info2

        info1 = OptionNode("task1", "shared", TOpt1())
        info2 = OptionNode("task1", "shared", TOpt2())
        assert info1 != info2

        info1 = OptionNode("task1", "shared", TOpt1())
        info2 = OptionNode("task1", "shared", TOpt1())
        info2(1234)
        assert info1 != info2


class TestOptionNodeList:
    def test_create(self):
        opts_list = OptionNodeList()
        assert opts_list == []

        opts_list = OptionNodeList([OptionNode("task1", "shared", TOpt1())])
        assert opts_list == [OptionNode("task1", "shared", TOpt1())]

    def test_call(self):
        opts_list = OptionNodeList(
            [
                OptionNode("task1", "shared", TOpt1()),
                OptionNode("task2", "shared", TOpt1()),
            ]
        )
        opts_list(1234)
        assert opts_list[0].option.shared == 1234
        assert opts_list[1].option.shared == 1234

    def test_get(self):
        opts_list = OptionNodeList(
            [
                OptionNode("task1", "shared", TOpt1()),
                OptionNode("task2", "shared", TOpt1()),
                OptionNode("task3", "shared", TOpt1()),
            ]
        )
        opts_list[0] = OptionNode("task1", "shared", TOpt2())
        opts_list[0:2] = OptionNodeList(
            [
                OptionNode("task1", "shared", TOpt2()),
                OptionNode("task2", "shared", TOpt2()),
            ]
        )

    def test_str(self):
        opts_list = OptionNodeList(
            [
                OptionNode("base", "shared", TOpt1()),
                OptionNode("base.task1", "shared", TOpt1()),
                OptionNode("base.inner_wf.task1", "shared", TOpt1()),
            ]
        )
        expected_output = (
            "---------------------------------------------------------------\n"
            "Task/Workflow                                      | Value     \n"
            "---------------------------------------------------------------\n"
            ".                                                  | 1         \n"
            "task1                                              | 1         \n"
            "inner_wf.task1                                     | 1         \n"
            "---------------------------------------------------------------"
        )
        assert str(opts_list) == expected_output


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
    }

    class B(WorkflowOptions):
        bob: int = 1

    b = B()
    nested_wf = A()
    nested_wf._task_options = {"task1": T()}
    b._task_options = {
        "task1": TaskOptions(),
        "nested_wf": nested_wf,
    }
    assert _get_all_fields(b) == {
        "bob",
        "alice",
        "shared",
        "not_shared",
        "logstore",
    }
