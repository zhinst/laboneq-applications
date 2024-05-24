```python
from typing import Literal

from pydantic import ValidationError

from laboneq_applications.core.options import (
    BaseExperimentOptions,
    BaseOptions,
    create_validate_opts,
)
from laboneq_applications.workflow import task
```

# Introduction

Workflow and tasks often require additional options flags for configuration. 
LabOneQ-Applications provides utility function `create_validate_opts` to create an option model as a template for validation and organizing these options. 
In addition, we provide `BaseExperimentOptions` as a base option model that can be used for all tasks that run L1Q experiments.

# Create a model option from scratch


```python
class ExampleOptions(BaseOptions):
    """Example options."""

    foo: int
    bar: str


input_options = {
    "foo": 10,
    "bar": "ge",
}

opt = create_validate_opts(input_options, base=ExampleOptions)
```

Inputs that have wrong types or values can be caught early by using a model option.


```python
input_options = {
    "foo": "not a number",
    "bar": 1,  # supposed to be a string
}
try:
    opt = create_validate_opts(input_options, base=ExampleOptions)
except ValidationError as e:
    print(e)
```

    2 validation errors for option_model
    foo
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='not a number', input_type=str]
        For further information visit https://errors.pydantic.dev/2.7/v/int_parsing
    bar
      Input should be a valid string [type=string_type, input_value=1, input_type=int]
        For further information visit https://errors.pydantic.dev/2.7/v/string_type


# Extend the base model on the fly


```python
class ExampleOptions(BaseOptions):
    """Example options."""

    foo: int
    bar: str


custom_options = {
    "fred": (int, ...),
    "default_fed": (str, "fed"),
}
# fred is required and must be an integer
# default_fed is optional and must be a string, with a default value of "fed"

options = {
    "foo": 10,
    "bar": "ge",
    "fred": 20,
}

opt = create_validate_opts(options, custom_options, base=ExampleOptions)
```




    option_model(foo=10, bar='ge', fred=20, default_fed='fed')



# Override the base model.


```python
class ExampleOptions(BaseOptions):
    """Example options."""

    foo: int
    bar: str


custom_options = {
    "foo": (int, 1),
}
# foo is now not required and defaults to 1

options = {
    "bar": "ge",
}

opt = create_validate_opts(options, custom_options, base=ExampleOptions)
```




    option_model(foo=1, bar='ge')



# Use options inside a task

Let's say we have a task that requires a few options to be set: `foo`, `bar`, and `baz`. 


```python
@task
def chore(options: dict):
    """Example task."""
    custom_options = {
        "baz": (int, ...),
    }
    opts = create_validate_opts(options, custom_options, base=ExampleOptions)
    print(opts)
```


```python
chore({"foo": 10, "bar": "ge", "baz": 20})
```

    foo=10 bar='ge' baz=20



```python
try:
    chore({"foo": 10, "bar": "ge", "baz": "not a number"})
except ValidationError as e:
    print(e)
```

    1 validation error for option_model
    baz
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='not a number', input_type=str]
        For further information visit https://errors.pydantic.dev/2.7/v/int_parsing


# Create a model option from BaseExperimentOptions and use it in a task

LaboneQ Applications provide a template options that can be used for creating a new option model for tasks that run L1Q experiments, `BaseExperimentOptions`.


```python
print(BaseExperimentOptions.model_fields)
```

    {'count': FieldInfo(annotation=int, required=True, metadata=[Ge(ge=0)]), 'acquisition_type': FieldInfo(annotation=Union[str, AcquisitionType], required=False, default=AcquisitionType.INTEGRATION), 'averaging_mode': FieldInfo(annotation=Union[str, AveragingMode], required=False, default=AveragingMode.CYCLIC), 'repetition_mode': FieldInfo(annotation=Union[str, RepetitionMode], required=False, default=RepetitionMode.FASTEST), 'repetition_time': FieldInfo(annotation=Union[float, NoneType], required=False, default=None), 'reset_oscillator_phase': FieldInfo(annotation=bool, required=False, default=False)}



```python
@task
def rabi(options: dict):
    """Run a Rabi experiment with the given options."""
    # rabi require extra options such as transition
    custom_options = {"transition": (Literal["ge", "ef"], "ge")}
    opts = create_validate_opts(options, custom_options, base=BaseExperimentOptions)
    print(opts)
    # Run experiment with these options
```


```python
options = {
    "count": 10,
    "transition": "ef",
    "acquisition_type": "integration_trigger",
    "averaging_mode": "cyclic",
    "repetition_mod": "fastest",
}
rabi(options)
```

    count=10 acquisition_type=AcquisitionType.INTEGRATION averaging_mode=AveragingMode.CYCLIC repetition_mode=RepetitionMode.FASTEST repetition_time=None reset_oscillator_phase=False transition='ef'

