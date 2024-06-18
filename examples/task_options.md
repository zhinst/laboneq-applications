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

```python
try:
    chore({"foo": 10, "bar": "ge", "baz": "not a number"})
except ValidationError as e:
    print(e)
```

# Create a model option from BaseExperimentOptions and use it in a task

LaboneQ Applications provide a template options that can be used for creating a new option model for tasks that run L1Q experiments, `BaseExperimentOptions`.


```python
print(BaseExperimentOptions.model_fields)
```

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
