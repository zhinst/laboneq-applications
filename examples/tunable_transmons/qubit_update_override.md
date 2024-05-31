```python
from laboneq_applications.tasks import update_qubits
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubit,
    modify_qubits,
    modify_qubits_context,
)
from laboneq_applications.workflow.workflow import Workflow
```

LaboneQ Application library supports the following data manipulation on TunableTransmons objects:
- Temporary replacement of parameters in qubit objects
- Update of qubit parameters

# Temporary replacement of parameters in qubit objects

## Replace parameters of a single qubit


```python
q0 = TunableTransmonQubit()
```


```python
params_to_replace_0 = {
    "reset_delay_length": 2e-6,
    "drive_parameters_ge": {
        "amplitude_pi": 2,
        "amplitude_pi2": 1,
        "length": 5e-9,
        "pulse": {"function": "drag", "beta": 0, "sigma": 0.25},
    },
    "drive_parameters_ef.amplitude_pi": 2,
}
q0_copied = q0.replace(params_to_replace_0)
```

The above example shows three supported scenarios in which parameters can be replaced.
1. Replace a top level attribute of a qubit object
2. Replace a nested attribute of a qubit object
3. Partially replace a nested attribute of a qubit object


## Replace parameters of multiple qubits


```python
q1 = TunableTransmonQubit()
params_to_replace_1 = {"reset_delay_length": 1e-6}
```


```python
q0_temp, q1_temp = modify_qubits([(q0, params_to_replace_0), (q1, params_to_replace_1)])
```

## Replace parameters using a context


```python
with modify_qubits_context(
    [(q0, params_to_replace_0), (q1, params_to_replace_1)]
) as qubits_temp:
    q0_temp, q1_temp = qubits_temp
    # do something with thew new q0 and q1
```

# Update parameters of qubits

Beside temporary replacement of parameters, the library also supports updating qubit parameters.
 
This is particularly useful in tune-up scenarios where the qubit parameters are updated based on the measurement results of previous tune-up steps.

Similarly to the temporary replacement of parameters, we also support updating qubit parameters partially.

## Single qubit update


```python
q0 = TunableTransmonQubit()
q0.update({"reset_delay_length": 1.2e-6})
q0.update({"drive_parameters_ge.amplitude_pi": 0.2})
```


```python
print(q0.parameters.reset_delay_length)
print(q0.parameters.drive_parameters_ge["amplitude_pi"])
```

    1.2e-06
    0.2


## Use tasks to update qubit parameters

Task `update_qubits` was provided specifically for updating qubits parameters in a `workflow` or a `taskbook`


```python
q0 = TunableTransmonQubit()
q1 = TunableTransmonQubit()
with Workflow() as wf:
    update_qubits(
        [(q0, {"reset_delay_length": 1.2e-6}), (q1, {"reset_delay_length": 2.2e-6})],
    )
```


```python
run = wf.run()
```


```python
print(q0.parameters.reset_delay_length)
print(q1.parameters.reset_delay_length)
```

    1.2e-06
    2.2e-06



```python

```
