# How-to Guides

This section of the documentation provides practical how-to guides and example notebooks to get you up-and-running quickly in LabOne Q. They are intended to be a starting point to be further adapted by you for your experimental needs. Please reach out to us at <info@zhinst.com> if you'd like to discuss your experimental needs or have questions about how we can help you with LabOne Q.

<div class="grid cards" markdown>

-   __[Superconducting Qubits](sources/01_superconducting_qubits/index.md)__

    ---

    [![Transmon](images/icons/labone_q_transmon.svg){ width="100%" }](sources/01_superconducting_qubits/index.md)

-   __[Bosonic Qubits](sources/01a_bosonic_qubits/index.md)__

    ---

    [![Boson](images/icons/labone_q_boson.svg){ width="100%" }](sources/01a_bosonic_qubits/index.md)

-   __[Spin Qubits](sources/02_spin_qubits/index.md)__

    ---

    [![Spin](images/icons/labone_q_spin.svg){ width="100%" }](sources/02_spin_qubits/index.md)

-   __[Color Centers](sources/03_color_centers/index.md)__

    ---

    [![NV](images/icons/labone_q_nv.svg){ width="100%" }](sources/03_color_centers/index.md)

-   __[OpenQASM](sources/04_qasm/index.md)__

    ---

    ``` { .text .no-copy linenums="0" }
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    cx q[0], q[1];
    c = measure q;
    ```

</div>

!!! note
    Most of the examples are generated from Jupyter Notebook files and the source
    can be downloaded on each page by pressing the download button at the top right corner
    of the page.
