# How-to Guides - Spin Qubits

Here, you'll find a collection of notebooks showcasing the use of the LabOne Q software framework in gradually increasing steps of complexity, with content focused towards the needs of spin qubit experiments. The first few examples are very general and are intended to be adapted to your own use case, while later examples demonstrate how LabOne Q can be used for a specific type of measurement. Depending on your specific use case and architecture, you will likely wish to modify these experiments and adapt them to your own workflow. Please get in touch at <info@zhinst.com> and we will be happy discuss your application.

## Sweeping a MFLI Lock-in Amplifier using a call function

This [notebook](00_neartime_callback_sweeps.ipynb) uses the call function to demonstrate a 2D sweep. Two axes are swept with call-back functions and, at each point of the sweep, the data is acquired with the MFLI DAQ module.

## Sweeping QCoDes parameters in LabOne Q

This [notebook](01_QCoDeS_sweeps.ipynb) is similar to the previous example. Here, you'll perform a 2D sweep, where the two sweep axes are set through a QCoDeS parameter, mimicking arbitrary instruments that can be controlled with a QCoDeS driver.

## Sequential Ramsey with HDAWG and CW Acquisition with MFLI

In this [notebook](02_MFLI_cw_acquisition.ipynb), you'll learn how to perform a Ramsey sequence in sequential mode with the HDAWG, and, at each iteration of the sweep, send a trigger from the HDAWG to the MFLI DAQ module for data acquisition. This is useful for CW experiments when integrating with a Lock-in amplifier, e.g., for transport readout of spin qubits.

## Pulsed acquisition with the UHFLI

[Here](03_UHFLI_pulsed_acquisition.ipynb), you'll use the HDAWG to perform a Ramsey sequence and send triggers to the UHFLI, which is used to do the acquisition. One trigger from the HDAWG is sent to the UHFLI DAQ module for data acquisition. A second trigger is sent to the UHFLI demodulator to gate the data transfer and enable fast measurements. For each shot of the experiment, a time trace is acquired with the UHFLI. For this example, to imitate the readout signal, a Gaussian pulse is played and acquired with the UHFLI

## Pulse sequences played with the HDAWG

In the [HDAWG pulse sequence notebook](04_HDAWG_pulse_sequences.ipynb), you'll use a HDAWG to demonstrate pulse sequences useful in various experiments. The pulse sequences are general with additions geared towards a typical spin qubit experiment by adding gate pulsing for control between Coulomb and spin blockade. The included sequences are:

* Rabi: length sweep of burst
* Ramsey 1: sweep delay with constant burst duration
* Ramsey 2: sweep burst duration at constant delay
* Ramsey 3: sweep phase of second burst and delay between bursts

