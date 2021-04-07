```
###########################################################################
##      __    __                                                         ##
##     |__|  |__|                                                        ##
##      __    __    __     Quantum networking experiments with NetSquid  ##
##   __|  |__|  |__|  |_                                                 ##
##  |__|  |__|  |_______|  (c) 2020 C. Cicconetti                        ##
##     |__|  |  |  |  |                                                  ##
##  _   _ _  |  |  |  |    Ubiquitous Internet @ IIT-CNR                 ##
## | | | | | |__|  |  |                                                  ##
## | |_| | |       |  |    https://github.com/ccicconetti/netsquid       ##
##  \___/|_|       |__|                                                  ##
##                                                                       ##
## Licensed under the MIT License <http://opensource.org/licenses/MIT>   ##
###########################################################################
```


# Introduction

[NetSquid](https://netsquid.org/) is an event-driven simulator of
networked quantum systems.

This repository contains a collection of tests, examples, and experiments
using NetSquid done as part of the research activities of the
[Ubiquitous Internet research group](http://cnd.iit.cnr.it/) of the
[Institute of Telematics and Informatics](https://www.iit.cnr.it/en) of the
[National Research Council](https://www.cnr.it/en).

# Structure

- `experiments`: experiments, one batch per directory, including the simulation scenarios, processing of results, and also some example plot generation scripts in Gnuplot; check the `README.md` within each directory for more information on each experiment
- `scripts`: miscellanea of scripts
- `uiiit`: extension of NetSquid that is used to run the experiments, also include non-NetSquid specific ancillary modules (e.g., for multi-process execution of simulations and post-processing of results); it also contains the unit tests in files starting with `test_`
- `Sandox`: random collection of experiments with NetSquid and its dependencies
- `Tutorial`: some of the examples provided by NetSquid revisited

# Code

The code assumes version 0.9.8 of Netsquid and has been tested
with Python 3.6.9 on Linux (Ubuntu 18.04) and Python 3.7.6 on Mac OS X.

## Dependencies

Python 3 must be used. It is suggested to use a virtual environment
so as to install all required packages with a matching version. For instance,
using [virtualenv](https://pypi.org/project/virtualenv/):

```
virtualenv -p python3 myvenv
(echo "PYTHONPATH=$PYTHONPATH:$PWD" ; echo "export PYTHONPATH") >> myvenv/bin/activate
source myvenv/bin/activate
```

The virtual environment will remain active in the shell from which the `source` command has been requested until logout. If you want to quit early the virtual environment without closing the shell you may run:

```
deactivate
```

To install all the dependencies at once using [pip](https://pypi.org/project/pip/) you have to first obtain a username and password for NetSquid by registering [here](https://forum.netsquid.org/ucp.php?mode=register), then issue the following command (you will be prompted for the username and password obtained):

```
pip install --extra-index-url https://pypi.netsquid.org -r requirements.txt
```

If no errors occurred `pip list` will include the packages just installed and you will be able to execute the examples and experiments, for instance:

```
cd Sandbox
python my_repeater_chain.py
```

## Unit tests

The unit tests use `unittest` Python library and can be executed by entering the `uiiit` directory and then running:

```
python -m unittest
```
