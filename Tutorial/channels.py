"""Example inspired from Netsquid's Modelling of network components tutorial:

https://docs.netsquid.org/latest-release/tutorial.components.html

Send a message on a channel, start simulation, then receive the message
and get the delay, which depends on the delay model set on that channel.

Note that the receive() method of a channel only returns the elements
that are available at the given simulation time: by calling it after the
simulation has started, we collect _all_ the messages ever sent with
the send() method.
"""

import numpy as np
import logging

import pydynaa
import netsquid as ns
from netsquid.components import Channel
from netsquid.components.models.delaymodels import (
    FixedDelayModel,
    GaussianDelayModel,
    FibreDelayModel
)
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.components.qchannel import QuantumChannel

def single_run(channel_model, run_id):
    # clear from previous run
    ns.sim_reset()

    rng = np.random.default_rng(seed=run_id)
    ns.set_random_state(seed=run_id)

    channel = Channel(name="ExampleChannel", length=3.)
    
    if channel_model == 'fixed_delay':
        fixed_model = FixedDelayModel(delay=10)
        channel.models['delay_model'] = fixed_model
    elif channel_model == 'gaussian_delay':
        gaussian_model = GaussianDelayModel(delay_mean=10, delay_std=1, rng=rng)
        channel.models['delay_model'] = gaussian_model
    elif channel_model == 'fibre':
        fibre_model = FibreDelayModel()
        fibre_model.properties['c'] = 3e8
        channel.models['delay_model'] = fibre_model
    else:
        raise Exception(f'unknown channel model {channel_model}')

    channel.send('hi')
    stats = ns.sim_run()

    __, delay = channel.receive()

    logging.info(stats)
    return delay

# configuration
num_repetitions = 10
channel_models = [
    'fixed_delay',
    'gaussian_delay',
    'fibre'
]
logging.basicConfig(level=logging.WARN)

for channel_model in channel_models:
    delays = []
    for run in range(num_repetitions):
        delays.append(single_run(channel_model, run))
    print('model {}, delays: {}'.format(
        channel_model,
        ','.join([f'{x:.2f}' for x in delays])
    ))