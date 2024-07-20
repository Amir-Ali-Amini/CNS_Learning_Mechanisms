"""
Dendritic behaviors.
"""

from pymonntorch import Behavior

import torch
import torch.nn.functional as F
from conex.behaviors.synapses.dendrites import BaseDendriticInput


class LateralDendriticInput(BaseDendriticInput):

    def __init__(self, *args, current_coef=1, inhibitory=None, **kwargs):
        super().__init__(
            *args, current_coef=current_coef, inhibitory=inhibitory, **kwargs
        )

    def initialize(self, synapse):
        super().initialize(synapse)

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay).to(
            self.def_dtype
        )
        number_of_spikes = spikes.sum()
        return (spikes - 1) * number_of_spikes
