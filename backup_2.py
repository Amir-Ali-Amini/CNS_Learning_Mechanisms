# %%
from pymonntorch import (
    NeuronGroup,
    SynapseGroup,
    Recorder,
    EventRecorder,
)

from conex import (
    Neocortex,
    prioritize_behaviors,
)

from conex.behaviors.neurons import (
    SimpleDendriteStructure,
    SimpleDendriteComputation,
    LIF,
    SpikeTrace,
    NeuronAxon,
    Fire,
    KWTA,
)
from conex.behaviors.synapses import (
    SynapseInit,
    WeightInitializer,
    SimpleDendriticInput,
    SimpleSTDP,
    WeightNormalization,
    WeightClip,
    LateralDendriticInput,
)

import torch

import InputData
from plot import plot
import activity as act


RECORDER_INDEX = 460
EV_RECORDER_INDEX = 461

OUT_R = 10
OUT_THRESHOLD = 15
OUT_TAU = 3
OUT_V_RESET = 0
OUT_V_REST = 5
OUT_TRACE_TAU = 10.0
# OUT_R = 10
# OUT_THRESHOLD = -55
# OUT_TAU = 3
# OUT_V_RESET = -70
# OUT_V_REST = -65
# OUT_TRACE_TAU = 1.0


def A(
    size=100,
    frac=0.5,
    mean1=100,
    mean2=200,
):
    n = int(size * frac)
    data = torch.zeros((2, size))
    data[0][:n] = (2 + torch.randn(n)) * mean1
    data[1][size - n :] = (2 + torch.randn(n)) * mean2
    return data


net = Neocortex(dt=1)

input_layer = NeuronGroup(
    net=net,
    size=100,
    tag="input_layer",
    behavior={
        **prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(
                    R=OUT_R,
                    threshold=OUT_THRESHOLD,
                    tau=OUT_TAU,
                    v_reset=OUT_V_RESET,
                    v_rest=OUT_V_REST,
                    # init_v=torch.randn(100) - 65,
                ),  # 260
                Fire(),  # 340
                SpikeTrace(tau_s=OUT_TRACE_TAU),
                NeuronAxon(),
            ]
        ),
        **{
            345: InputData.Encode(
                data=A(),
                time=40,
                method="poisson",
            ),
            350: act.Activity(),
            RECORDER_INDEX: Recorder(
                variables=["v", "I", "T"],
                tag="in_recorder",
            ),
            EV_RECORDER_INDEX: EventRecorder("spikes", tag="in_ev_recorder"),
        },
    },
)
output_layer = NeuronGroup(
    net=net,
    size=2,
    tag="output_layer",
    behavior={
        **prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(
                    R=OUT_R,
                    threshold=OUT_THRESHOLD,
                    tau=OUT_TAU,
                    v_reset=OUT_V_RESET,
                    v_rest=OUT_V_REST,
                ),
                Fire(),
                SpikeTrace(tau_s=OUT_TRACE_TAU),
                NeuronAxon(),
                KWTA(k=1),
            ]
        ),
        **{
            350: act.Activity(),
            RECORDER_INDEX: Recorder(
                variables=["v", "I", "T"],
                tag="out_recorder",
            ),
            EV_RECORDER_INDEX: EventRecorder("spikes", tag="out_ev_recorder"),
        },
    },
)


sg_in_out = SynapseGroup(
    net=net,
    src=input_layer,
    dst=output_layer,
    tag="Proximal,sg_in_out",
    behavior=prioritize_behaviors(
        [
            SynapseInit(),
            WeightInitializer(
                mode="random",
            ),
            SimpleDendriticInput(current_coef=15),
            SimpleSTDP(
                a_plus=0.01,
                a_minus=0.005,
                positive_bound="soft_bound",
                negative_bound="soft_bound",
            ),
            WeightNormalization(),
            # WeightClip(),
            # LateralDendriticInput(),
        ]
    ),
)

sg_in_out.add_behavior(
    RECORDER_INDEX, Recorder(variables=["weights"], tag="sg_inp_out")
)

net.initialize(info=False)

net.simulate_iterations(3000)

plot(
    net=net,
    ngs=[input_layer, output_layer],
    scaling_factor=8,
    sgs=[sg_in_out],
    recorder_index=RECORDER_INDEX,
    env_recorder_index=EV_RECORDER_INDEX,
)

# %%
