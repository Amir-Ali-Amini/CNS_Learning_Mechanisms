# %%
from pymonntorch import (
    NeuronGroup,
    SynapseGroup,
    Recorder,
    EventRecorder,
    NeuronDimension,
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
)

import torch

import InputData
from plot import plot
import activity as act
import copy
import Test1 as Test
from lateral_inhibition import LateralDendriticInput

RECORDER_INDEX = 460
EV_RECORDER_INDEX = 461

OUT_R = 10
OUT_THRESHOLD = 15
OUT_TAU = 3
OUT_V_RESET = 0
OUT_V_REST = 5

# OUT_R = 10
# OUT_THRESHOLD = -55
# OUT_TAU = 3
# OUT_V_RESET = -70
# OUT_V_REST = -65
# OUT_TRACE_TAU = 1.0


def simulate(
    input_data,
    title="",
    in_size=None,
    out_size=2,
    j_0=20,
    AP=0.01,
    AM=0.005,
    tau_src=3,
    tau_dst=2,
    iteration=3000,
    plot_scale=0,
    output_p_behaviors=[
        # KWTA(k=2)
    ],
    output_behaviors={
        350: act.Activity(),
        RECORDER_INDEX: Recorder(
            variables=["v", "I", "T"],
            tag="out_recorder",
        ),
        EV_RECORDER_INDEX: EventRecorder("spikes", tag="out_ev_recorder"),
    },
    synapse_behaviors={},
    synapse_p_behaviors=[
        WeightNormalization(),
        # WeightClip(),
        # LateralDendriticInput(),
    ],
):
    net = Neocortex(dt=1)

    input_layer = NeuronGroup(
        net=net,
        size=NeuronDimension(
            height=(in_size or len(input_data.init_kwargs["data"][0]))
        ),
        tag="input_layer",
        behavior={
            **prioritize_behaviors(
                [
                    SimpleDendriteStructure(),
                    SimpleDendriteComputation(),
                    LIF(
                        R=OUT_R,
                        threshold=torch.tensor(
                            [OUT_THRESHOLD]
                            * (in_size or len(input_data.init_kwargs["data"][0]))
                        ),
                        tau=OUT_TAU,
                        v_reset=OUT_V_RESET,
                        v_rest=OUT_V_REST,
                    ),  # 260
                    Fire(),  # 340
                    SpikeTrace(tau_s=tau_src),
                    NeuronAxon(),
                ]
            ),
            **{
                10: InputData.ResetMemory(),
                345: input_data,
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
        size=NeuronDimension(height=out_size),
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
                    SpikeTrace(tau_s=tau_dst),
                    NeuronAxon(),
                    *output_p_behaviors,
                ]
            ),
            **output_behaviors,
        },
    )

    sg_in_out = SynapseGroup(
        net=net,
        src=input_layer,
        dst=output_layer,
        tag="Proximal,sg_in_out",
        behavior={
            **prioritize_behaviors(
                [
                    SynapseInit(),
                    WeightInitializer(
                        mode="random",
                    ),
                    SimpleDendriticInput(current_coef=j_0),
                    SimpleSTDP(
                        a_plus=AP,
                        a_minus=AM,
                        positive_bound="soft_bound",
                        negative_bound="soft_bound",
                    ),
                    *synapse_p_behaviors,
                ]
            ),
            **synapse_behaviors,
        },
    )
    sg_lateral = SynapseGroup(
        net=net,
        src=output_layer,
        dst=output_layer,
        tag="Proximal,sg_lateral",
        behavior={
            **prioritize_behaviors(
                [
                    SynapseInit(),
                    WeightInitializer(
                        mode="random",
                    ),
                    LateralDendriticInput(current_coef=j_0),
                ]
            )
        },
    )

    sg_in_out.add_behavior(
        RECORDER_INDEX, Recorder(variables=["weights"], tag="sg_inp_out")
    )

    net.initialize(info=False)

    net.simulate_iterations(iteration)

    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print(output_layer.threshold)
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("------------------------------------------------------")

    import matplotlib.pyplot as plt

    plt.plot(output_layer[460, 0].variables["v"])
    plt.show()

    plot(
        net=net,
        ngs=[input_layer, output_layer],
        scaling_factor=5,
        sgs=[sg_in_out],
        title=title,
        recorder_index=RECORDER_INDEX,
        env_recorder_index=EV_RECORDER_INDEX,
    )
    # print(sg_in_out.weights.shape, "----------- shape --------------")
    return net, {
        "W": sg_in_out.weights,
        "input_size": in_size or len(input_data.init_kwargs["data"][0]),
        "input_data": copy.deepcopy(input_data.init_kwargs),
        "duration": input_data.init_kwargs["time"],
        "output_size": out_size,
        "tau_src": tau_src,
        "tau_dst": tau_dst,
        "AM": AM,
        "AP": AP,
        "j_0": j_0,
        "output_behaviors": output_behaviors,
        "synapse_behaviors": synapse_behaviors,
        "output_p_behaviors": output_p_behaviors,
        "synapse_p_behaviors": synapse_p_behaviors,
        "title": title,
        "scaling_factor": plot_scale
        or (3 + len(input_data.init_kwargs["data"][0]) / 40),
    }


def A(
    size=100,
    frac=0.5,
    j0=20,
    mean1=100,
    mean2=110,
    iteration=2000,
    time=50,
    AP=0.01,
    AM=0.005,
    title="",
    test=False,
    tau_src=3,
    tau_dst=2,
):
    n = int(size * frac)
    data = torch.zeros((2, size))
    data[0][:n] = (2 + torch.randn(n)) * mean1
    data[1][size - n :] = (2 + torch.randn(n)) * mean2
    net, parameters = simulate(
        iteration=iteration,
        out_size=2,
        input_data=InputData.Encode(
            data=data,
            # range=255,
            time=time,
            method="poisson",
        ),
        title=title or "STDP learning with Poisson_encoding for input data",
        tau_src=tau_src,
        tau_dst=tau_dst,
        AP=AP,
        AM=AM,
        j_0=j0 * (0.5 / frac),
    )
    if test:
        print(parameters["W"].max(), parameters["W"].min())
        Test.Test(
            input_data=InputData.Encode(
                data=data,
                epsilon=0,
                # range=255,
                time=time,
                method="poisson",
            ),
            title="STDP Test Result",
            parameters=parameters,
        )


A()
# %%
# %%
A(
    frac=0.7,
)
# %%
A(
    frac=0.9,
)
# %%
A(
    frac=1,
)

# %%
