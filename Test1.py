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
from plotTest import plot
import activity as act
import copy

import Result
from lateral_inhibition import LateralDendriticInput



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


def Test(
    parameters,
    input_data,
    n=0,
    title="",
):
    net = Neocortex(dt=1)

    input_layer = NeuronGroup(
        net=net,
        size=parameters["input_size"],
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
                    ),  # 260
                    Fire(),  # 340
                    SpikeTrace(tau_s=parameters["tau_src"]),
                    NeuronAxon(),
                ]
            ),
            **{
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
        size=parameters["output_size"],
        tag="output_layer",
        behavior={
            **prioritize_behaviors(
                [
                    SimpleDendriteStructure(),
                    SimpleDendriteComputation(),
                    LIF(
                        R=OUT_R,
                        threshold=parameters["output_thresholds"],
                        tau=OUT_TAU,
                        v_reset=OUT_V_RESET,
                        v_rest=OUT_V_REST,
                    ),
                    Fire(),
                    SpikeTrace(tau_s=parameters["tau_dst"]),
                    NeuronAxon(),
                    *parameters["output_p_behaviors"],
                ]
            ),
            **parameters["output_behaviors"],
            **{361: Result.Result(print_details=False)},
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
                        weights=parameters["W"],
                    ),
                    SimpleDendriticInput(current_coef=parameters["j_0"]),
                ]
            ),
        },
    )

    sg_in_out.add_behavior(
        RECORDER_INDEX, Recorder(variables=["weights"], tag="sg_inp_out")
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
                    LateralDendriticInput(current_coef=parameters["j_0_lateral"]),
                ]
            )
        },
    )

    net.initialize(info=False)

    net.simulate_iterations(
        (parameters["duration"] + 10) * (n or parameters["output_size"])
    )
    output_layer.result
    plot(
        net=net,
        ngs=[input_layer, output_layer],
        title= title or parameters["title"],
        scaling_factor=parameters["scaling_factor"],
        env_recorder_index=EV_RECORDER_INDEX,
    )
