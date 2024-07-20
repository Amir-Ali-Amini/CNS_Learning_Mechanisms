from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.spatial.distance import cosine


def memoized_number():
    dic = [0]

    def inside():
        dic[0] += 1
        return f"plot({dic[0]}):"

    return inside


def plot(
    net,
    title=None,
    ngs=[],
    scaling_factor=3,
    label_font_size=6,
    env_recorder_index=10,
):
    n = len(ngs)
    fig_counter = memoized_number()
    fig, axd = plt.subplot_mosaic(
        """
        B
        C
        """,
        layout="constrained",
        # "image" will contain a square image. We fine-tune the width so that
        # there is no excess horizontal or vertical margin around the image.
        figsize=(12 * scaling_factor, 3 * scaling_factor),
    )

    fig.suptitle(
        title or "Plot",
        fontsize=(label_font_size + 7) * scaling_factor,
        fontweight="bold",
    )

    axd["B"].scatter(
        ngs[0][env_recorder_index, 0].variables["spikes"][:, 0].cpu(),
        ngs[0][env_recorder_index, 0].variables["spikes"][:, 1].cpu(),
        c=(
            ngs[0][env_recorder_index, 0].variables["spikes"][:, 0].cpu()
            // ngs[0].network.input_period
        )
        %ngs[0].network.number_of_inputs,
        # s=5000 // ngs[1].size,
        vmin=0,
        vmax=ngs[1].size,
        label=f"{ngs[0].tag}",
    )
    axd["B"].set_ylabel(
        "spikes (neuron number)", fontsize=label_font_size * scaling_factor
    )
    axd["B"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["B"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["B"].set_xlim(0, net.iteration)
    axd["B"].set_ylim(-1, ngs[0].size)

    # axd["B"].set_title(
    #     "1st Neuron Group", fontsize=(label_font_size + 1) * scaling_factor
    # )
    axd["B"].set_xlabel(
        f"{fig_counter()} time ({ngs[0].tag}) - \"A-N: Assigned Neuron(s)\"",
        fontsize=label_font_size * scaling_factor,
    )
    temp = [0] * (len(ngs[1].result) + ngs[1].size)
    for i in ngs[1].result:
        if i != -1 and i[0].shape[0]!=ngs[1].size:
            for j in i[0]:
                temp[j] += 1
    for i in range(len(ngs[1].result)):
        B_result= [str(int(t)) for t in ngs[1].result[i][0]]
        B_text_result = ", ".join(B_result) if len(B_result)!=ngs[1].size else "None"
        axd["B"].text(
            ngs[0].network.input_period * i + ngs[0].network.input_period // 2,
            (
                int(ngs[0].size  * 0.8)
                if i < (ngs[0].network.number_of_inputs // 2)
                else int(ngs[0].size * 0.1)
            ),
            "A-N(s):\n" +  B_text_result,
            fontsize=int(
                label_font_size
                * scaling_factor
                * (1 if ngs[1].size < 17 else 0.6)
            ),
            horizontalalignment="center",
            color="black",
            fontweight="bold",
        )

    for i in range(ngs[1].size):
        axd["C"].text(
            0,
            i,
            f"assiged: {temp[i]}",
            fontsize=int(
                label_font_size
                * scaling_factor
                * 0.5
                * (1 if ngs[1].size < 15 else 0.6)
            ),
            color="black",
            fontweight="bold",
        )

    axd["C"].scatter(
        ngs[1][env_recorder_index, 0].variables["spikes"][:, 0].cpu(),
        ngs[1][env_recorder_index, 0].variables["spikes"][:, 1].cpu(),
        # c=ngs[1][env_recorder_index, 0].variables["spikes"][:, 1].cpu(),
        c=(
            ngs[1][env_recorder_index, 0].variables["spikes"][:, 0].cpu()
            // ngs[1].network.input_period
        )
        % ngs[1].network.number_of_inputs,
        # s=2000 // ngs[1].size if ngs[1].size < 16 else 16,
        s=5000 // ngs[1].network.number_of_inputs,
        vmin=0,
        vmax=ngs[1].size,
        label=f"{ngs[1].tag}",
    )
    axd["C"].set_ylabel(
        "spikes (neuron number)", fontsize=label_font_size * scaling_factor
    )
    axd["C"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["C"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["C"].set_xlim(0, net.iteration)
    axd["C"].set_ylim(-1, ngs[1].size)
    # axd["C"].set_title(
    #     "2nd Neuron Group", fontsize=(label_font_size + 1) * scaling_factor
    # )
    axd["C"].set_xlabel(
        f"{fig_counter()} time ({ngs[1].tag})",
        fontsize=label_font_size * scaling_factor,
    )

    for step in range(0, ngs[0].network.iteration, ngs[0].network.input_period):
        axd["B"].axvline(x=step, color="grey", linestyle="--", linewidth=2)
        axd["C"].axvline(x=step, color="grey", linestyle="--", linewidth=2)

    for step in range(
        0, ngs[0].network.iteration, ngs[0].network.input_period * ngs[1].size
    ):
        axd["B"].axvline(x=step, color="black", linestyle="--", linewidth=3)
        axd["C"].axvline(x=step, color="black", linestyle="--", linewidth=3)

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Show the plot
    fig.show()
