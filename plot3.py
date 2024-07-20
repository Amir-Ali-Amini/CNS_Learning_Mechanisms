from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.spatial.distance import cosine
import calculate_learning_convergence


def memoized_number():
    dic = [0]

    def inside():
        dic[0] += 1
        return f"plot({dic[0]}):"

    return inside


def format_parameters(inp_parameters, num_columns):
    """
    Formats a list of strings into a table-like string with specified number of columns.

    Args:
    parameters (list of str): List of parameter strings.
    num_columns (int): Number of columns in the table.

    Returns:
    str: Formatted table-like string.
    """
    max_len = max(len(param) for param in inp_parameters) + 2  # +2 for padding
    if max_len > 30:num_columns = 2

    parameters = inp_parameters + [
        "" for _ in range(num_columns - ((len(inp_parameters) % num_columns) or num_columns))
    ]

    rows = (len(parameters) + num_columns - 1) // num_columns  # ceiling division
    table_str = ""

    for r in range(rows):
        row_params = parameters[r::rows]  # Get every nth element starting from r
        row_str = " | ".join(param.ljust(max_len) for param in row_params)
        table_str += f" {row_str} \n"

    border_len = len(table_str.split("\n")[0]) - 1
    table_str = table_str

    return table_str


def plot(
    net,
    title=None,
    parameters=[],
    ngs=[],
    sgs=[],
    scaling_factor=3,
    label_font_size=6,
    recorder_index=461,
    env_recorder_index=461,
    show_similarity=False,
    num_columns=4,
):
    n = len(ngs)
    fig_counter = memoized_number()
    fig, axd = plt.subplot_mosaic(
        (
            """
                    AAAA
                    BBCC
                    BBCC
                    IIII
                    """
            if len(parameters)
            else """
                    AAAA
                    BBCC
                    BBCC
                    IIII
                    """
        ),
        layout="constrained",
        # "image" will contain a square image. We fine-tune the width so that
        # there is no excess horizontal or vertical margin around the image.
        figsize=(12 * scaling_factor, 3.5 * scaling_factor),
    )
    # fig, axd = plt.subplot_mosaic(
    #     (
    #         """
    #                 BBCC
    #                 BBCC
    #                 EEGG
    #                 DDAA
    #                 DDFF
    #                 DDHH
    #                 IIII
    #                 """
    #         if len(parameters)
    #         else """
    #                 BBCC
    #                 BBCC
    #                 EEGG
    #                 FFDD
    #                 HHDD
    #                 """
    #     ),
    #     layout="constrained",
    #     # "image" will contain a square image. We fine-tune the width so that
    #     # there is no excess horizontal or vertical margin around the image.
    #     figsize=(12 * scaling_factor, 6 * scaling_factor),
    # )

    fig.suptitle(
        title or "Plot",
        fontsize=(label_font_size + 7) * scaling_factor,
        fontweight="bold",
    )

    # Add parameters as text on the plot
    if len(parameters):
        axd["I"].axis("off")

        params_text = format_parameters(parameters, num_columns)
        axd["I"].text(
            0.5,
            0.5,
            params_text,
            fontsize=label_font_size * scaling_factor,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axd["I"].transAxes,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    sg = sgs[0]
    weights = sg[recorder_index, 0].variables["weights"]
    w_max = weights.max()
    w_min = weights.min()
    LC = torch.tensor([calculate_learning_convergence.CLC(w,w_max,w_min) for w in weights])

    similarity=[]
    if show_similarity:
        LC = LC/LC.max()
        for i in range(net.iteration):
            similarity.append(
                1
                - cosine(
                    sg[recorder_index, 0].variables["weights"][i, :, 0].cpu(),
                    sg[recorder_index, 0].variables["weights"][i, :, 1].cpu(),
                )
            )


        axd["A"].plot(similarity,label = "Similarity")
    axd["A"].plot(LC,label="Learning Convergence")
    axd["A"].set_ylabel("", fontsize=label_font_size * scaling_factor)
    axd["A"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["A"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["A"].set_xlim(0, net.iteration)
    axd["A"].set_title("Custom Learning Convergence", fontsize=(label_font_size + 1) * scaling_factor)
    axd["A"].set_xlabel(
        f"{fig_counter()} time",
        fontsize=label_font_size * scaling_factor,
    )
    axd["A"].legend(
        loc="best",
        fontsize=label_font_size * scaling_factor / 2,
    )

    axd["B"].scatter(
        ngs[0][env_recorder_index, 0].variables["spikes"][:, 0].cpu(),
        ngs[0][env_recorder_index, 0].variables["spikes"][:, 1].cpu(),
        c=(
            ngs[0][env_recorder_index, 0].variables["spikes"][:, 0].cpu()
            // ngs[0].network.input_period
        )
        % ngs[1].size,
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

    axd["B"].set_title(
        "Input Neuron Group", fontsize=(label_font_size + 1) * scaling_factor
    )
    axd["B"].set_xlabel(
        f"{fig_counter()} time ({ngs[0].tag})",
        fontsize=label_font_size * scaling_factor,
    )


    axd["C"].scatter(
        ngs[1][env_recorder_index, 0].variables["spikes"][:, 0].cpu(),
        ngs[1][env_recorder_index, 0].variables["spikes"][:, 1].cpu(),
        # c=ngs[1][env_recorder_index, 0].variables["spikes"][:, 1].cpu(),
        c=(
            ngs[1][env_recorder_index, 0].variables["spikes"][:, 0].cpu()
            // ngs[1].network.input_period
        )
        % ngs[1].size,
        s=100,
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
    axd["C"].set_title(
        "Output Neuron Group", fontsize=(label_font_size + 1) * scaling_factor
    )
    axd["C"].set_xlabel(
        f"{fig_counter()} time ({ngs[1].tag})",
        fontsize=label_font_size * scaling_factor,
    )

    for step in range(0, ngs[0].network.iteration, ngs[0].network.input_period):
        axd["B"].axvline(x=step, color="grey", linestyle="-", linewidth=0.5)
        axd["C"].axvline(x=step, color="grey", linestyle="-", linewidth=0.5)

    for step in range(
        0, ngs[0].network.iteration, ngs[0].network.input_period * ngs[1].size
    ):
        axd["B"].axvline(x=step, color="black", linestyle="-", linewidth=0.5)
        axd["C"].axvline(x=step, color="black", linestyle="-", linewidth=0.5)
    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Show the plot
    fig.show()
