import pymonntorch as pmt
import scipy.stats
import torch
import numpy as np
import copy
import scipy


class Encode(pmt.Behavior):  # time to first spikes
    def initialize(self, ng):
        self.data = self.parameter("data", required=True)
        self.theta = self.parameter("theta", default=2)
        self.rng = self.parameter("range", None)
        self.time = int(
            (self.parameter("time", default=10) / ng.network.dt)
        )  # time to iteration number
        self.epsilon = self.parameter("epsilon", default=0.001)
        ng.network.input_period = self.parameter("input_period", None)
        self.data_max = self.rng or self.data.max()
        self.data_min = self.data.min()
        self.input_size = len(self.data)
        ng.network.number_of_inputs = self.input_size

        if not ng.network.input_period:
            ng.network.input_period = self.time + 10
        else:
            ng.network.input_period = int(ng.network.input_period / ng.network.dt)

        self.method = self.parameter("method", "")
        ng.network.current_inp_indx = 0
        ng.spikes = ng.spikes = ng.vector(0) != 0

        ng.encoded_matrix = []
        for i in range(self.input_size):
            data = torch.tensor(self.data[i])
            if self.method == "TTFS_lin":
                ng.encoded_matrix.append(self.encode_lin(ng, data))
            elif self.method == "poisson":
                ng.encoded_matrix.append(self.encode_poison(ng, data))
            else:  # TTFS_exp
                ng.encoded_matrix.append(self.encode_exp(ng, data))

    def forward(self, ng):
        ng.spikes = torch.logical_or(
            (
                ng.encoded_matrix[ng.network.current_inp_indx][
                    (ng.network.iteration - 1) % ng.network.input_period
                ]
                if (ng.network.iteration - 1) % ng.network.input_period < self.time
                else ng.vector(0) != 0
            ),
            ng.spikes,
        )
        ng.network.current_inp_indx = int(
            (ng.network.iteration // ng.network.input_period) % self.input_size
        )

    def encode_exp(self, ng, d):
        data = self.scale_data(ng, copy.deepcopy(d))
        encoded_matrix = torch.zeros((self.time, data.shape[0]), dtype=torch.bool)

        tau = -self.time / np.log(self.epsilon / self.theta)
        print(f"theta: {self.theta}, tau: {tau}, epsilon: {self.epsilon}")
        for t in range(self.time):
            # threshold = self.theta * np.exp(-(t + 1) / tau)
            threshold = np.exp(-(t + 1) / tau)
            encoded_matrix[t, :] = data >= threshold
            data[data >= threshold] = 0
        return encoded_matrix.type(torch.bool)

    def encode_lin(self, ng, d):
        data = self.scale_data(ng, copy.deepcopy(d))
        encoded_matrix = torch.zeros((self.time, data.shape[0]), dtype=torch.bool)

        for t in range(self.time):
            threshold = ((self.time - t - 1) / self.time) + self.epsilon
            encoded_matrix[t, :] = data >= threshold
            data[data >= threshold] = 0
        return encoded_matrix.type(torch.bool)

    def encode_poison(self, ng, d):
        data = self.scale_data(ng, copy.deepcopy(d))
        num_neurons = len(data)
        # encoded_matrix = np.zeros((self.time, data.shape[0]), dtype=bool)
        encoded_matrix = np.zeros((data.shape[0], self.time), dtype=bool)

        for i in range(num_neurons):
            spikes_times = np.random.poisson(data[i], self.time)
            for j, t in enumerate(spikes_times):
                if t > 0:
                    encoded_matrix[i, j : t + j] = 1
        return torch.tensor(encoded_matrix.T)

    def scale_data(self, ng, data):
        if len(data.shape) != 1:
            raise Exception(f"data shape must be (n*m,), not {data.shape}")

        data_min = self.data_min
        data_max = self.data_max
        data_range = data_max - data_min
        data = (data - data_min) / (data_range)  # -> [0,1]
        data = data * (1 - self.epsilon) + self.epsilon  # -> [epsilon,1]
        return data


class InputMatrix(pmt.Behavior):  # time to first spikes
    def initialize(self, ng):
        self.encoded_matrix = self.parameter("encoded_matrix", None, required=True)
        self.time = int(
            (self.parameter("time", default=10) / ng.network.dt)
        )  # time to iteration number
        self.epsilon = self.parameter("epsilon", default=0.001)
        ng.network.input_period = self.parameter("input_period", None)

        if not ng.network.input_period:
            ng.network.input_period = self.time + 10
        else:
            ng.network.input_period = int(ng.network.input_period / ng.network.dt)

        self.method = self.parameter("method", "lin")
        ng.network.current_inp_indx = 0
        ng.spikes = ng.spikes = ng.vector(0) != 0

        ng.encoded_matrix = copy.deepcopy(self.encoded_matrix)
        self.input_size = len(self.encoded_matrix)

    def forward(self, ng):
        ng.spikes = torch.logical_or(
            (
                ng.encoded_matrix[ng.network.current_inp_indx][
                    (ng.network.iteration - 1) % ng.network.input_period
                ]
                if (ng.network.iteration - 1) % ng.network.input_period < self.time
                else ng.vector(0) != 0
            ),
            ng.spikes,
        )
        ng.network.current_inp_indx = int(
            (ng.network.iteration / ng.network.input_period) % self.input_size
        )


class Number(pmt.Behavior):
    def initialize(self, ng):
        self.data = self.parameter("data", required=True)
        self.range = self.parameter("range", [0, 10])
        self.epsilon = self.parameter("epsilon", default=0.001)
        self.std_dev = self.parameter("std_dev", 1)
        self.time = int(
            (self.parameter("time", default=10) / ng.network.dt)
        )  # time to iteration number
        ng.network.input_period = int(
            self.parameter("input_period", default=1000000) / ng.network.dt
        )
        ng.spikes = ng.vector(0) != 0
        ng.encoded_matrix = self.gen_encode_matrix(ng)

    def forward(self, ng):
        ng.spikes = torch.logical_or(
            (
                ng.encoded_matrix[(ng.network.iteration - 1) % ng.network.input_period]
                if ng.network.iteration % ng.network.input_period <= self.time
                else ng.vector(0) != 0
            ),
            ng.spikes,
        )

    def gen_encode_matrix(self, ng):
        range_length = np.abs(self.range[0] - self.range[1])
        scaled_value = 10 * self.data / range_length  # -> [0,10]
        N = ng.size
        encoded_matrix = torch.zeros((self.time, N), dtype=torch.bool)
        prob_values = torch.tensor(
            [
                self.norm_prob(scaled_value, mean=i + 1, std_dev=self.std_dev)
                for i in range(N)
            ]
        )
        scaled_prob = prob_values / prob_values.max()

        for t in range(self.time):
            threshold = (self.time - t) / (self.time + 1)
            encoded_matrix[t, :] = scaled_prob >= threshold
            scaled_prob[scaled_prob >= threshold] = 0
        return encoded_matrix

    def norm_prob(self, value, mean=0, std_dev=1):

        # Calculate the probability density at x
        pdf_at_x = scipy.stats.norm.pdf(value, loc=mean, scale=std_dev)

        return pdf_at_x


class ResetMemory(pmt.Behavior):  # reset network memory
    def initialize(self, ng):
        self.neuron_groups = ng.network["NeuronGroup"]

    def forward(self, ng):
        if ng.network.iteration % ng.network.input_period == 1:
            for i in range(len(self.neuron_groups)):
                self.neuron_groups[i].trace = self.neuron_groups[i].vector(
                    0, dtype=torch.float64
                )
                self.neuron_groups[i].v *= 0
                self.neuron_groups[i].v += self.neuron_groups[i].v_reset
