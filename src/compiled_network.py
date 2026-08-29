"""NumPy-compiled feed-forward network matching neat-python activate semantics."""

import neat
import numpy as np


class CompiledNetwork:
    """Fast forward pass equivalent to neat.nn.FeedForwardNetwork.activate."""

    def __init__(self, input_nodes, output_nodes, layers, key_to_index):
        # Ordered input/output node keys from the NEAT genome config.
        self.input_nodes = tuple(input_nodes)
        self.output_nodes = tuple(output_nodes)
        # Each layer: (out_index, link_indices, link_weights, bias, response, act_fn).
        self._layers = layers
        # Map NEAT node keys (negative inputs) to dense buffer indices.
        self._key_to_index = key_to_index
        self._values = np.zeros(len(key_to_index), dtype=np.float64)

    @classmethod
    def from_genome(cls, genome, config):
        """Compile a NEAT genome into a NumPy forward-pass representation."""
        reference = neat.nn.FeedForwardNetwork.create(genome, config)
        all_keys = set(reference.input_nodes + reference.output_nodes)
        for node, _act, _agg, _bias, _response, links in reference.node_evals:
            all_keys.add(node)
            for inode, _weight in links:
                all_keys.add(inode)
        ordered_keys = sorted(all_keys)
        key_to_index = {key: index for index, key in enumerate(ordered_keys)}

        layers = []
        for node, act_func, _agg_func, bias, response, links in reference.node_evals:
            if links:
                link_idx = np.array(
                    [key_to_index[inode] for inode, _w in links], dtype=np.int32
                )
                link_w = np.array([w for _inode, w in links], dtype=np.float64)
            else:
                link_idx = np.empty(0, dtype=np.int32)
                link_w = np.empty(0, dtype=np.float64)
            out_index = key_to_index[node]
            layers.append(
                (
                    out_index,
                    link_idx,
                    link_w,
                    float(bias),
                    float(response),
                    act_func,
                )
            )
        return cls(
            reference.input_nodes,
            reference.output_nodes,
            layers,
            key_to_index,
        )

    def activate(self, inputs):
        """Run one forward pass; returns output values in NEAT output order."""
        if len(inputs) != len(self.input_nodes):
            raise RuntimeError(
                f"Expected {len(self.input_nodes)} inputs, got {len(inputs)}"
            )
        self._values.fill(0.0)
        for node_key, value in zip(self.input_nodes, inputs):
            self._values[self._key_to_index[node_key]] = value
        for out_index, link_idx, link_w, bias, response, act_func in self._layers:
            if len(link_idx) == 0:
                summed = 0.0
            else:
                # Python sum matches neat aggregation (np.dot can differ at ~1e-16).
                summed = sum(
                    float(self._values[i]) * float(w)
                    for i, w in zip(link_idx, link_w)
                )
            self._values[out_index] = float(act_func(bias + response * summed))
        return [
            float(self._values[self._key_to_index[node_key]])
            for node_key in self.output_nodes
        ]
