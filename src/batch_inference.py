"""Batch neural network inference for the simulation step loop."""


class BatchInferenceEngine:
    """Compile genomes once; organisms use _compiled_network in take_action."""

    def __init__(self):
        # genome_id -> CompiledNetwork
        self._networks = {}

    def register_genome(self, genome_id, genome, config):
        """Compile and cache the network for one population member."""
        from compiled_network import CompiledNetwork

        self._networks[genome_id] = CompiledNetwork.from_genome(genome, config)

    def clear(self):
        """Drop compiled networks after a generation finishes."""
        self._networks.clear()
