/** Feed-forward activate matching neat.nn.FeedForwardNetwork. */

import type { GenomeConfig } from "./config";
import { type ConnectionGene, type Genome, type NodeGene } from "./genome";

interface NodeEval {
  node: number;
  activation: string;
  bias: number;
  response: number;
  links: { from: number; weight: number }[];
}

export class CompiledNetwork {
  constructor(
    private inputNodes: number[],
    private outputNodes: number[],
    private evals: NodeEval[],
  ) {}

  /** One forward pass. Link sums follow enabled-connection insertion order. */
  activate(inputs: number[]): number[] {
    if (inputs.length !== this.inputNodes.length) {
      throw new Error(`Expected ${this.inputNodes.length} inputs, got ${inputs.length}`);
    }
    const values = new Map<number, number>();
    for (let i = 0; i < this.inputNodes.length; i += 1) {
      values.set(this.inputNodes[i], inputs[i]);
    }
    for (const node of this.outputNodes) {
      if (!values.has(node)) {
        values.set(node, 0);
      }
    }
    for (const evaluation of this.evals) {
      let summed = 0;
      for (const link of evaluation.links) {
        summed += (values.get(link.from) ?? 0) * link.weight;
      }
      values.set(evaluation.node, activate(evaluation.activation, evaluation.bias + evaluation.response * summed));
    }
    return this.outputNodes.map((node) => values.get(node) ?? 0);
  }

  static fromGenome(genome: Genome, config: GenomeConfig): CompiledNetwork {
    const enabled: ConnectionGene[] = [];
    for (const connection of genome.connections.values()) {
      if (connection.enabled) {
        enabled.push(connection);
      }
    }
    const pairs = enabled.map((connection) => [connection.inNode, connection.outNode] as const);
    const { layers, required } = feedForwardLayers(config.inputKeys, config.outputKeys, pairs);
    const requiredWithInputs = new Set<number>([...required, ...config.inputKeys]);
    const evals: NodeEval[] = [];
    for (const layer of layers) {
      for (const node of layer) {
        const links: { from: number; weight: number }[] = [];
        for (const connection of enabled) {
          if (connection.outNode === node && requiredWithInputs.has(connection.inNode)) {
            links.push({ from: connection.inNode, weight: connection.weight });
          }
        }
        const gene = genome.nodes.get(node) as NodeGene;
        evals.push({
          node,
          activation: gene.activation,
          bias: gene.bias,
          response: gene.response,
          links,
        });
      }
    }
    return new CompiledNetwork(config.inputKeys, config.outputKeys, evals);
  }
}

function activate(name: string, z: number): number {
  if (name === "tanh") {
    const clipped = Math.max(-60, Math.min(60, 2.5 * z));
    return Math.tanh(clipped);
  }
  if (name === "sigmoid") {
    const clipped = Math.max(-60, Math.min(60, 5 * z));
    return 1 / (1 + Math.exp(-clipped));
  }
  if (name === "relu") {
    return z > 0 ? z : 0;
  }
  return Math.tanh(Math.max(-60, Math.min(60, 2.5 * z)));
}

/** neat.graphs.feed_forward_layers, with set insertion following connection order. */
function feedForwardLayers(
  inputs: number[],
  outputs: number[],
  connections: readonly (readonly [number, number])[],
): { layers: number[][]; required: Set<number> } {
  const inputSet = new Set(inputs);
  const required = new Set(outputs);
  let frontier = new Set(outputs);
  while (true) {
    const incoming = new Set<number>();
    for (const [from, to] of connections) {
      if (frontier.has(to) && !frontier.has(from)) {
        incoming.add(from);
      }
    }
    if (incoming.size === 0) {
      break;
    }
    const layerNodes = [...incoming].filter((node) => !inputSet.has(node));
    if (layerNodes.length === 0) {
      break;
    }
    for (const node of layerNodes) {
      required.add(node);
    }
    frontier = new Set([...frontier, ...incoming]);
  }
  const nodesWithInputs = new Set<number>();
  for (const [, to] of connections) {
    nodesWithInputs.add(to);
  }
  const bias = [...required].filter((node) => !nodesWithInputs.has(node));
  const layers: number[][] = [];
  const ready = new Set<number>([...inputs, ...bias]);
  if (bias.length) {
    layers.push(bias);
  }
  while (true) {
    const candidates: number[] = [];
    const seen = new Set<number>();
    for (const [from, to] of connections) {
      if (ready.has(from) && !ready.has(to) && !seen.has(to)) {
        seen.add(to);
        candidates.push(to);
      }
    }
    const next: number[] = [];
    for (const node of candidates) {
      if (!required.has(node)) {
        continue;
      }
      const inbound = connections.filter(([from, to]) => to === node && required.has(from));
      if (inbound.every(([from]) => ready.has(from))) {
        next.push(node);
      }
    }
    if (next.length === 0) {
      break;
    }
    layers.push(next);
    for (const node of next) {
      ready.add(node);
    }
  }
  return { layers, required };
}
