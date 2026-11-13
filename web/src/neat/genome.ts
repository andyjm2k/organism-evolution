/** DefaultGenome operations the living world calls, matched to neat-python 2.0.0. */

import type { PythonRandom } from "../rng/python-random";
import type { GenomeConfig } from "./config";
import type { PartLevels } from "../sim/morphology";

export interface NodeGene {
  key: number;
  bias: number;
  response: number;
  activation: string;
  aggregation: string;
  timeConstant: number;
}

export interface ConnectionGene {
  inNode: number;
  outNode: number;
  weight: number;
  enabled: boolean;
  innovation: number;
}

export class Genome {
  key: number;
  fitness: number | null = null;
  nodes = new Map<number, NodeGene>();
  connections = new Map<string, ConnectionGene>();
  /** Living-world fields carried on the genome object in Python. */
  dietKey = 1;
  bodyGenes: BodyGeneState | null = null;

  constructor(key: number) {
    this.key = key;
  }

  /** Unconnected starter: output nodes only when initial_connection is unconnected. */
  configureNew(config: GenomeConfig, rng: PythonRandom): void {
    for (const nodeKey of config.outputKeys) {
      this.nodes.set(nodeKey, createNode(config, nodeKey, rng));
    }
    for (let i = 0; i < config.numHidden; i += 1) {
      const nodeKey = allocateNodeKey(config, this.nodes);
      this.nodes.set(nodeKey, createNode(config, nodeKey, rng));
    }
  }

  /** NEAT crossover. Equal fitness selects genome2 as the fitter parent. */
  configureCrossover(genome1: Genome, genome2: Genome, config: GenomeConfig, rng: PythonRandom): void {
    const fit1 = genome1.fitness ?? -Infinity;
    const fit2 = genome2.fitness ?? -Infinity;
    const parent1 = fit1 > fit2 ? genome1 : genome2;
    const parent2 = fit1 > fit2 ? genome2 : genome1;
    const innov1 = indexByInnovation(parent1);
    const innov2 = indexByInnovation(parent2);
    const all = new Set<number>([...innov1.keys(), ...innov2.keys()]);
    for (const innovation of all) {
      const cg1 = innov1.get(innovation);
      const cg2 = innov2.get(innovation);
      let gene: ConnectionGene | null = null;
      if (cg1 && cg2 && cg1.inNode === cg2.inNode && cg1.outNode === cg2.outNode) {
        gene = crossoverConnection(cg1, cg2, rng);
      } else if (cg1) {
        gene = copyConnection(cg1);
      }
      if (!gene) {
        continue;
      }
      if (config.feedForward && createsCycle(this.connections, gene.inNode, gene.outNode)) {
        continue;
      }
      this.connections.set(connKey(gene.inNode, gene.outNode), gene);
    }
    for (const [key, ng1] of parent1.nodes) {
      const ng2 = parent2.nodes.get(key);
      this.nodes.set(key, ng2 ? crossoverNode(ng1, ng2, rng) : copyNode(ng1));
    }
  }

  /** Structural mutations, then every connection gene, then every node gene. */
  mutate(config: GenomeConfig, rng: PythonRandom): void {
    if (rng.random() < config.nodeAddProb) {
      this.mutateAddNode(config, rng);
    }
    if (rng.random() < config.nodeDeleteProb) {
      this.mutateDeleteNode(config, rng);
    }
    if (rng.random() < config.connAddProb) {
      this.mutateAddConnection(config, rng);
    }
    if (rng.random() < config.connDeleteProb) {
      this.mutateDeleteConnection(rng);
    }
    for (const connection of this.connections.values()) {
      mutateConnection(connection, config, rng);
    }
    for (const node of this.nodes.values()) {
      mutateNode(node, config, rng);
    }
  }

  /** Split one connection into a hidden node. */
  mutateAddNode(config: GenomeConfig, rng: PythonRandom): void {
    if (this.connections.size === 0) {
      if (config.structuralMutationSurer) {
        this.mutateAddConnection(config, rng);
      }
      return;
    }
    const chosen = rng.choice([...this.connections.values()]);
    const newId = allocateNodeKey(config, this.nodes);
    const node = createNode(config, newId, rng);
    node.bias = 0;
    this.nodes.set(newId, node);
    chosen.enabled = false;
    const inInnov = config.innovation.getInnovationNumber(chosen.inNode, newId, "add_node_in");
    const outInnov = config.innovation.getInnovationNumber(newId, chosen.outNode, "add_node_out");
    this.addConnection(config, rng, chosen.inNode, newId, 1, true, inInnov);
    this.addConnection(config, rng, newId, chosen.outNode, chosen.weight, true, outInnov);
  }

  /** Add one feed-forward connection between a legal pair. */
  mutateAddConnection(config: GenomeConfig, rng: PythonRandom): void {
    const outputs = [...this.nodes.keys()];
    const outNode = rng.choice(outputs);
    const inputs = possibleInputs(this.nodes, config.inputKeys, config.outputKeys);
    const inNode = rng.choice(inputs);
    const key = connKey(inNode, outNode);
    if (this.connections.has(key)) {
      if (config.structuralMutationSurer) {
        this.connections.get(key)!.enabled = true;
      }
      return;
    }
    const outputSet = new Set(config.outputKeys);
    if (outputSet.has(inNode) && outputSet.has(outNode)) {
      return;
    }
    if (config.feedForward && createsCycle(this.connections, inNode, outNode)) {
      return;
    }
    const innovation = config.innovation.getInnovationNumber(inNode, outNode, "add_connection");
    const connection = createConnection(config, rng, inNode, outNode, innovation);
    this.connections.set(key, connection);
  }

  /** Drop one hidden node and every wire that touched it. */
  mutateDeleteNode(config: GenomeConfig, rng: PythonRandom): void {
    const hidden = [...this.nodes.keys()].filter((key) => !config.outputKeys.includes(key));
    if (hidden.length === 0) {
      return;
    }
    const delKey = rng.choice(hidden);
    for (const [key, connection] of [...this.connections]) {
      if (connection.inNode === delKey || connection.outNode === delKey) {
        this.connections.delete(key);
      }
    }
    this.nodes.delete(delKey);
  }

  /** Drop one connection gene. */
  mutateDeleteConnection(rng: PythonRandom): void {
    if (this.connections.size === 0) {
      return;
    }
    const key = rng.choice([...this.connections.keys()]);
    this.connections.delete(key);
  }

  /** neat DefaultGenome.distance. Node and connection terms are both normalized. */
  distance(other: Genome, config: GenomeConfig): number {
    let nodeDistance = 0;
    if (this.nodes.size || other.nodes.size) {
      let disjoint = 0;
      for (const key of other.nodes.keys()) {
        if (!this.nodes.has(key)) {
          disjoint += 1;
        }
      }
      for (const [key, node] of this.nodes) {
        const partner = other.nodes.get(key);
        if (!partner) {
          disjoint += 1;
        } else {
          nodeDistance += nodeDistancePair(node, partner, config);
        }
      }
      const maxNodes = Math.max(this.nodes.size, other.nodes.size);
      nodeDistance = (nodeDistance + config.compatibilityDisjoint * disjoint) / maxNodes;
    }
    let connectionDistance = 0;
    if (this.connections.size || other.connections.size) {
      let disjoint = 0;
      for (const key of other.connections.keys()) {
        if (!this.connections.has(key)) {
          disjoint += 1;
        }
      }
      for (const [key, connection] of this.connections) {
        const partner = other.connections.get(key);
        if (!partner) {
          disjoint += 1;
        } else {
          connectionDistance += connectionDistancePair(connection, partner, config);
        }
      }
      const maxConn = Math.max(this.connections.size, other.connections.size);
      connectionDistance = (connectionDistance + config.compatibilityDisjoint * disjoint) / maxConn;
    }
    return nodeDistance + connectionDistance;
  }

  /** Insert a connection, initializing attributes before the caller overwrites weight. */
  addConnection(
    config: GenomeConfig,
    rng: PythonRandom,
    inNode: number,
    outNode: number,
    weight: number,
    enabled: boolean,
    innovation: number,
  ): void {
    const connection = createConnection(config, rng, inNode, outNode, innovation);
    connection.weight = weight;
    connection.enabled = enabled;
    this.connections.set(connKey(inNode, outNode), connection);
  }
}

export interface BodyGeneState {
  red: number;
  green: number;
  blue: number;
  diet: number;
  size: number;
  /** Organ sizes. Absent on older saves, which read as the founding body. */
  parts?: Partial<PartLevels>;
}

export function connKey(inNode: number, outNode: number): string {
  return `${inNode},${outNode}`;
}

/** Shared hidden-node counter. The first call starts one past the highest existing key. */
export function allocateNodeKey(config: GenomeConfig, nodes: Map<number, NodeGene>): number {
  if (config.nextNodeId === null) {
    config.nextNodeId = nodes.size ? Math.max(...nodes.keys()) + 1 : config.numOutputs;
  }
  const id = config.nextNodeId;
  config.nextNodeId += 1;
  return id;
}

function createNode(config: GenomeConfig, key: number, rng: PythonRandom): NodeGene {
  return {
    key,
    bias: initGauss(rng, config.biasInitMean, config.biasInitStdev, config.biasMin, config.biasMax),
    response: initGauss(rng, config.responseInitMean, config.responseInitStdev, config.responseMin, config.responseMax),
    activation: config.activationDefault,
    aggregation: config.aggregationDefault,
    // time_constant uses gauss(1, 0) and still consumes the generator.
    timeConstant: initGauss(rng, 1, 0, 0.01, 10),
  };
}

function createConnection(
  config: GenomeConfig,
  rng: PythonRandom,
  inNode: number,
  outNode: number,
  innovation: number,
): ConnectionGene {
  return {
    inNode,
    outNode,
    weight: initGauss(rng, config.weightInitMean, config.weightInitStdev, config.weightMin, config.weightMax),
    enabled: config.enabledDefault,
    innovation,
  };
}

function copyNode(node: NodeGene): NodeGene {
  return { ...node };
}

function copyConnection(connection: ConnectionGene): ConnectionGene {
  return { ...connection };
}

function crossoverNode(a: NodeGene, b: NodeGene, rng: PythonRandom): NodeGene {
  return {
    key: a.key,
    bias: rng.random() > 0.5 ? a.bias : b.bias,
    response: rng.random() > 0.5 ? a.response : b.response,
    activation: rng.random() > 0.5 ? a.activation : b.activation,
    aggregation: rng.random() > 0.5 ? a.aggregation : b.aggregation,
    timeConstant: rng.random() > 0.5 ? a.timeConstant : b.timeConstant,
  };
}

function crossoverConnection(a: ConnectionGene, b: ConnectionGene, rng: PythonRandom): ConnectionGene {
  const gene: ConnectionGene = {
    inNode: a.inNode,
    outNode: a.outNode,
    weight: rng.random() > 0.5 ? a.weight : b.weight,
    enabled: rng.random() > 0.5 ? a.enabled : b.enabled,
    innovation: a.innovation,
  };
  if (!a.enabled || !b.enabled) {
    if (rng.random() < 0.75) {
      gene.enabled = false;
    }
  }
  return gene;
}

function mutateNode(node: NodeGene, config: GenomeConfig, rng: PythonRandom): void {
  node.bias = mutateFloat(rng, node.bias, config.biasMutateRate, config.biasReplaceRate, config.biasMutatePower, config.biasMin, config.biasMax, config.biasInitMean, config.biasInitStdev);
  node.response = mutateFloat(rng, node.response, config.responseMutateRate, config.responseReplaceRate, config.responseMutatePower, config.responseMin, config.responseMax, config.responseInitMean, config.responseInitStdev);
  node.activation = mutateString(rng, node.activation, config.activationMutateRate, config.activationOptions);
  node.aggregation = mutateString(rng, node.aggregation, config.aggregationMutateRate, config.aggregationOptions);
  node.timeConstant = mutateFloat(rng, node.timeConstant, 0, 0, 0, 0.01, 10, 1, 0);
}

function mutateConnection(connection: ConnectionGene, config: GenomeConfig, rng: PythonRandom): void {
  connection.weight = mutateFloat(rng, connection.weight, config.weightMutateRate, config.weightReplaceRate, config.weightMutatePower, config.weightMin, config.weightMax, config.weightInitMean, config.weightInitStdev);
  connection.enabled = mutateBool(rng, connection.enabled, config.enabledMutateRate);
}

function initGauss(rng: PythonRandom, mean: number, stdev: number, min: number, max: number): number {
  return clamp(rng.gauss(mean, stdev), min, max);
}

/** FloatAttribute.mutate_value always draws one random() before the rate checks. */
function mutateFloat(
  rng: PythonRandom,
  value: number,
  mutateRate: number,
  replaceRate: number,
  power: number,
  min: number,
  max: number,
  initMean: number,
  initStdev: number,
): number {
  const roll = rng.random();
  if (roll < mutateRate) {
    return clamp(value + rng.gauss(0, power), min, max);
  }
  if (roll < replaceRate + mutateRate) {
    return initGauss(rng, initMean, initStdev, min, max);
  }
  return value;
}

function mutateString(rng: PythonRandom, value: string, rate: number, options: string[]): string {
  if (rate > 0 && rng.random() < rate) {
    return rng.choice(options);
  }
  return value;
}

function mutateBool(rng: PythonRandom, value: boolean, rate: number): boolean {
  if (rate > 0 && rng.random() < rate) {
    return rng.random() < 0.5;
  }
  return value;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function indexByInnovation(genome: Genome): Map<number, ConnectionGene> {
  const indexed = new Map<number, ConnectionGene>();
  for (const connection of genome.connections.values()) {
    indexed.set(connection.innovation, connection);
  }
  return indexed;
}

function possibleInputs(nodes: Map<number, NodeGene>, inputKeys: number[], outputKeys: number[]): number[] {
  const outputs = new Set(outputKeys);
  const result: number[] = [];
  const seen = new Set<number>();
  for (const key of nodes.keys()) {
    if (!outputs.has(key)) {
      result.push(key);
      seen.add(key);
    }
  }
  for (const key of inputKeys) {
    if (!seen.has(key)) {
      result.push(key);
      seen.add(key);
    }
  }
  return result;
}

function nodeDistancePair(a: NodeGene, b: NodeGene, config: GenomeConfig): number {
  let distance = Math.abs(a.bias - b.bias) + Math.abs(a.response - b.response);
  distance += Math.abs(a.timeConstant - b.timeConstant);
  if (a.activation !== b.activation) {
    distance += 1;
  }
  if (a.aggregation !== b.aggregation) {
    distance += 1;
  }
  return distance * config.compatibilityWeight;
}

function connectionDistancePair(a: ConnectionGene, b: ConnectionGene, config: GenomeConfig): number {
  let distance = Math.abs(a.weight - b.weight);
  if (a.enabled !== b.enabled) {
    distance += 1;
  }
  return distance * config.compatibilityWeight;
}

/** True when adding in->out would close a directed cycle. */
function createsCycle(connections: Map<string, ConnectionGene>, input: number, output: number): boolean {
  if (input === output) {
    return true;
  }
  const visited = new Set<number>([output]);
  const edges = [...connections.values()].map((connection) => [connection.inNode, connection.outNode] as const);
  while (true) {
    let added = 0;
    for (const [from, to] of edges) {
      if (visited.has(from) && !visited.has(to)) {
        if (to === input) {
          return true;
        }
        visited.add(to);
        added += 1;
      }
    }
    if (added === 0) {
      return false;
    }
  }
}
