/** NEAT settings this harness reads from neat-config.ini. neat-python 2.0.0 is the oracle. */

export interface GenomeConfig {
  popSize: number;
  numInputs: number;
  numOutputs: number;
  numHidden: number;
  feedForward: boolean;
  compatibilityDisjoint: number;
  compatibilityWeight: number;
  connAddProb: number;
  connDeleteProb: number;
  nodeAddProb: number;
  nodeDeleteProb: number;
  singleStructuralMutation: boolean;
  structuralMutationSurer: boolean;
  initialConnection: string;
  compatibilityThreshold: number;
  inputKeys: number[];
  outputKeys: number[];
  biasInitMean: number;
  biasInitStdev: number;
  biasMax: number;
  biasMin: number;
  biasMutatePower: number;
  biasMutateRate: number;
  biasReplaceRate: number;
  responseInitMean: number;
  responseInitStdev: number;
  responseMax: number;
  responseMin: number;
  responseMutatePower: number;
  responseMutateRate: number;
  responseReplaceRate: number;
  weightInitMean: number;
  weightInitStdev: number;
  weightMax: number;
  weightMin: number;
  weightMutatePower: number;
  weightMutateRate: number;
  weightReplaceRate: number;
  activationDefault: string;
  activationMutateRate: number;
  activationOptions: string[];
  aggregationDefault: string;
  aggregationMutateRate: number;
  aggregationOptions: string[];
  enabledDefault: boolean;
  enabledMutateRate: number;
  /** Hidden-node id counter shared by every genome, matching neat's node_indexer. */
  nextNodeId: number | null;
  innovation: InnovationTracker;
}

export class InnovationTracker {
  globalCounter = 0;
  private generation = new Map<string, number>();

  exportState(): { globalCounter: number; generation: [string, number][] } {
    return { globalCounter: this.globalCounter, generation: [...this.generation.entries()] };
  }

  importState(state: { globalCounter: number; generation: [string, number][] }): void {
    this.globalCounter = state.globalCounter;
    this.generation = new Map(state.generation);
  }

  /** Same (src, dst, kind) in one generation reuses the innovation number. */
  getInnovationNumber(inputNode: number, outputNode: number, mutationType: string): number {
    const key = `${inputNode}|${outputNode}|${mutationType}`;
    const existing = this.generation.get(key);
    if (existing !== undefined) {
      return existing;
    }
    this.globalCounter += 1;
    this.generation.set(key, this.globalCounter);
    return this.globalCounter;
  }
}

/** Parse the ini into the fields the living world actually uses. */
export function parseNeatConfig(text: string): GenomeConfig {
  const values = parseIni(text);
  const numInputs = int(values, "num_inputs", 31);
  const numOutputs = int(values, "num_outputs", 4);
  const inputKeys: number[] = [];
  for (let i = 0; i < numInputs; i += 1) {
    inputKeys.push(-i - 1);
  }
  const outputKeys: number[] = [];
  for (let i = 0; i < numOutputs; i += 1) {
    outputKeys.push(i);
  }
  return {
    popSize: int(values, "pop_size", 60),
    numInputs,
    numOutputs,
    numHidden: int(values, "num_hidden", 0),
    feedForward: bool(values, "feed_forward", true),
    compatibilityDisjoint: num(values, "compatibility_disjoint_coefficient", 1),
    compatibilityWeight: num(values, "compatibility_weight_coefficient", 0.5),
    connAddProb: num(values, "conn_add_prob", 0),
    connDeleteProb: num(values, "conn_delete_prob", 0),
    nodeAddProb: num(values, "node_add_prob", 0),
    nodeDeleteProb: num(values, "node_delete_prob", 0),
    singleStructuralMutation: bool(values, "single_structural_mutation", false),
    structuralMutationSurer: bool(values, "structural_mutation_surer", false),
    initialConnection: str(values, "initial_connection", "unconnected"),
    compatibilityThreshold: num(values, "compatibility_threshold", 3),
    inputKeys,
    outputKeys,
    biasInitMean: num(values, "bias_init_mean", 0),
    biasInitStdev: num(values, "bias_init_stdev", 1),
    biasMax: num(values, "bias_max_value", 30),
    biasMin: num(values, "bias_min_value", -30),
    biasMutatePower: num(values, "bias_mutate_power", 0),
    biasMutateRate: num(values, "bias_mutate_rate", 0),
    biasReplaceRate: num(values, "bias_replace_rate", 0),
    responseInitMean: num(values, "response_init_mean", 1),
    responseInitStdev: num(values, "response_init_stdev", 0),
    responseMax: num(values, "response_max_value", 30),
    responseMin: num(values, "response_min_value", -30),
    responseMutatePower: num(values, "response_mutate_power", 0),
    responseMutateRate: num(values, "response_mutate_rate", 0),
    responseReplaceRate: num(values, "response_replace_rate", 0),
    weightInitMean: num(values, "weight_init_mean", 0),
    weightInitStdev: num(values, "weight_init_stdev", 1),
    weightMax: num(values, "weight_max_value", 30),
    weightMin: num(values, "weight_min_value", -30),
    weightMutatePower: num(values, "weight_mutate_power", 0),
    weightMutateRate: num(values, "weight_mutate_rate", 0),
    weightReplaceRate: num(values, "weight_replace_rate", 0),
    activationDefault: str(values, "activation_default", "tanh"),
    activationMutateRate: num(values, "activation_mutate_rate", 0),
    activationOptions: str(values, "activation_options", "tanh").split(" ").filter(Boolean),
    aggregationDefault: str(values, "aggregation_default", "sum"),
    aggregationMutateRate: num(values, "aggregation_mutate_rate", 0),
    aggregationOptions: str(values, "aggregation_options", "sum").split(" ").filter(Boolean),
    enabledDefault: bool(values, "enabled_default", true),
    enabledMutateRate: num(values, "enabled_mutate_rate", 0),
    nextNodeId: null,
    innovation: new InnovationTracker(),
  };
}

/** Flat key/value map. Later sections override earlier keys with the same name. */
function parseIni(text: string): Map<string, string> {
  const values = new Map<string, string>();
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.split("#")[0].trim();
    if (!line || line.startsWith("[")) {
      continue;
    }
    const eq = line.indexOf("=");
    if (eq < 0) {
      continue;
    }
    values.set(line.slice(0, eq).trim(), line.slice(eq + 1).trim());
  }
  return values;
}

function str(values: Map<string, string>, key: string, fallback: string): string {
  return values.get(key) ?? fallback;
}

function num(values: Map<string, string>, key: string, fallback: number): number {
  const raw = values.get(key);
  return raw === undefined ? fallback : Number(raw);
}

function int(values: Map<string, string>, key: string, fallback: number): number {
  return Math.trunc(num(values, key, fallback));
}

function bool(values: Map<string, string>, key: string, fallback: boolean): boolean {
  const raw = values.get(key);
  if (raw === undefined) {
    return fallback;
  }
  const lowered = raw.toLowerCase();
  if (["1", "true", "yes", "on"].includes(lowered)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(lowered)) {
    return false;
  }
  return fallback;
}
