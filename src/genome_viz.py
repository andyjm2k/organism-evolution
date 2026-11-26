"""Render a NEAT genome as a compact network graph."""

import pygame


def build_genome_surface(genome, config, width=260, height=180, font=None):
    """Rasterize genome nodes and connections to a pygame surface."""
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill((248, 250, 252, 240))
    if genome is None or config is None:
        return surface
    num_inputs = config.genome_config.num_inputs
    num_outputs = config.genome_config.num_outputs
    # NEAT node keys: inputs are -1..-num_inputs, outputs are 0..num_outputs-1,
    # hidden nodes are >= num_outputs. The old ranges (inputs 0..num_inputs-1,
    # outputs -num_outputs+1..0) were output/hidden keys, so no founding
    # edge ever had both endpoints placed and the graph drew zero edges.
    input_nodes = list(range(-1, -num_inputs - 1, -1))
    hidden_keys = sorted(k for k in genome.nodes if k >= num_outputs)
    output_keys = sorted(k for k in genome.nodes if 0 <= k < num_outputs)
    if not output_keys:
        output_keys = list(range(num_outputs))
    columns = []
    columns.append(("in", input_nodes))
    if hidden_keys:
        columns.append(("hid", hidden_keys[:8]))
    columns.append(("out", output_keys))
    col_count = len(columns)
    margin_x, margin_y = 28, 16
    usable_w = width - margin_x * 2
    usable_h = height - margin_y * 2
    node_positions = {}
    for col_index, (_label, keys) in enumerate(columns):
        x = margin_x + (usable_w * col_index / max(1, col_count - 1))
        for row, key in enumerate(keys):
            y = margin_y + (usable_h * (row + 1) / (len(keys) + 1))
            node_positions[key] = (int(x), int(y))
    for conn_key, conn in genome.connections.items():
        if not conn.enabled:
            continue
        in_key, out_key = conn_key
        if in_key not in node_positions or out_key not in node_positions:
            continue
        weight = conn.weight
        color = _weight_color(weight)
        pygame.draw.line(
            surface,
            color,
            node_positions[in_key],
            node_positions[out_key],
            1,
        )
    for key, pos in node_positions.items():
        # Match the web genome graph: inputs green, outputs orange, hidden blue.
        if key < 0:
            color = (120, 180, 120)
        elif key < num_outputs:
            color = (220, 120, 80)
        else:
            color = (80, 140, 220)
        pygame.draw.circle(surface, color, pos, 5)
        pygame.draw.circle(surface, (40, 40, 40), pos, 5, 1)
    if font is not None:
        label = font.render(
            f"{len(genome.nodes)}n {len(genome.connections)}c",
            True,
            (60, 60, 60),
        )
        surface.blit(label, (8, height - 22))
    pygame.draw.rect(surface, (200, 205, 210), surface.get_rect(), 1, border_radius=6)
    return surface


def _weight_color(weight):
    """Map connection weight to a blue/red edge color."""
    clamped = max(-1.0, min(1.0, weight / 30.0))
    if clamped >= 0:
        g = int(120 + 100 * clamped)
        return (80, g, 220)
    r = int(120 + 100 * abs(clamped))
    return (r, 100, 100)
