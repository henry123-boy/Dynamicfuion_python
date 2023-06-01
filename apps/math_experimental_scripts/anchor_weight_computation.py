import math


def compute_anchor_weight(distance, node_coverage):
    return math.exp(-distance**2 / (2 * node_coverage ** 2))


