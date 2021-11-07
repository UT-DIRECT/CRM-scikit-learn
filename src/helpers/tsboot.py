from math import floor

import numpy as np

def mb_bootstrap(X, y, b_length):
    l = len(X)
    indicies = np.linspace(0, l - 1, l).astype(np.int8)
    size = floor(l / b_length)
    blocks = _get_blocks(indicies, b_length)
    n_blocks = len(blocks)
    sample_blocks = _get_sampled_blocks(n_blocks, size)
    sample_X = []
    sample_y = []
    for block in sample_blocks:
        sample_X += [X[i] for i in blocks[block]]
        sample_y += [y[i] for i in blocks[block]]
    return (sample_X, sample_y)


def _get_sampled_blocks(n_blocks, size):
    block_indicies = np.linspace(0, n_blocks - 1, n_blocks).astype(np.int8)
    sample_blocks = np.random.choice(block_indicies, size=size)
    return sample_blocks

def _get_blocks(indicies, b_length):
    blocks = []
    for start in indicies:
        end = start + b_length
        block = indicies[start:end]
        if len(block) == b_length:
            blocks.append(block)
        else:
            break
    return blocks
