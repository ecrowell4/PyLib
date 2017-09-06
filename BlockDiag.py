def diagblock(v, k=0):
    """ Creates a block diagonal matrix, with the elements of v
    as the diagonals.
    """
    import numpy as np
    
    shapes = np.array([a.shape for a in v])
    out = np.zeros(np.sum(shapes, axis=0) + abs(k)*shapes[0], dtype=v[0].dtype)

    if k >= 0:
        r, c = 0, abs(k)*shapes[0][0]
    else:
        r, c = abs(k)*shapes[0][0], 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = v[i]
        r += rr
        c += cc
    return out
