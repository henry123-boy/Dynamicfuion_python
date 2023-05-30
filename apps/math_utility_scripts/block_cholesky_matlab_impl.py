# Original MATLAB code by Alexander Mamonov, 2015
import numpy as np
from scipy.linalg import sqrtm


def block_start_and_end(block_size, j):
    """ Helper function for the mblockchol-function
        Returns: a list of the "active indexes" of size `block_size`
        Corresponding to "ind = @(j) in (1:Ns) + Ns*(j-1)" in Matlab code
        Added a -1 since k in range(1, Nt) and we want to have starting index 0
    """
    return block_size * j, block_size * (j + 1)


def mblockchol(M, block_size, block_count):
    """ The function for the Block-Cholesky factorization
        mblockchol: Block Cholesky M = R' * R
    """

    L = np.zeros([block_size * block_count, block_size * block_count])

    for k in range(0, block_count):
        block_sum = np.zeros((block_size, block_size), dtype=np.float64)
        k_start, k_end = block_start_and_end(block_size, k)
        for j in range(0, k):
            j_start, j_end = block_start_and_end(block_size, j)
            L_kj = L[k_start:k_end, j_start:j_end]
            block_sum += L_kj @ L_kj.transpose()

        # Update L-matrix diagonal blocks
        L_kk = sqrtm(
            M[k_start:k_end, k_start:k_end] - block_sum
        )
        L[k_start:k_end, k_start:k_end] = L_kk

        # Update L-matrix blocks below the diagonal
        for i in range(k, block_count):
            block_sum = np.zeros((block_size, block_size), dtype=np.float64)
            i_start, i_end = block_start_and_end(block_size, i)
            for j in range(0, k):
                j_start, j_end = block_start_and_end(block_size, j)
                L_ij = L[i_start:i_end, j_start:j_end]
                L_kj = L[k_start:k_end, j_start:j_end]
                block_sum += L_ij @ L_kj.transpose()

            # update "inner" matrix blocks
            M_new = M[i_start:i_end, k_start:k_end] - block_sum

            L_ik = M_new @ np.linalg.inv(L_kk)

            L[i_start:i_end, k_start:k_end] = L_ik

    # Convert to final resulting matrix
    R = L.transpose()
    return R


if __name__ == '__main__':
    # Given input data
    block_size = 2
    block_count = 3
    M = np.array([[6, 5, 4, 3, 2, 1],
                  [5, 6, 5, 4, 3, 2],
                  [4, 5, 6, 5, 4, 3],
                  [3, 4, 5, 6, 5, 4],
                  [2, 3, 4, 5, 6, 5],
                  [1, 2, 3, 4, 5, 6]], dtype=np.float64)
    # print(M)

    # Test cholesky function
    R = mblockchol(M, block_size, block_count)
    # print("Result of function:")
    # print(R)
    #
    # print("\nCheck R.T * R to see if correct (--> should be M exactly):")
    # print(R.T@R)

    print(f"R.T * R == M: {np.allclose(R.T @ R, M)}")
    # @formatter:off
    A = np.array([[2.45169038, 1.42894668, 0.63165919, 0.        , 0.        , 0.        , 1.07932224, 1.40288045, 2.23818168, 0.        , 0.        , 0.        , 1.37510478, 1.14987882, 1.1943462 ],
                  [1.42894668, 0.83990118, 0.34912156, 0.        , 0.        , 0.        , 0.59722243, 0.83648366, 1.30617922, 0.        , 0.        , 0.        , 0.83513189, 0.64321492, 0.65543784],
                  [0.63165919, 0.34912156, 0.24187581, 0.        , 0.        , 0.        , 0.43872485, 0.36800668, 0.59935502, 0.        , 0.        , 0.        , 0.24749794, 0.40630276, 0.45841593],
                  [0.        , 0.        , 0.        , 0.87227288, 0.57239411, 0.67728253, 0.86560283, 0.10094673, 0.53603499, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
                  [0.        , 0.        , 0.        , 0.57239411, 0.87288195, 1.15125361, 0.93027362, 0.43293302, 0.61620908, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
                  [0.        , 0.        , 0.        , 0.67728253, 1.15125361, 1.54236828, 1.21180286, 0.55726037, 0.74590869, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
                  [1.07932224, 0.59722243, 0.43872485, 0.86560283, 0.93027362, 1.21180286, 2.75921511, 1.59557984, 2.13321083, 0.70043394, 0.96690401, 0.91985543, 0.43772105, 0.73069373, 0.80650692],
                  [1.40288045, 0.83648366, 0.36800668, 0.10094673, 0.43293302, 0.55726037, 1.59557984, 2.77477735, 2.33361114, 0.88064416, 0.8345952 , 1.08545706, 0.92045626, 0.66243422, 0.61645061],
                  [2.23818168, 1.30617922, 0.59935502, 0.53603499, 0.61620908, 0.74590869, 2.13321083, 2.33361114, 3.11453978, 0.48350104, 0.6275925 , 0.62482637, 1.27226137, 1.08019913, 1.1076815 ],
                  [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.70043394, 0.88064416, 0.48350104, 0.9315922 , 1.08369109, 1.07768489, 0.        , 0.        , 0.        ],
                  [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.96690401, 0.8345952 , 0.6275925 , 1.08369109, 1.45538459, 1.27073765, 0.        , 0.        , 0.        ],
                  [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.91985543, 1.08545706, 0.62482637, 1.07768489, 1.27073765, 1.39484058, 0.        , 0.        , 0.        ],
                  [1.37510478, 0.83513189, 0.24749794, 0.        , 0.        , 0.        , 0.43772105, 0.92045626, 1.27226137, 0.        , 0.        , 0.        , 0.96901102, 0.49293692, 0.43591069],
                  [1.14987882, 0.64321492, 0.40630276, 0.        , 0.        , 0.        , 0.73069373, 0.66243422, 1.08019913, 0.        , 0.        , 0.        , 0.49293692, 0.69536983, 0.77337116],
                  [1.1943462 , 0.65543784, 0.45841593, 0.        , 0.        , 0.        , 0.80650692, 0.61645061, 1.1076815 , 0.        , 0.        , 0.        , 0.43591069, 0.77337116, 0.8891878 ]])
    # @formatter:on
    L = mblockchol(A, 3, 5)
    print(f"L.T * L == A: {np.allclose(L.T @ L, A)}")
