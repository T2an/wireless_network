import numpy as np
import numpy as np
import math
# Module to display T/F matrix
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
# Module for Unit testing
import pytest
# Module for convolutionnal decoder
import sk_dsp_comm.fec_conv as fec


my_data = np.genfromtxt('tfMatrix.csv', delimiter=";")
mat_complex = my_data[:,0::2] +1j*my_data[:,1::2]
first_part = mat_complex[:, 0:313]
second_part = mat_complex[:, 713:1024]

# Regrouper les deux matrices horizontalement
combined_matrix = np.hstack((first_part, second_part))


def powerDistributionGraph(Z):
    """
    Draw the power distribution graph
    """
    Z=np.abs(Z)
    fig, ax = plt.subplots()
    cs = ax.contourf(np.linspace(0, len(Z[0]), len(Z[0])), np.linspace(0,
    len(Z), len(Z)), Z)
    cbar = fig.colorbar(cs)
    ax.set_title('Distribution de la puissance de la matrice temps fréquence')
    ax.set_xlabel('Fréquence (sous-porteuses)')
    ax.set_ylabel('Temps (symboles temporels)')
    plt.show()


def bpsk_demod(qamSeq):
    """a
    Demodulates a BPSK stream into a binary stream.

    Args:
    - qamSeq (numpy array): Input BPSK stream as a numpy array.

    Returns:
    - numpy array: Demodulated binary stream.
    """
    # Copier la séquence QAM pour éviter de modifier la séquence originale
    bitSeq = qamSeq.copy()
    # Transformer les -1 en 0 et les 1 en 1
    bitSeq[bitSeq >= 0] = 1
    bitSeq[bitSeq < 0] = 0
    return bitSeq



def hamming748_decode(bitSeq):
    """
    Decode a binary sequence that has been coded with a Hamming(7,4,8) coder.

    Args:
    - bitSeq (list): Input binary sequence.

    Returns:
    - list: Decoded binary sequence.
    """
    # Define the syndrome matrix H
    H = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ])

    decoded_bits = []

    for i in range(0, len(bitSeq), 8):
    # Calculate the syndrome ≤ = H × y748
        block = bitSeq[i:i+7]
        syndrome = np.dot(H, block) % 2

        # Check the syndrome
        if np.array_equal(syndrome, [0, 0, 0]):
            # No errors
            decoded_bits.extend(block[:4])
        elif np.count_nonzero(syndrome) == 1:
            # Single-bit error
            error_index = np.where(syndrome == 1)[0][0]
            print(error_index)
            # Correct the error by flipping the corresponding bit
            #bitSeq[error_index] ^= 1

            block[error_index] = 1 if block[error_index] == 0 else 0

            decoded_bits.extend(block[:4])
        else:
            return None

    return decoded_bits



matrice_without_sync=combined_matrix[2:, :]
print("La taille de la nouvelle matrice est :", matrice_without_sync.shape)
matrice_PBCH=matrice_without_sync[0, :]
print("La taille de la nouvelle matrice est :", matrice_PBCH.shape)
#print(matrice_PBCH)

# Tester la fonction avec un exemple
bitSeq = bpsk_demod(matrice_PBCH.real)
   # print("Séquence binaire après la démodulation BPSK :", bitSeq)

matrice_PBCH_user=matrice_PBCH[:48]
matrice_PBCH_user_demod=bpsk_demod(matrice_PBCH_user.real)
print(hamming748_decode(matrice_PBCH_user_demod))


