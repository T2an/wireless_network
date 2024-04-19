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
first_part = mat_complex[:, 1:313]
second_part = mat_complex[:, 712:1024]

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
        block = bitSeq[i:i+7]
        parity_bit = bitSeq[i+7]
        syndrome = np.dot(H, block) % 2
        
        # If the syndrome is equal to [0,0,0]
        if np.array_equal(syndrome, [0, 0, 0]):
            decoded_bits.extend(block[:4])
        # If the syndrome is not equal to [0,0,0] but the parity bit is good
        elif (np.sum(block) + 1) % 2 == parity_bit:
            # Get the index of the error
            error_index = int(syndrome[2]*4 + syndrome[1]*2 + syndrome[0]) - 1

            # Correct the error by flipping the corresponding bit
            block[error_index] = 1 if block[error_index] == 0 else 0

            decoded_bits.extend(block[:4])
        # Two errors detected then return None
        else:
            return None

    return decoded_bits

def bin2dec(nb):
    """
    Transform a binary list to an integer
    """
    n = "0b"
    for b in nb:
        n = n + str(int(b))
    return int(n, 2)

def getPBCHU(matrice_PBCH, ident):
    for i in range(0, len(matrice_PBCH), 48):
        matrice_PBCH_tmp = matrice_PBCH[i:i+48]
        matrice_PBCH_demod = bpsk_demod(matrice_PBCH_tmp.real)
        bitDec = hamming748_decode(matrice_PBCH_demod)
        if bitDec != None:
            userIdent = bin2dec(bitDec[0:8])
            pdcchuMCS = bin2dec(bitDec[8:10])
            pdcchuSymbStart = bin2dec(bitDec[10:14])
            pdcchuRbStart = bin2dec(bitDec[14:20])
            pdcchuHARQ = bin2dec(bitDec[20:24])

            if ident == userIdent:
                # print("User ident : ", userIdent)
                # print("MCS of PDCCHU : ", pdcchuMCS)
                # print("Symb start of PDCCHU : ", pdcchuSymbStart)
                # print("RB start of PDCCHU : ", pdcchuRbStart)
                # print("HARQ of PDCCHU : ", pdcchuHARQ)
                # print("--------------------------------------")
                PBCHU = {"userIdent": userIdent, "pdcchuMCS": pdcchuMCS, "pdcchuSymbStart": pdcchuSymbStart, "pdcchuRbStart": pdcchuRbStart, "pdcchuHARQ": pdcchuHARQ}
                return PBCHU


# Enlève les deux canaux de synchronisation
matrice_without_sync=combined_matrix[2:, :]
print("La taille de la nouvelle matrice est :", matrice_without_sync.shape)

# Garde uniquement la ligne contenant le canal de diffusion
matrice_PBCH=matrice_without_sync[0, :]
print("La taille de la nouvelle matrice est :", matrice_PBCH.shape)

# Garde uniquement les éléments de 1 à 48 contenant les informations utilisateur
matrice_PBCH_user=matrice_PBCH[0:48]

matrice_PBCH_user_demod=bpsk_demod(matrice_PBCH_user.real)
bitDec = hamming748_decode(matrice_PBCH_user_demod)
print("Flux de bits après le décodage Hamming748 des informations PBCH : ", bitDec)

print("Cell ident : ", bin2dec(bitDec[0:18]))
print("Nb Users : ", bin2dec(bitDec[18:24]))

# TODO : Change the user ident
vecteur_PBCH = matrice_without_sync.flatten()
PBCHU = getPBCHU(matrice_PBCH=vecteur_PBCH[48:48*19], ident=11)
print(PBCHU)
