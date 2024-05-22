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

from qam16_demod import *
from crc import *
from binary_transformation import *

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
                PBCHU = {"userIdent": userIdent, "pdcchuMCS": pdcchuMCS, "pdcchuSymbStart": pdcchuSymbStart, "pdcchuRbStart": pdcchuRbStart, "pdcchuHARQ": pdcchuHARQ}
                return PBCHU

def qpsk_demod(matrice):
    result = []
    for complex in matrice:
        result.append(1 if complex.real >= 0 else 0)
        result.append(1 if complex.imag >= 0 else 0)

    return result

def decodePDCCHU(matrice, mcsFlag):
    bitDec = []
    # BPSK
    if mcsFlag == 0:
        bitDec = bpsk_demod(matrice.real)
    # QPSK
    elif mcsFlag == 2:
        bitDec = qpsk_demod(matrice)
    
    return hamming748_decode(bitDec)

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
nbUsers = bin2dec(bitDec[18:24])
print("Nb Users : ", nbUsers)

# TODO : Change the user ident
vecteur_PBCH = matrice_without_sync.flatten()
PBCHU = getPBCHU(matrice_PBCH=vecteur_PBCH[48:48* (nbUsers + 1)], ident=11)

print(PBCHU)
qam_size = 72 if PBCHU["pdcchuMCS"] == 0 else 36
qamSequence = vecteur_PBCH[(PBCHU["pdcchuSymbStart"]-3) * 624 + (PBCHU["pdcchuRbStart"] -1) * 12:(PBCHU["pdcchuSymbStart"]-3) * 624 + (PBCHU["pdcchuRbStart"]- 1) * 12 + qam_size]

qamSequecenceDecode = decodePDCCHU(qamSequence, PBCHU["pdcchuMCS"])

# Figure 1.12: PDCCHU informations page 16
userIdent = bin2dec(qamSequecenceDecode[0:8])
pdschuMCS = bin2dec(qamSequecenceDecode[8:14])
pdschuSymbStart = bin2dec(qamSequecenceDecode[14:18])
pdschuRbStart = bin2dec(qamSequecenceDecode[18:24])
pdschuRbSize = bin2dec(qamSequecenceDecode[24:34])
crcFlag = bin2dec(qamSequecenceDecode[34:36])

PDSCH = {"userIdent": userIdent, "pdschuMCS": pdschuMCS, "pdschuSymbStart": pdschuSymbStart, "pdschuRbStart": pdschuRbStart, "pdschuRbSize": pdschuRbSize, "crcFlag": crcFlag}
print(PDSCH)

#We provide the 16-QAM decoding function. Check that the test test_qam16.m passes.
# pytest tests_modulation.py::test_qam16

def pdsch_demod(qamSeq, mcs):
    if (mcs > 19):
        raise NotImplementedError("Higher modulation are not supported")

    if mcs % 5 == 0:
        return bpsk_demod(qamSeq)
    elif (mcs - 1) % 5 == 0:
        return qpsk_demod(qamSeq)
    elif (mcs - 2) % 5 == 0:
        return qam16_demod(qamSeq)

def pdsch_fec(qamSeq, mcs):
    # Valeurs de MCS de constellation BPSK, QPSK et 16-QAM correspondant à un codage Hamming748 (supporté dans le projet)
    if mcs in [25, 26, 27]:
        return hamming748_decode(qamSeq)
    else:
        raise NotImplementedError("Hamming124, Hamming128 and Hamming2416 are not supported")

# The CRC decoding functions are provided and you will have to use crcDecode and getCRCPoly. Test with the unitary test function test_crcDecode
# pytest crc.py::test_crcGen crc.py::test_crcDecode

def pdsch_crc(qamSeq, crcFlag):
    crcSize = crcFlag + 1 * 8
    gx = get_crc_poly(crcSize)
    crc_check = crc_decode(qamSeq, gx)

    # Return true if the CRC is correct
    return crc_check == 1


pdschSeq = vecteur_PBCH[(PDSCH["pdschuSymbStart"]-3) * 624 + (PDSCH["pdschuRbStart"] -1) * 12:(PDSCH["pdschuRbStart"]-3) * 624 + (PDSCH["pdschuRbStart"]- 1) * 12 + PDSCH["pdschuRbSize"] * 12]

pdschSeqDemod = pdsch_demod(pdschSeq, PDSCH["pdschuMCS"])
    
cc1 = fec.FECConv(("1011011","1111001"),6)
dec = cc1.viterbi_decoder(np.array(pdschSeqDemod).astype(int),"hard")
crc = pdsch_crc(dec, PDSCH["crcFlag"])

if crc:
    # Assuming the message decoding from QPSK or QAM16 is qamSeq
    # Convert the binary sequence into bytes
    mess = bitToByte(dec)
    # Bytes are "encrypted", uncrypt them
    real_mess = cesarDecode(11,mess); # USER is your user group
    final_mess = toASCII(real_mess)
    print(final_mess)