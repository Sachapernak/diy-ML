"""
Tests pour BasicNDArray

Lancer avec pytest test_ndarray.py -v ? voir uv run
Si test qui plante :  pytest test_ndarray.py::test_nom -v

A pas oublier : strides NumPy sont en octet, mais les miens sont e, nb d'octets
comparer avec np_arr.strides == tuple(s * np_arr.itemsize for s in ndarr.strides)
"""

import pytest
import numpy as np

from s00_autograd_and_tensors.autograd import BasicNDArray


# ---------------------------------------------------------------------------
# Array de reference
# ---------------------------------------------------------------------------

@pytest.fixture
def data_2x3():
    """[[0, 1, 2], [3, 4, 5]]"""
    return [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


@pytest.fixture
def data_2x3x4():
    """Valeurs 0..23 : valeur == l'index buffer."""
    return [[[float(i * 12 + j * 4 + k) for k in range(4)] for j in range(3)] for i in range(2)]


@pytest.fixture
def arr_2x3(data_2x3):
    return BasicNDArray(data_2x3)


@pytest.fixture
def arr_2x3x4(data_2x3x4):
    return BasicNDArray(data_2x3x4)


# ---------------------------------------------------------------------------
# 1. Shape
# ---------------------------------------------------------------------------

def test_shape_matrice_2x3(arr_2x3):
    # attendu : (2, 3)
    ...


def test_shape_3d(arr_2x3x4):
    ...


def test_shape_un_seul_axe():
    # BasicNDArray([1.0, 2.0, 3.0])
    ...


def test_shape_scalaire():
    # BasicNDArray(5.0) -> reflechir a la shape que c'est censé donner.
    ...


# ---------------------------------------------------------------------------
# 2. Strides
#    (dont test unitaire _get_stride)
# ---------------------------------------------------------------------------

def test_strides_2_3_4():
    # BasicNDArray._get_stride([2, 3, 4]) -> ?
    ...


def test_strides_un_axe():
    ...


def test_strides_shape_vide():
    ...


def test_strides_coherents_avec_numpy(data_2x3x4):
    # Comparer avec np.array(data_2x3x4).strides (en prenant compte le truc des octets np)
    ...


# ---------------------------------------------------------------------------
# 3. Buffer / aplatissement
# ---------------------------------------------------------------------------

def test_buffer_ordre_row_major(arr_2x3):
    # [[0,1,2],[3,4,5]] -> [0,1,2,3,4,5]
    ...


def test_taille_buffer_egale_produit_shape(arr_2x3x4):
    # len(buffer) -> 2*3*4 (invariant)
    ...


def test_flatten_rejette_longueurs_inegales():
    # [[1, 2], [3]]
    with pytest.raises(ValueError):
        ...


def test_flatten_rejette_melange_feuille_branche():
    # [[1, 2], 3]
    with pytest.raises(ValueError):
        ...


# ---------------------------------------------------------------------------
# 4. Indexation
# ---------------------------------------------------------------------------

def test_indexation_complete_tous_les_points(arr_2x3x4, data_2x3x4):
    # verifier un a un en comparant avec np
    ref = np.array(data_2x3x4)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                ...


def test_composition_offset(arr_2x3x4):
    # check si t[1][2][3] == t[1, 2, 3].
    ...


def test_indexation_partielle_shape(arr_2x3x4):
    # t[1] -> shape (3, 4), t[1][2] -> shape (4,)
    ...


def test_index_negatif(arr_2x3):
    # t[-1] == t[1], t[-1, -1] == t[1, 2]
    ...


def test_hors_bornes_leve_indexerror(arr_2x3):
    with pytest.raises(IndexError):
        ...


def test_negatif_hors_bornes_leve_indexerror(arr_2x3):
    # t[-5] sur un axe de taille 2
    with pytest.raises(IndexError):
        ...


def test_trop_d_indices_leve_indexerror(arr_2x3):
    with pytest.raises(IndexError):
        ...


# ---------------------------------------------------------------------------
# 5. Vues
# ---------------------------------------------------------------------------

def test_vue_partage_le_buffer(arr_2x3x4):
    #check avec "is" pour voir si c'est le meme buffer comme avec un ndarray de np"""
    ...


def test_vue_offset_correct(arr_2x3x4):
    # t[1] -> offset 12
    ...


def test_vue_strides_correct(arr_2x3x4):
    # t[1] -> strides (4, 1)
    ...


# ---------------------------------------------------------------------------
# 6. Cas limites
# ---------------------------------------------------------------------------

def test_liste_vide():
    # BasicNDArray([]) -> relechir a ce que c'est censé faire
    ...


def test_scalaire_indexation():
    # BasicNDArray(5.0)[()] -> ?
    ...


# ---------------------------------------------------------------------------
# 7. Transpose (a faire)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="transpose pas encore implemente")
def test_transpose_ne_copie_pas(arr_2x3):
    ...


@pytest.mark.skip(reason="transpose pas encore implemente")
def test_transpose_valeurs(arr_2x3):
    # pour tout (i, j) : t.T[i, j] == t[j, i]
    ...