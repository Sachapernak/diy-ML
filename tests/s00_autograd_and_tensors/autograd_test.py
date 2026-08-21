"""
Tests pour BasicNDArray

Lancer avec pytest test_ndarray.py -v ? voir uv run
Si test qui plante :  pytest test_ndarray.py::test_nom -v

A pas oublier : strides NumPy sont en octet, mais les miens sont e, nb d'octets
comparer avec np_arr.strides == tuple(s * np_arr.itemsize for s in ndarr.strides)
"""

import pytest
import numpy as np

from src.s00_autograd_and_tensors.autograd import BasicNDArray


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

def test_shape_matrice_2x3(data_2x3):
    # attendu : (2, 3)
    np_arr = np.array(data_2x3)
    basic = BasicNDArray(data_2x3)
    assert basic.shape == np_arr.shape



def test_shape_3d(data_2x3x4):
    np_arr = np.array(data_2x3x4)
    basic = BasicNDArray(data_2x3x4)
    assert basic.shape == np_arr.shape


def test_shape_un_seul_axe():
    np_arr = np.array([1.0, 2.0, 3.0])
    basic = BasicNDArray([1.0, 2.0, 3.0])
    assert basic.shape == np_arr.shape


def test_shape_scalaire():
    # BasicNDArray(5.0) -> reflechir a la shape que c'est censé donner.
    np_arr = np.array(5.0)
    basic = BasicNDArray(5.0)
    assert basic.shape == np_arr.shape


# ---------------------------------------------------------------------------
# 2. Strides
#    (dont test unitaire _get_stride)
# ---------------------------------------------------------------------------

def test_strides_2_3(data_2x3x4):
    np_arr = np.array(data_2x3x4)
    basic = BasicNDArray(data_2x3x4)
    assert basic.strides == tuple(s / np_arr.itemsize for s in np_arr.strides)


def test_strides_un_axe():
    np_arr = np.array([1.0, 2.0, 3.0])
    basic = BasicNDArray([1.0, 2.0, 3.0])
    assert basic.strides == tuple(s / np_arr.itemsize for s in np_arr.strides)


def test_strides_shape_vide():
    np_arr = np.array(5.0)
    basic = BasicNDArray(5.0)
    assert basic.strides == tuple(s / np_arr.itemsize for s in np_arr.strides)



# ---------------------------------------------------------------------------
# 3. Buffer / aplatissement
# ---------------------------------------------------------------------------

def test_buffer_ordre_row_major(arr_2x3):
    # [[0,1,2],[3,4,5]] -> [0,1,2,3,4,5]
    assert arr_2x3._array == [0,1,2,3,4,5]


def test_taille_buffer_egale_produit_shape(arr_2x3x4):
    # len(buffer) -> 2*3*4 (invariant)
    assert len(arr_2x3x4._array) == 2*3*4


def test_flatten_rejette_longueurs_inegales():
    # [[1, 2], [3]]
    with pytest.raises(ValueError):
        basic = BasicNDArray([[1, 2], [3]])


def test_flatten_rejette_melange_feuille_branche():
    # [[1, 2], 3]
    with pytest.raises(ValueError):
        basic = BasicNDArray([[1, 2], 3])


# ---------------------------------------------------------------------------
# 4. Indexation
# ---------------------------------------------------------------------------

def test_indexation_complete_tous_les_points(arr_2x3x4, data_2x3x4):
    # verifier un a un en comparant avec np
    ref = np.array(data_2x3x4)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert (arr_2x3x4[i, j, k] == ref[i, j, k])


def test_composition_offset(arr_2x3x4):
    # check si t[1][2][3] == t[1, 2, 3].
    assert arr_2x3x4[1][2][3] == arr_2x3x4[1, 2, 3]


def test_indexation_partielle_shape(arr_2x3x4):
    # t[1] -> shape (3, 4), t[1][2] -> shape (4,)
    assert arr_2x3x4[1].shape == (3, 4)
    assert arr_2x3x4[1][2].shape == (4,)


def test_index_negatif(arr_2x3):
    # t[-1] == t[1], t[-1, -1] == t[1, 2]
    assert arr_2x3[-1].shape == arr_2x3[1].shape
    assert arr_2x3[-1].strides == arr_2x3[1].strides
    assert arr_2x3[-1]._offset == arr_2x3[1]._offset

    assert arr_2x3[-1, -1] == arr_2x3[1, 2]


def test_hors_bornes_leve_indexerror(arr_2x3):
    with pytest.raises(IndexError):
        arr = arr_2x3[20]


def test_negatif_hors_bornes_leve_indexerror(arr_2x3):
    # t[-5] sur un axe de taille 2
    with pytest.raises(IndexError):
        arr = arr_2x3[-5]


def test_trop_d_indices_leve_indexerror(arr_2x3):
    with pytest.raises(IndexError):
        arr = arr_2x3[1,1,1]


# ---------------------------------------------------------------------------
# 5. Vues
# ---------------------------------------------------------------------------

def test_vue_partage_le_buffer(arr_2x3x4):
    #check avec "is" pour voir si c'est le meme buffer comme avec un ndarray de np"""
    assert arr_2x3x4._array is arr_2x3x4[1]._array


def test_vue_offset_correct(arr_2x3x4):
    # t[1] -> offset 12
    assert arr_2x3x4[1]._offset == 12


def test_vue_strides_correct(arr_2x3x4):
    # t[1] -> strides (4, 1)
    assert arr_2x3x4[1].strides == (4,1 )


# ---------------------------------------------------------------------------
# 6. Cas limites
# ---------------------------------------------------------------------------

def test_liste_vide():
    # BasicNDArray([]) -> relechir a ce que c'est censé faire
    arr = BasicNDArray([])
    assert arr._array == []
    assert arr.strides == (1,)
    assert arr.shape == (0,)


def test_scalaire_indexation():
    # BasicNDArray(5.0)[()] -> ?
    assert BasicNDArray(5.0)[()] == 5.0


# ---------------------------------------------------------------------------
# 7. Transpose
# ---------------------------------------------------------------------------

def test_transpose_shared_array(arr_2x3):
    transposed = arr_2x3.transpose()

    transposed._array[0] = 10.0
    # doit overwrite array[0] = 10.0
    arr_2x3._array[0] = 9.0

    assert arr_2x3._array == transposed._array
    assert arr_2x3._array is transposed._array

def test_transpose_stride_shape_inverse(arr_2x3):
    transposed = arr_2x3.transpose()

    assert arr_2x3.shape[::-1] == transposed.shape
    assert arr_2x3.strides[::-1] == transposed.strides

def test_transpose_1d_array():
    basic = BasicNDArray([1, 2, 3])
    assert basic.array_equal(basic.transpose())

def test_transpose_valeurs(arr_2x3):
    transposed = arr_2x3.transpose()
    for i in range(arr_2x3.shape[0]):
        for j in range(arr_2x3.shape[1]):
            assert arr_2x3[i,j] == transposed[j,i]

def test_transpose_array_plus_contigue(arr_2x3x4):
        transposed = arr_2x3x4.T

        assert transposed.strides != BasicNDArray._get_stride(list(transposed.shape))

def test_transpose_avec_axes(arr_2x3x4):
    transposed = arr_2x3x4.transpose((2,0,1))

    for i in range(arr_2x3x4.shape[0]):
        for j in range(arr_2x3x4.shape[1]):
            for k in range(arr_2x3x4.shape[2]):
                assert arr_2x3x4[i,j,k] == transposed[k,i,j]

def test_double_transpose(arr_2x3x4):
    d_transposed = arr_2x3x4.T.T
    assert d_transposed.array_equal(arr_2x3x4)

def test_transpose_pas_assez_axes(arr_2x3x4):
    with pytest.raises(ValueError):
        arr_2x3x4.transpose((3,))

def test_transpose_axes_doublons(arr_2x3x4):
    with pytest.raises(ValueError):
        arr_2x3x4.transpose((2,1,1))

def test_transpose_out_of_bound(arr_2x3x4):
    with pytest.raises(ValueError):
        arr_2x3x4.transpose((0,1,3))

# ---------------------------------------------------------------------------
# 8. Flat
# ---------------------------------------------------------------------------
def test_flat_contigue():
    basic = BasicNDArray([[1, 2], [3,4]])
    assert list(basic.flat()) == basic._array

def test_flat_sur_vue(arr_2x3x4):
    assert list(arr_2x3x4[1].flat()) == list(range(12, 24))

def test_flat_change_avec_transpose(arr_2x3):
    assert list(arr_2x3.T.flat()) != arr_2x3._array

def test_flat_scalaire():
    basic = BasicNDArray(3.0)
    assert next(basic.flat()) == 3.0

def test_flat_vide():
    basic = BasicNDArray([])
    assert len(list(basic.flat())) == 0

def test_taille_flat(arr_2x3x4):
    assert len(list(arr_2x3x4.flat())) == arr_2x3x4.size

# ---------------------------------------------------------------------------
# 9. Iter
# ---------------------------------------------------------------------------

def test_iter_longueur(arr_2x3):
    assert len(list(arr_2x3)) == len(arr_2x3)


def test_iter_coherent_avec_getitem(arr_2x3x4):
    sous_tableaux = list(arr_2x3x4)
    for i, sous in enumerate(sous_tableaux):
        assert sous.array_equal(arr_2x3x4[i])


def test_iter_1d_produit_des_floats():
    basic = BasicNDArray([1.0, 2.0, 3.0])
    for element in basic:
        assert isinstance(element, float)


def test_iter_2d_produit_des_ndarray(arr_2x3):
    for element in arr_2x3:
        assert isinstance(element, BasicNDArray)
        assert element.shape == (3,)


def test_iter_scalaire_leve_typeerror():
    basic = BasicNDArray(5.0)
    with pytest.raises(TypeError):
        list(basic)


def test_iter_apres_transpose(arr_2x3):
    # 2x3 transpose -> 3x2 : trois sous-tableaux de taille 2
    transposed = arr_2x3.T
    sous = list(transposed)
    assert len(sous) == 3
    assert all(s.shape == (2,) for s in sous)


# ---------------------------------------------------------------------------
# 10. len / size / nb_dim / is_scalar
# ---------------------------------------------------------------------------

def test_len_egale_premier_axe(arr_2x3, arr_2x3x4):
    assert len(arr_2x3) == 2
    assert len(arr_2x3x4) == 2


def test_len_scalaire_leve_typeerror():
    with pytest.raises(TypeError):
        len(BasicNDArray(5.0))


def test_size_3d(arr_2x3x4):
    assert arr_2x3x4.size == 24


def test_size_scalaire():
    assert BasicNDArray(5.0).size == 1


def test_size_vide():
    assert BasicNDArray([]).size == 0


def test_size_invariant_avec_flat(arr_2x3, arr_2x3x4):
    for arr in (arr_2x3, arr_2x3x4, arr_2x3.T, arr_2x3x4[1]):
        assert arr.size == len(list(arr.flat()))


def test_nb_dim_coherent(arr_2x3x4):
    assert arr_2x3x4.nb_dim == len(arr_2x3x4.shape) == len(arr_2x3x4.strides)


def test_nb_dim_valeurs():
    assert BasicNDArray(5.0).nb_dim == 0
    assert BasicNDArray([1.0, 2.0]).nb_dim == 1
    assert BasicNDArray([[1.0, 2.0]]).nb_dim == 2


def test_is_scalar():
    assert BasicNDArray(5.0).is_scalar
    assert not BasicNDArray([1.0]).is_scalar
    assert not BasicNDArray([]).is_scalar


def test_vue_reduit_nb_dim(arr_2x3x4):
    assert arr_2x3x4[0].nb_dim == 2
    assert arr_2x3x4[0][0].nb_dim == 1


# ---------------------------------------------------------------------------
# 11. Repr
# ---------------------------------------------------------------------------

def test_repr_ne_plante_pas(arr_2x3, arr_2x3x4):
    for arr in (BasicNDArray(5.0), BasicNDArray([]), BasicNDArray([1.0, 2.0]), arr_2x3, arr_2x3x4):
        assert isinstance(repr(arr), str)


def test_repr_1d():
    assert repr(BasicNDArray([1.0, 2.0, 3.0])) == "[1.0, 2.0, 3.0]"


def test_repr_2d(arr_2x3):
    assert repr(arr_2x3) == "[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]"


def test_repr_vide():
    assert repr(BasicNDArray([])) == "[]"


def test_repr_reflete_le_transpose(arr_2x3):
    # 2x3 -> 3x2 : l'affichage doit suivre les strides, pas le buffer
    assert repr(arr_2x3.T) == "[[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]"


# ---------------------------------------------------------------------------
# 12. Indexation
# ---------------------------------------------------------------------------

def test_hors_bornes_deuxieme_axe(arr_2x3):
    with pytest.raises(IndexError):
        invalid = arr_2x3[0, 7]


def test_hors_bornes_troisieme_axe(arr_2x3x4):
    with pytest.raises(IndexError):
        invalid = arr_2x3x4[0, 0, 99]


def test_indexation_vue_shape(arr_2x3x4):
    assert arr_2x3x4[1].shape == (3, 4)
    assert arr_2x3x4[1][2].shape == (4,)
    assert isinstance(arr_2x3x4[1][2][3], float)


def test_indexer_un_scalaire_leve_indexerror():
    with pytest.raises(IndexError):
        invalid = BasicNDArray(5.0)[0]


def test_indexation_apres_transpose(arr_2x3x4, data_2x3x4):
    ref = np.array(data_2x3x4).transpose(2, 0, 1)
    transposed = arr_2x3x4.transpose((2, 0, 1))
    for i in range(transposed.shape[0]):
        for j in range(transposed.shape[1]):
            for k in range(transposed.shape[2]):
                assert transposed[i, j, k] == ref[i, j, k]


def test_vue_de_vue_partage_le_buffer(arr_2x3x4):
    assert arr_2x3x4[1][2]._array is arr_2x3x4._array


# ---------------------------------------------------------------------------
# 13. Broadcast
# ---------------------------------------------------------------------------

def test_broadcast_ajoute_un_axe():
    basic = BasicNDArray([1.0, 2.0, 3.0])
    b = basic.broadcast_to((2, 3))
    assert b.shape == (2, 3)
    assert b.strides == (0, 1)


def test_broadcast_ne_copie_pas():
    basic = BasicNDArray([1.0, 2.0, 3.0])
    b = basic.broadcast_to((100, 3))

    assert b._array is basic._array
    assert len(b._array) == 3


def test_broadcast_valeurs_repetees():
    basic = BasicNDArray([1.0, 2.0, 3.0])
    b = basic.broadcast_to((4, 3))
    for i in range(4):
        for j in range(3):
            assert b[i, j] == basic[j]


def test_broadcast_axe_interne():
    # (2, 1) -> (2, 3) : c'est le deuxieme axe qui est etendu
    basic = BasicNDArray([[1.0], [2.0]])
    b = basic.broadcast_to((2, 3))
    assert b.shape == (2, 3)
    assert b.strides == (1, 0)
    for i in range(2):
        for j in range(3):
            assert b[i, j] == basic[i, 0]


def test_broadcast_scalaire():
    basic = BasicNDArray(7.0)
    b = basic.broadcast_to((2, 3))
    assert b.shape == (2, 3)
    assert b.strides == (0, 0)
    assert all(v == 7.0 for v in b.flat())


def test_broadcast_shape_identique(arr_2x3):
    b = arr_2x3.broadcast_to((2, 3))
    assert b.shape == arr_2x3.shape
    assert b.strides == arr_2x3.strides
    assert b.array_equal(arr_2x3)


def test_broadcast_plusieurs_axes_ajoutes():
    basic = BasicNDArray([1.0, 2.0])
    b = basic.broadcast_to((3, 4, 2))
    assert b.shape == (3, 4, 2)
    assert b.strides == (0, 0, 1)


def test_broadcast_size_vs_buffer():
    basic = BasicNDArray([1.0, 2.0, 3.0])
    b = basic.broadcast_to((5, 3))
    assert b.size == 15
    assert len(list(b.flat())) == 15
    assert len(b._array) == 3


def test_broadcast_contre_numpy():
    data = [[1.0], [2.0]]
    basic = BasicNDArray(data).broadcast_to((2, 3))
    ref = np.broadcast_to(np.array(data), (2, 3))
    for i in range(2):
        for j in range(3):
            assert basic[i, j] == ref[i, j]


def test_broadcast_rejette_shape_trop_courte(arr_2x3):
    with pytest.raises(ValueError):
        arr_2x3.broadcast_to((3,))


def test_broadcast_rejette_dimension_incompatible():
    basic = BasicNDArray([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        basic.broadcast_to((2, 5))  # 3 -> 5 impossible


def test_broadcast_rejette_reduction():
    basic = BasicNDArray([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        basic.broadcast_to((2, 2))  # 3 -> 2 impossible


def test_broadcast_puis_transpose():
    basic = BasicNDArray([1.0, 2.0, 3.0])
    b = basic.broadcast_to((2, 3)).T
    assert b.shape == (3, 2)
    assert b.strides == (1, 0)
    for i in range(3):
        for j in range(2):
            assert b[i, j] == basic[i]


# ---------------------------------------------------------------------------
# 14. Matmul 2D
# ---------------------------------------------------------------------------

@pytest.fixture
def mat_2x3():
    return BasicNDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def mat_3x4():
    return BasicNDArray([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0, 12.0]])


def test_matmul_shape(mat_2x3, mat_3x4):
    res = mat_2x3.matmul(mat_3x4)
    assert res.shape == (2, 4)


def test_matmul_contre_numpy(mat_2x3, mat_3x4):
    res = mat_2x3.matmul(mat_3x4)
    ref = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) @ np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    for i in range(2):
        for j in range(4):
            assert res[i, j] == pytest.approx(ref[i, j])


def test_matmul_identite(mat_2x3):
    identite = BasicNDArray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    res = mat_2x3.matmul(identite)
    assert res.array_equal(mat_2x3)


def test_matmul_carre():
    a = BasicNDArray([[1.0, 2.0], [3.0, 4.0]])
    b = BasicNDArray([[5.0, 6.0], [7.0, 8.0]])
    res = a.matmul(b)
    attendu = BasicNDArray([[19.0, 22.0], [43.0, 50.0]])
    assert res.array_equal(attendu)


def test_matmul_non_commutatif():
    a = BasicNDArray([[1.0, 2.0], [3.0, 4.0]])
    b = BasicNDArray([[5.0, 6.0], [7.0, 8.0]])
    assert not a.matmul(b).array_equal(b.matmul(a))


def test_matmul_alloue_un_nouveau_buffer(mat_2x3, mat_3x4):
    res = mat_2x3.matmul(mat_3x4)
    assert res._array is not mat_2x3._array
    assert res._array is not mat_3x4._array
    assert len(res._array) == 8


def test_matmul_resultat_contigu(mat_2x3, mat_3x4):
    res = mat_2x3.matmul(mat_3x4)
    assert list(res.strides) == BasicNDArray._get_stride(list(res.shape))
    assert res._offset == 0


def test_matmul_sur_transpose(mat_2x3):
    # (3,2) @ (2,3) -> (3,3)
    res = mat_2x3.T.matmul(mat_2x3)
    ref = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ref = ref.T @ ref
    assert res.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            assert res[i, j] == pytest.approx(ref[i, j])


def test_matmul_sur_vue(arr_2x3x4, data_2x3x4):
    gauche = arr_2x3x4[0].T          # (4,3)
    droite = arr_2x3x4[1]            # (3,4)
    res = gauche.matmul(droite)
    ref = np.array(data_2x3x4)[0].T @ np.array(data_2x3x4)[1]
    assert res.shape == (4, 4)
    for i in range(4):
        for j in range(4):
            assert res[i, j] == pytest.approx(ref[i, j])


def test_matmul_sur_broadcast():
    v = BasicNDArray([1.0, 2.0, 3.0]).broadcast_to((2, 3))   # (2,3), strides (0,1)
    m = BasicNDArray([[1.0], [1.0], [1.0]])                   # (3,1)
    res = v.matmul(m)
    assert res.shape == (2, 1)
    assert res[0, 0] == pytest.approx(6.0)
    assert res[1, 0] == pytest.approx(6.0)


def test_matmul_dimension_interne_incompatible(mat_2x3):
    autre = BasicNDArray([[1.0, 2.0], [3.0, 4.0]])   # (2,2), 3 != 2
    with pytest.raises(ValueError):
        mat_2x3.matmul(autre)


def test_matmul_rejette_1d(mat_2x3):
    vecteur = BasicNDArray([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        mat_2x3.matmul(vecteur)


def test_matmul_rejette_scalaire(mat_2x3):
    with pytest.raises(ValueError):
        mat_2x3.matmul(BasicNDArray(2.0))


def test_matmul_rejette_3d(mat_2x3, arr_2x3x4):
    with pytest.raises(ValueError):
        mat_2x3.matmul(arr_2x3x4)


def test_operateur_arobase(mat_2x3, mat_3x4):
    """Ne passe que si __matmul__ est defini."""
    assert (mat_2x3 @ mat_3x4).array_equal(mat_2x3.matmul(mat_3x4))


# ---------------------------------------------------------------------------
# 15. Matmul 1D  (squelettes)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="pas imp")
def test_matmul_vecteur_fois_matrice():
    # (3,) @ (3, 4) -> (4,)
    ...

@pytest.mark.skip(reason="pas imp")
def test_matmul_matrice_fois_vecteur():
    # (2, 3) @ (3,) -> (2,)
    ...

@pytest.mark.skip(reason="pas imp")
def test_matmul_produit_scalaire():
    # (3,) @ (3,) -> 0-d
    ...

@pytest.mark.skip(reason="pas imp")
def test_matmul_1d_contre_numpy():
    ...

@pytest.mark.skip(reason="pas imp")
def test_matmul_1d_dimension_incompatible():
    # (3,) @ (4, 2) doit lever
    ...


# ---------------------------------------------------------------------------
# 16. Reshape
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="pas imp")
def test_reshape_shape():
    # (2, 3) -> (3, 2), puis -> (6,)
    ...

@pytest.mark.skip(reason="pas imp")
def test_reshape_conserve_l_ordre_row_major():
    # les valeurs doivent se lire dans le meme ordre qu'avant
    ...

@pytest.mark.skip(reason="pas imp")
def test_reshape_contigu_ne_copie_pas():
    # buffer partage (is) quand le tableau de depart est contigu
    ...

@pytest.mark.skip(reason="pas imp")
def test_reshape_recalcule_les_strides():
    # (2,3) contigu -> (3,2) : strides (2, 1)
    ...

@pytest.mark.skip(reason="pas imp")
def test_reshape_apres_transpose():
    # t.T est plus contigu.
    # voir si raise erreur ou copier le truc
    ...

@pytest.mark.skip(reason="pas imp")
def test_reshape_taille_incompatible():
    # (2, 3) -> (4, 2) : 6 != 8 donc pas compatible
    ...

@pytest.mark.skip(reason="pas imp")
def test_reshape_scalaire():
    # 0-d -> (1,) et (1,) -> 0-d
    ...

@pytest.mark.skip(reason="pas imp")
def test_reshape_contre_numpy():
    ...

@pytest.mark.skip(reason="pas imp")
def test_is_contiguous():
    # strides == _get_stride(shape) vrai sur nouveaux tab et faux apres transpose
    ...