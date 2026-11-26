# generators.py
import random
from typing import List, Dict, Any
from .registry import registry
from itertools import permutations
from itertools import product


@registry.register("param_generators", "alphabetical_elements")
def generate_alphabetical_elements(count: int, prefix: str = "") -> List[str]:
    """Generates a list of named elements like 'a', 'b', 'c'...'"""
    if count > 26 and not prefix:
        print(
            "Warning: More than 26 elements, consider using a prefix for clarity."
        )
    return [f"{prefix}{chr(ord('a') + i)}" for i in range(count)]


@registry.register("param_generators", "modular_addition_table")
def generate_modular_addition_table(size: int) -> List[List[int]]:
    """Generates the Cayley table for addition modulo size."""
    return [[(i + j) % size for j in range(size)] for i in range(size)]


@registry.register("param_generators", "random_latin_square")
def generate_random_latin_square(size: int,
                                 seed: int = None) -> List[List[int]]:
    """Generates a random Latin square, which forms a quasigroup."""
    if seed is not None:
        random.seed(seed)

    # A simple but effective algorithm for generating a Latin Square
    square = [[(i + j) % size for j in range(size)] for i in range(size)]

    # Shuffle rows randomly
    random.shuffle(square)

    # Shuffle columns by transposing, shuffling, and transposing back
    transposed = list(zip(*square))
    random.shuffle(transposed)
    square = [list(row) for row in zip(*transposed)]

    # To make it less predictable, randomly permute the symbols (0..n-1)
    symbols = list(range(size))
    random.shuffle(symbols)

    return [[symbols[val] for val in row] for row in square]


def _find_permutation_cycles(p: tuple) -> str:
    """Helper to convert a permutation tuple like (1, 2, 0) to cycle notation string like '(0 1 2)'."""
    n = len(p)
    not_visited = [True] * n
    cycles = []
    for i in range(n):
        if not_visited[i]:
            cycle = []
            j = i
            while not_visited[j]:
                not_visited[j] = False
                cycle.append(j)
                j = p[j]
            if len(cycle) > 1:
                cycles.append(" ".join(map(str, cycle)))
    if not cycles:
        return "()"  # Identity permutation
    return "".join(f"({c})" for c in cycles)


@registry.register("param_generators", "permutation_elements")
def generate_permutation_elements(n: int) -> List[str]:
    """Generates human-readable names for all permutations in S_n using cycle notation."""
    perms = list(permutations(range(n)))
    # Convert to cycle notation for display
    # We will map these names to integer indices 0, 1, 2...
    return [_find_permutation_cycles(p) for p in perms]


@registry.register("param_generators", "permutation_composition_table")
def generate_permutation_composition_table(n: int) -> List[List[int]]:
    """Generates the full Cayley table for the symmetric group S_n."""
    # The canonical list of permutations
    perms_list = list(permutations(range(n)))

    # A reverse map from permutation tuple to its index for quick lookup
    perm_to_index = {p: i for i, p in enumerate(perms_list)}

    num_perms = len(perms_list)
    table = [[0] * num_perms for _ in range(num_perms)]

    for i in range(num_perms):
        for j in range(num_perms):
            p1 = perms_list[i]
            p2 = perms_list[j]

            # Composition p1 ∘ p2 means apply p2 then p1
            # p_comp[k] = p1[p2[k]]
            p_comp = tuple(p1[p2[k]] for k in range(n))

            # Find the index of the resulting permutation
            result_index = perm_to_index[p_comp]
            table[i][j] = result_index

    return table


@registry.register("param_generators", "dihedral_group_table")
def generate_dihedral_group_table(n_gon: int) -> List[List[int]]:
    """Generates the Cayley table for the Dihedral group D_n (symmetries of an n-gon)."""
    if n_gon != 4:
        raise NotImplementedError(
            "This generator is currently hardcoded for D4 (a square).")

    # D4 has 8 elements. We map them to integers 0-7:
    # 0-3: Rotations R0, R90, R180, R270
    # 4-7: Reflections H, V, D1, D2

    # Element names for reference: ["R0", "R90", "R180", "R270", "H", "V", "D1", "D2"]
    R0, R90, R180, R270 = 0, 1, 2, 3
    H, V, D1, D2 = 4, 5, 6, 7

    # Hardcoded Cayley table for D4. This is the result of composing the transformations.
    # For example, table[R90][H] = D2, so table[1][4] = 7.
    return [
        # R0   R90  R180 R270   H    V    D1   D2   (Applied second)
        #----------------------------------------------------
        [R0, R90, R180, R270, H, V, D1, D2],  # R0
        [R90, R180, R270, R0, D1, D2, V, H],  # R90
        [R180, R270, R0, R90, V, H, D2, D1],  # R180
        [R270, R0, R90, R180, D2, D1, H, V],  # R270
        [H, D2, V, D1, R0, R180, R270, R90],  # H
        [V, D1, H, D2, R180, R0, R90, R270],  # V
        [D1, H, D2, V, R90, R270, R0, R180],  # D1
        [D2, V, D1, H, R270, R90, R180, R0]  # D2
    ]


# 在 generators.py 文件末尾添加


@registry.register("param_generators", "direct_product_elements")
def generate_direct_product_elements(elements_a: List[str],
                                     elements_b: List[str]) -> List[str]:
    """Generates string names for the Cartesian product of two element sets."""
    return [f"({a},{b})" for a in elements_a for b in elements_b]


@registry.register("param_generators", "direct_product_table")
def generate_direct_product_table(table_a: List[List[int]],
                                  table_b: List[List[int]]) -> List[List[int]]:
    """
    Generates the Cayley table for the direct product of two groups G x H.
    The operation is component-wise: (g1, h1) * (g2, h2) = (g1*g2, h1*h2).
    """
    size_a = len(table_a)
    size_b = len(table_b)
    new_size = size_a * size_b

    new_table = [[0] * new_size for _ in range(new_size)]

    for k1 in range(new_size):
        # Decode index k1 into its components (i_a, i_b) from G and H
        i_a = k1 // size_b
        i_b = k1 % size_b

        for k2 in range(new_size):
            # Decode index k2 into its components (j_a, j_b)
            j_a = k2 // size_b
            j_b = k2 % size_b

            # Perform the operation component-wise
            res_a = table_a[i_a][j_a]
            res_b = table_b[i_b][j_b]

            # Encode the result back into a single index
            res_k = res_a * size_b + res_b
            new_table[k1][k2] = res_k

    return new_table


@registry.register("param_generators", "bitwise_xor_table")
def generate_bitwise_xor_table(size: int) -> List[List[int]]:
    """Generates a Cayley table for bitwise XOR. Assumes size is a power of 2."""
    return [[(i ^ j) for j in range(size)] for i in range(size)]


@registry.register("param_generators", "quaternion_group_q8_table")
def generate_quaternion_group_q8_table() -> List[List[int]]:
    """Generates the hardcoded Cayley table for the Quaternion group Q8."""
    # Mapping: 1, -1, i, -i, j, -j, k, -k  ->  0, 1, 2, 3, 4, 5, 6, 7
    return [
        [0, 1, 2, 3, 4, 5, 6, 7],  # 1
        [1, 0, 3, 2, 5, 4, 7, 6],  # -1
        [2, 3, 1, 0, 6, 7, 5, 4],  # i
        [3, 2, 0, 1, 7, 6, 4, 5],  # -i
        [4, 5, 7, 6, 1, 0, 2, 3],  # j
        [5, 4, 6, 7, 0, 1, 3, 2],  # -j
        [6, 7, 4, 5, 3, 2, 1, 0],  # k
        [7, 6, 5, 4, 2, 3, 0, 1]  # -k
    ]


# Don't forget to add this helper from the previous request if it's not there yet
@registry.register("param_generators", "group_inverse_table")
def generate_group_inverse_table(table: List[List[int]],
                                 identity_index: int = 0) -> List[int]:
    """
    Generates a 1D lookup table for inverses from a group's Cayley table.
    """
    size = len(table)
    if size == 0:
        return []

    inverses = [0] * size
    for i in range(size):
        try:
            inverse_of_i = table[i].index(identity_index)
            inverses[i] = inverse_of_i
        except ValueError:
            raise ValueError(
                f"Could not find an inverse for element {i}. The provided table does not represent a group."
            )

    return inverses
