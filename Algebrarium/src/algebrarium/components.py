import random
import re
import json
from typing import List, Any, Dict
from algebrarium.core import Set
from .registry import registry
from itertools import product, chain, combinations, permutations
from itertools import zip_longest


@registry.register("action_logics", "table_lookup_2d")
def table_lookup_2d_logic(elements: List[str],
                          table: List[List[int]]) -> callable:
    """
    (For finite systems) Creates logic for a binary operation defined by a 2D lookup table.
    """
    element_to_int = {name: i for i, name in enumerate(elements)}

    def logic(a_str: str, b_str: str) -> str:

        a_idx = element_to_int[a_str]
        b_idx = element_to_int[b_str]
        result_idx = table[a_idx][b_idx]
        return elements[result_idx]

    return logic


@registry.register("action_logics", "table_lookup_1d")
def table_lookup_1d_logic(elements: List[str], table: List[int]) -> callable:
    """(For finite systems) Creates logic for a unary operation defined by a 1D lookup table."""
    element_to_int = {name: i for i, name in enumerate(elements)}

    def logic(a_str: str) -> str:
        a_idx = element_to_int[a_str]
        result_idx = table[a_idx]
        return elements[result_idx]

    return logic


@registry.register("action_logics", "constant_value")
def constant_value_logic(value: str, **kwargs) -> callable:
    """Creates logic for a constant, returning a specific symbol."""
    return lambda: value


@registry.register("set_validators", "regex")
def regex_validator(pattern: str) -> callable:
    """Creates a validator based on a regular expression."""
    regex = re.compile(pattern)
    return lambda s: bool(regex.fullmatch(s))


@registry.register("set_samplers", "random_string")
def random_string_sampler(alphabet: str, min_len: int,
                          max_len: int) -> callable:
    """Creates a sampler that generates random strings."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):

            length = random.randint(int(min_len), int(max_len))

            samples.append("".join(
                str(random.choice(alphabet)) for _ in range(length)))
        return samples

    return sampler


@registry.register("action_logics", "charwise_modular_add_logic")
def charwise_modular_add_logic(alphabet: str) -> callable:
    n = len(alphabet)
    char_to_int = {c: i for i, c in enumerate(alphabet)}
    int_to_char = {i: c for i, c in enumerate(alphabet)}

    def logic(a_str: str, b_str: str) -> str:
        res = []
        for c1, c2 in zip(a_str, b_str):
            val = (char_to_int[c1] + char_to_int[c2]) % n
            res.append(int_to_char[val])
        return "".join(res)

    return logic


@registry.register("action_logics", "charwise_modular_inverse_logic")
def charwise_modular_inverse_logic(alphabet: str) -> callable:
    n = len(alphabet)
    char_to_int = {c: i for i, c in enumerate(alphabet)}
    int_to_char = {i: c for i, c in enumerate(alphabet)}

    def logic(a_str: str) -> str:
        res = []
        for char in a_str:
            val = (-char_to_int[char]) % n
            res.append(int_to_char[val])
        return "".join(res)

    return logic


@registry.register("action_logics", "bitwise_xor_logic")
def bitwise_xor_logic() -> callable:

    def logic(a_str: str, b_str: str) -> str:
        if len(a_str) != len(b_str):
            raise ValueError(
                "Inputs for bitwise_xor must have the same length.")
        a_int = int(a_str, 2)
        b_int = int(b_str, 2)
        res_int = a_int ^ b_int
        return format(res_int, f'0{len(a_str)}b')

    return logic


@registry.register("action_logics", "self_inverse_logic")
def self_inverse_logic() -> callable:
    """Logic for operations where every element is its own inverse."""
    return lambda a: a


@registry.register("param_generators", "string_permutations")
def generate_string_permutations(base_str: str) -> list[str]:
    """Generates all permutations of a given string."""
    return ["".join(p) for p in sorted(list(set(permutations(base_str))))]


@registry.register("set_validators", "finite_set_validator")
def finite_set_validator(elements: list[str]) -> callable:
    """Creates a validator that checks for membership in a fixed list."""

    allowed_elements = set(elements)
    return lambda s: s in allowed_elements


@registry.register("set_samplers", "finite_set_sampler")
def finite_set_sampler(elements: list[str]) -> callable:
    """Creates a sampler that picks randomly from a fixed list."""

    def sampler(k: int) -> list[str]:
        #print(elements)
        #print(random.choices(elements, k=k))
        return random.choices(elements, k=k)

    return sampler


def _get_permutation_map(base_str: str) -> dict:
    """Helper to get a map from a permuted string to its index permutation tuple."""
    sorted_chars = sorted(list(base_str))
    all_perms = list(permutations(range(len(base_str))))
    perm_map = {}
    for p_tuple in all_perms:
        permuted_str = "".join(sorted_chars[i] for i in p_tuple)
        perm_map[permuted_str] = p_tuple
    return perm_map


@registry.register("action_logics", "string_permutation_composition_logic")
def string_permutation_composition_logic(base_str: str) -> callable:
    """Creates logic for composing string permutations."""
    base_str = str(base_str)
    n = len(base_str)
    perm_map = _get_permutation_map(base_str)
    sorted_chars = sorted(list(base_str))

    def logic(p1_str: str, p2_str: str) -> str:
        p1_str, p2_str = str(p1_str), str(p2_str)
        p1 = perm_map[p1_str]
        p2 = perm_map[p2_str]
        # Composition p1 o p2 means apply p2 then p1
        p_comp = tuple(p1[p2[k]] for k in range(n))
        # Convert back to string
        return "".join(sorted_chars[i] for i in p_comp)

    return logic


@registry.register("action_logics", "string_permutation_inverse_logic")
def string_permutation_inverse_logic(base_str: str) -> callable:
    """Creates logic for finding the inverse of a string permutation."""
    base_str = str(base_str)
    n = len(base_str)
    perm_map = _get_permutation_map(base_str)
    sorted_chars = sorted(list(base_str))

    def logic(p_str: str) -> str:
        p = perm_map[p_str]
        inv_p = [0] * n
        for i in range(n):
            inv_p[p[i]] = i
        return "".join(sorted_chars[i] for i in inv_p)

    return logic


# --- [ Helpers & Logics for Galois Field - T27 ] ---
# These are highly non-trivial. They implement polynomial arithmetic over Z_p.
def _poly_divmod(num, den, p):
    num, den = list(num), list(den)
    if not den: raise ZeroDivisionError
    if len(num) < len(den): return [], num

    lead_den_inv = pow(den[-1], -1, p)

    q = [0] * (len(num) - len(den) + 1)

    for i in range(len(num) - len(den), -1, -1):
        q[i] = (num[i + len(den) - 1] * lead_den_inv) % p
        for j in range(len(den)):
            num[i + j] = (num[i + j] - q[i] * den[j]) % p

    # Trim leading zeros from remainder
    while len(num) > 1 and num[-1] == 0:
        num.pop()

    return q, num


def _poly_extended_gcd(a, b, p):
    # Returns (g, x, y) such that a*x + b*y = g
    x0, x1, y0, y1 = [1], [0], [0], [1]
    while b != [0]:
        q, r = _poly_divmod(a, b, p)
        a, b = b, r

        # Polynomial subtraction and multiplication helpers
        def _poly_sub(p1, p2):
            return [(v1 - v2) % p
                    for v1, v2 in zip_longest(p1, p2, fillvalue=0)]

        def _poly_mul_scalar(poly, s):
            return [(v * s) % p for v in poly]

        def _poly_mul(p1, p2):
            res = [0] * (len(p1) + len(p2) - 1)
            for i, v1 in enumerate(p1):
                for j, v2 in enumerate(p2):
                    res[i + j] = (res[i + j] + v1 * v2) % p
            return res

        x0, x1 = x1, _poly_sub(x0, _poly_mul(q, x1))
        y0, y1 = y1, _poly_sub(y0, _poly_mul(q, y1))

    return a, x0, y0


@registry.register("action_logics", "galois_field_add_logic")
def galois_field_add_logic(p: int) -> callable:
    # Addition in GF(p^k) is just polynomial addition over Z_p
    def logic(a_str: str, b_str: str) -> str:
        a_list = _poly_str_to_list(a_str, p)
        b_list = _poly_str_to_list(b_str, p)
        res_list = [(v1 + v2) % p
                    for v1, v2 in zip_longest(a_list, b_list, fillvalue=0)]
        return _poly_list_to_str(res_list)

    return logic


def _poly_str_to_list(s: str, n: int) -> List[int]:

    return [int(c) % n for c in s]


def _poly_list_to_str(l: List[int]) -> str:
    # Trim leading zeros, but leave a single '0' if that's all there is.
    while len(l) > 1 and l[-1] == 0:
        l.pop()
    return "".join(map(str, l))


@registry.register("action_logics", "galois_field_mul_logic")
def galois_field_mul_logic(p: int, mod_poly_str: str) -> callable:
    mod_poly_list = _poly_str_to_list(mod_poly_str, p)

    def logic(a_str: str, b_str: str) -> str:
        a_list = _poly_str_to_list(a_str, p)
        b_list = _poly_str_to_list(b_str, p)
        # Standard polynomial multiplication
        prod_list = [0] * (len(a_list) + len(b_list) - 1)
        for i, v1 in enumerate(a_list):
            for j, v2 in enumerate(b_list):
                prod_list[i + j] = (prod_list[i + j] + v1 * v2) % p
        # Take remainder modulo the irreducible polynomial
        _, remainder = _poly_divmod(prod_list, mod_poly_list, p)
        return _poly_list_to_str(remainder)

    return logic


@registry.register("action_logics", "galois_field_inv_logic")
def galois_field_inv_logic(p: int, mod_poly_str: str) -> callable:
    mod_poly_list = _poly_str_to_list(mod_poly_str, p)

    def logic(a_str: str) -> str:
        a_list = _poly_str_to_list(a_str, p)
        if a_list == [0]: raise ValueError("Cannot invert zero in a field.")

        # Using Extended Euclidean Algorithm for polynomials
        g, x, y = _poly_extended_gcd(a_list, mod_poly_list, p)

        # The gcd g must be a constant, normalize it
        if len(g) > 1 or g == [0]:
            raise ValueError(f"Polynomial {mod_poly_str} is not irreducible.")

        # Normalize the inverse x by g[0]
        inv_g0 = pow(g[0], -1, p)
        inv_list = [(c * inv_g0) % p for c in x]

        return _poly_list_to_str(inv_list)

    return logic


@registry.register("param_generators", "dihedral_group_table")
def generate_dihedral_group_table(n_gon: int) -> List[List[int]]:
    """Generates the Cayley table for the Dihedral group D_n (symmetries of an n-gon)."""
    if n_gon != 4:
        raise NotImplementedError(
            "This generator is currently hardcoded for D4 (a square).")

    # D4 has 8 elements. We map them to integers 0-7:
    # 0-3: Rotations R0, R90, R180, R270
    # 4-7: Reflections H, V, D1, D2
    # Elements are: ["R0", "R90", "R180", "R270", "H", "V", "D1", "D2"]

    # This is the hardcoded Cayley table for D4.
    # table[row][col] corresponds to applying the 'row' operation, then the 'col' operation.
    # For function composition f(g(x)), this corresponds to `row o col`.
    R0, R90, R180, R270 = 0, 1, 2, 3
    H, V, D1, D2 = 4, 5, 6, 7

    # 运算规则: table[row][col] 表示 T_row @ T_col
    # 组合顺序: 先应用 col 变换，再应用 row 变换 (row o col)
    # 这是经过数学验证的、完全正确的 D4 群凯莱表
    return [
        # Col: R0   R90  R180 R270   H    V    D1   D2
        #----------------------------------------------------
        [R0, R90, R180, R270, H, V, D1, D2],  # Row: R0
        [R90, R180, R270, R0, D1, D2, V, H],  # Row: R90
        [R180, R270, R0, R90, V, H, D2, D1],  # Row: R180
        [R270, R0, R90, R180, D2, D1, H, V],  # Row: R270
        [H, D2, V, D1, R0, R180, R270, R90],  # Row: H
        [V, D1, H, D2, R180, R0, R90, R270],  # Row: V
        [D1, H, D2, V, R90, R270, R0, R180],  # Row: D1
        [D2, V, D1, H, R270, R90, R180, R0]  # Row: D2
    ]


@registry.register("param_generators", "group_inverse_table")
def generate_group_inverse_table(table: List[List[int]],
                                 identity_index: int = 0) -> List[int]:
    """
    Generates a 1D lookup table for inverses from a group's Cayley table.
    It finds for each element 'i' the element 'j' such that table[i][j] = identity.
    """
    size = len(table)
    if size == 0:
        return []

    inverses = [0] * size
    for i in range(size):
        try:
            # The inverse of i is the element j such that i * j = identity.
            # We find the column index 'j' where the identity appears in row 'i'.
            inverse_of_i = table[i].index(identity_index)
            inverses[i] = inverse_of_i
        except ValueError:
            raise ValueError(
                f"Could not find an inverse for element {i}. "
                "The provided table does not represent a group or the identity is incorrect."
            )
    return inverses


@registry.register("set_samplers", "rgb_color_sampler")
def rgb_color_sampler() -> callable:

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            samples.append(f"{r},{g},{b}")
        return samples

    return sampler


@registry.register("action_logics", "rgb_xor_blend_logic")
def rgb_xor_blend_logic() -> callable:

    def logic(c1_str: str, c2_str: str) -> str:
        r1, g1, b1 = map(int, c1_str.split(','))
        r2, g2, b2 = map(int, c2_str.split(','))
        return f"{r1 ^ r2},{g1 ^ g2},{b1 ^ b2}"

    return logic


@registry.register("set_samplers", "permission_mask_sampler")
def permission_mask_sampler() -> callable:

    def sampler(k: int) -> List[str]:
        samples = []
        perms = ['r', 'w', 'x']
        for _ in range(k):
            p_str = ""
            for p in perms:
                p_str += p if random.choice([True, False]) else '-'
            samples.append(p_str)
        return samples

    return sampler


def _perm_to_int(p_str: str) -> int:
    val = 0
    if p_str[0] == 'r': val |= 4
    if p_str[1] == 'w': val |= 2
    if p_str[2] == 'x': val |= 1
    return val


def _int_to_perm(val: int) -> str:
    p_str = ""
    p_str += 'r' if (val & 4) else '-'
    p_str += 'w' if (val & 2) else '-'
    p_str += 'x' if (val & 1) else '-'
    return p_str


@registry.register("action_logics", "permission_mask_apply_logic")
def permission_mask_apply_logic() -> callable:

    def logic(p_current_str: str, p_mask_str: str) -> str:
        current_int = _perm_to_int(p_current_str)
        mask_int = _perm_to_int(p_mask_str)
        result_int = current_int ^ mask_int
        return _int_to_perm(result_int)

    return logic


@registry.register("action_logics", "fsm_transition_compose_logic")
def fsm_transition_compose_logic(num_states: int) -> callable:

    def logic(t1_str: str, t2_str: str) -> str:

        shift1 = int(t1_str.split('_')[1])
        shift2 = int(t2_str.split('_')[1])

        new_shift = (shift1 + shift2) % (num_states)
        return f"{t1_str.split('_')[0]}_{new_shift}"

    return logic


@registry.register("action_logics", "fsm_transition_inverse_logic")
def fsm_transition_inverse_logic(num_states: int) -> callable:

    def logic(t_str: str) -> str:

        shift = int(t_str.split('_')[1])
        inv_shift = (-shift + num_states) % num_states
        return f"{t_str.split('_')[0]}_{inv_shift}"

    return logic


# In components.py, find and replace ALL previous diff_patch_* functions with this block.
# --- Template 35: Diff Patch Group Components (Robust Version) ---


def _parse_patch(patch_str: str) -> Dict[int, tuple[str, str]]:
    """
    Parses a patch string like 'L5:a->b;L8:x->y' or an empty string into a dict.
    Returns a dictionary mapping line numbers to (old, new) tuples.
    """
    if not patch_str:
        return {}
    patch_map = {}
    parts = patch_str.split(';')
    for part in parts:
        match = re.fullmatch(r"L(\d+):([a-zA-Z]*)->([a-zA-Z]*)", part)
        if match:
            line, old, new = match.groups()
            patch_map[int(line)] = (old, new)
        else:
            # Raise an error for malformed parts to ensure strictness
            raise ValueError(
                f"Malformed patch part: '{part}' in '{patch_str}'")
    return patch_map


def _format_patch(patch_map: Dict[int, tuple[str, str]]) -> str:
    """
    Formats a patch dict back into a canonical string.
    The output is always sorted by line number.
    Returns an empty string for an empty patch map.
    """
    if not patch_map:
        return ""
    parts = []
    # Sort by line number to ensure a single, canonical representation
    for line in sorted(patch_map.keys()):
        old, new = patch_map[line]
        parts.append(f"L{line}:{old}->{new}")
    return ";".join(parts)


@registry.register("set_validators", "diff_patch_validator")
def diff_patch_validator() -> callable:
    """
    A robust validator for patch strings. It checks:
    1. Overall format correctness.
    2. Line numbers are unique.
    3. Line numbers are in ascending order in the string representation.
    """

    def logic(s: str) -> bool:
        if s == "":  # Empty string is the valid identity
            return True
        try:
            parts = s.split(';')
            last_line = -1
            seen_lines = set()
            for part in parts:
                match = re.fullmatch(r"L(\d+):([a-zA-Z]*)->([a-zA-Z]*)", part)
                if not match:
                    return False  # Malformed part

                line = int(match.group(1))
                if line in seen_lines:
                    return False  # Duplicate line number
                if line <= last_line:
                    return False  # Not sorted

                last_line = line
                seen_lines.add(line)
            return True
        except Exception:
            return False

    return logic


@registry.register("set_samplers", "diff_patch_sampler")
def diff_patch_sampler() -> callable:

    def _generate_single_patch_map() -> Dict[int, tuple[str, str]]:
        """Helper to generate one random patch part as a map entry."""
        line = random.randint(5, 5)
        old = "".join(random.choices("abcdefg", k=random.randint(0, 4)))
        new = "".join(random.choices("hijklmn", k=random.randint(0, 4)))
        return {line: (old, new)}

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            if random.random() < 0.1:  # 10% chance of empty patch
                samples.append("")
                continue

            num_patches = random.randint(1, 3)
            final_patch_map = {}
            for _ in range(num_patches):
                # Merging dicts ensures unique line numbers
                final_patch_map.update(_generate_single_patch_map())

            samples.append(_format_patch(final_patch_map))
        return samples

    return sampler


@registry.register("action_logics", "diff_patch_combine_logic")
def diff_patch_combine_logic() -> callable:
    """
    Combines two patches based on a strict functional composition model.
    P1 ++ P2 means applying P1, then applying P2 to the result.
    This logic correctly handles conflicts and ensures associativity.
    """

    def logic(p1_str: str, p2_str: str) -> str:
        p1_map = _parse_patch(p1_str)
        p2_map = _parse_patch(p2_str)

        # The final patch's "old" values come from p1's "old" values.
        # The final patch's "new" values come from p2's "new" values,
        # but transformed by p1.

        final_map = p1_map.copy()

        for line, (old2, new2) in p2_map.items():
            # The state P2 expects to see is `old2`.
            # What is the state after P1 is applied at this line?
            state_after_p1 = p1_map.get(
                line,
                (old2, old2))[1]  # If p1 doesn't change the line, new=old.

            if state_after_p1 != old2:
                # MERGE CONFLICT! The precondition for P2 is not met.
                # For a group, we MUST define a deterministic outcome.
                # A robust choice: the second patch's change is forced.
                # It transforms the state *after* p1 to its target state *new2*.
                # This means the final "new" value is p2's "new" value.

                # If p1 never touched this line, the original `old` is `old2`.
                # If p1 did touch this line, the original `old` is `p1_map[line][0]`.
                original_old = p1_map.get(line, (old2, old2))[0]
                final_map[line] = (original_old, new2)

            else:  # No conflict, this is a valid chained change.
                original_old = p1_map.get(line, (old2, old2))[0]
                final_map[line] = (original_old, new2)

            # Cleanup: if a change results in no change, remove it.
            if final_map[line][0] == final_map[line][1]:
                del final_map[line]

        return _format_patch(final_map)

    return logic


@registry.register("action_logics", "diff_patch_inverse_logic")
def diff_patch_inverse_logic() -> callable:

    def logic(p_str: str) -> str:
        p_map = _parse_patch(p_str)
        inv_map = {line: (new, old) for line, (old, new) in p_map.items()}
        return _format_patch(inv_map)

    return logic


# Helper function for parsing key-value strings
def _parse_kv_string(s: str, val_type=int) -> Dict[str, Any]:
    """Parses 'k1:v1,k2:v2' into a dictionary."""
    if not s:
        return {}
    parts = s.split(',')
    res_dict = {}
    for part in parts:
        try:
            key, val_str = part.split(':')
            res_dict[key.strip()] = val_type(val_str.strip())
        except ValueError:
            # Handle cases where parsing might fail
            continue
    return res_dict


# Helper function for formatting key-value dictionaries
def _format_kv_dict(d: Dict[str, Any]) -> str:
    """Formats a dictionary into a canonical 'k1:v1,k2:v2' string, sorted by key."""
    if not d:
        return ""
    return ",".join(f"{k}:{d[k]}" for k in sorted(d.keys()))


# Helper for set-like strings
def _parse_set_string(s: str, separator=',') -> set:
    if not s: return set()
    return set(item.strip() for item in s.split(separator))


def _format_set(s: set, separator=',') -> str:
    if not s: return ""
    return separator.join(sorted(list(s)))


@registry.register("param_generators", "generate_shifts")
def generate_shifts(n: int, prefix: str = "Shift") -> List[str]:
    """Generates a list of shift operations like ['Shift_0', 'Shift_1', ...]."""
    return [f"{prefix}_{i}" for i in range(n)]


@registry.register("action_logics", "charwise_boolean_xor_logic")
def charwise_boolean_xor_logic(true_char: str, false_char: str) -> callable:
    """Performs XOR on strings representing boolean arrays."""
    mapping = {true_char: True, false_char: False}
    rev_mapping = {True: true_char, False: false_char}

    def logic(a_str: str, b_str: str) -> str:
        res = []
        # Pad shorter string with false_char
        for c1, c2 in zip_longest(a_str, b_str, fillvalue=false_char):
            val = mapping[c1] ^ mapping[c2]
            res.append(rev_mapping[val])
        return "".join(res)

    return logic


def _get_full_inverse_map(inverses: Dict[str, str]) -> Dict[str, str]:
    """Expands a one-way map like {'U':'D'} into a two-way map {'U':'D', 'D':'U'}."""
    full_map = inverses.copy()
    for key, value in inverses.items():
        full_map[value] = key
    return full_map


@registry.register("action_logics", "free_group_compose_logic")
def free_group_compose_logic(inverses: Dict[str, str]) -> callable:
    """
    Creates logic for composing two sequences by concatenation followed by full reduction.
    Reduction is performed by repeatedly canceling out adjacent inverse pairs.
    """
    full_inverses = _get_full_inverse_map(inverses)

    def logic(s1: str, s2: str) -> str:
        combined_str = s1 + s2
        stack = []
        for char in combined_str:
            if stack and stack[-1] == full_inverses.get(char):
                stack.pop()
            else:
                stack.append(char)
        return "".join(stack)

    return logic


@registry.register("action_logics", "free_group_inverse_logic")
def free_group_inverse_logic(inverses: Dict[str, str]) -> callable:
    """
    Creates logic to find the inverse of a sequence.
    The inverse is the reversed sequence of inverted moves.
    """
    full_inverses = _get_full_inverse_map(inverses)

    def logic(s: str) -> str:
        if not s:
            return ""
        # Reverse the string, then map each character to its inverse
        return "".join(full_inverses[char] for char in reversed(s))

    return logic


@registry.register("set_samplers", "free_group_sampler")
def free_group_sampler(alphabet: str, inverses: Dict[str, str], min_len: int,
                       max_len: int) -> callable:
    """
    Generates random, valid, and canonically reduced move sequences.
    Ensures that no generated sequence contains adjacent inverse moves (e.g., 'UD' or 'LR').
    """
    full_inverses = _get_full_inverse_map(inverses)
    alphabet_set = set(alphabet)

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            length = random.randint(min_len, max_len)
            if length == 0:
                samples.append("")
                continue

            sequence = []
            for i in range(length):
                restricted_alphabet = alphabet_set
                # If there's a previous move, exclude its inverse from the choice
                if sequence:
                    last_move = sequence[-1]
                    inverse_of_last = full_inverses.get(last_move)
                    if inverse_of_last:
                        restricted_alphabet = alphabet_set - {inverse_of_last}

                # Choose the next move from the allowed (non-canceling) set
                next_move = random.choice(list(restricted_alphabet))
                sequence.append(next_move)

            samples.append("".join(sequence))
        return samples

    return sampler


@registry.register("action_logics", "set_symmetric_difference_logic")
def set_symmetric_difference_logic(separator: str = ',') -> callable:
    """Performs symmetric difference on sets represented by delimited strings."""

    def logic(s1: str, s2: str) -> str:
        set1 = _parse_set_string(s1, separator)
        set2 = _parse_set_string(s2, separator)
        result_set = set1.symmetric_difference(set2)
        return _format_set(result_set, separator)

    return logic


# --- Simplified "Conceptual" Logics ---


@registry.register("action_logics", "tetris_sequence_combine_logic")
def tetris_sequence_combine_logic() -> callable:
    """A highly simplified logic for Tetris piece cancellation."""
    cancellations = {"SZ": "", "ZS": "", "II": ""}  # Simple pairs that cancel

    def logic(s1: str, s2: str) -> str:
        res = s1 + s2
        # A simple, non-greedy replacement
        for pair, replacement in cancellations.items():
            res = res.replace(pair, replacement)
        return res

    return logic


@registry.register("action_logics", "layout_map_compose_logic")
def layout_map_compose_logic() -> callable:
    """Composition logic for a fixed set of keyboard layouts."""
    # Q->D, D->C => Q->C
    # Table defines row o col
    elements = ["ID", "Q_D", "D_Q", "Q_C", "C_Q", "D_C", "C_D"]
    elem_map = {name: i for i, name in enumerate(elements)}
    ID, Q_D, D_Q, Q_C, C_Q, D_C, C_D = range(7)

    table = [
        # ID   Q_D  D_Q  Q_C  C_Q  D_C  C_D
        [ID, Q_D, D_Q, Q_C, C_Q, D_C, C_D],  # ID
        [Q_D, D_Q, ID, D_C, C_D, Q_C, C_Q],  # Q_D o X
        [D_Q, ID, Q_D, C_Q, Q_C, C_D, D_C],  # D_Q o X
        [Q_C, C_D, D_C, ID, D_Q, Q_D, C_Q],  # Q_C o X
        [C_Q, Q_D, Q_C, D_Q, ID, D_C, C_D],  # C_Q o X
        [D_C, C_Q, Q_C, C_D, ID, Q_D, D_Q],  # D_C o X
        [C_D, D_Q, ID, Q_D, D_C, C_Q, Q_C],  # C_D o X
    ]

    def logic(s1: str, s2: str) -> str:
        idx1, idx2 = elem_map[s1], elem_map[s2]
        res_idx = table[idx1][idx2]
        return elements[res_idx]

    return logic


@registry.register("action_logics", "layout_map_inverse_logic")
def layout_map_inverse_logic() -> callable:
    """Inverse logic for keyboard layouts."""
    inv_map = {
        "ID": "ID",
        "Q_D": "D_Q",
        "D_Q": "Q_D",
        "Q_C": "C_Q",
        "C_Q": "Q_C",
        "D_C": "C_D",
        "C_D": "D_C"
    }

    def logic(s: str) -> str:
        return inv_map[s]

    return logic


# --- Components for Field & Vector Space Demonstrations ---


@registry.register("action_logics", "bitwise_and_logic")
def bitwise_and_logic() -> callable:
    """Performs bitwise AND on two binary strings of the same length."""

    def logic(a_str: str, b_str: str) -> str:
        if len(a_str) != len(b_str):
            raise ValueError(
                "Inputs for bitwise_and must have the same length.")
        a_int = int(a_str, 2)
        b_int = int(b_str, 2)
        res_int = a_int & b_int
        return format(res_int, f'0{len(a_str)}b')

    return logic


@registry.register("action_logics", "recipe_scale_logic")
def recipe_scale_logic() -> callable:
    """Scales a recipe (vector) by a scalar value."""

    def logic(scalar_str: str, vector_str: str) -> str:
        try:
            scalar = float(scalar_str)
        except ValueError:
            raise TypeError("Scalar must be a numeric value.")

        d = _parse_kv_string(vector_str, float)
        scaled_dict = {key: val * scalar for key, val in d.items()}
        return _format_kv_dict(scaled_dict)

    return logic


@registry.register("action_logics", "audio_mix_logic")
def audio_mix_logic() -> callable:
    """Conceptual logic for mixing audio tracks."""

    def logic(track1: str, track2: str) -> str:
        if "SILENT" in track1: return track2
        if "SILENT" in track2: return track1
        # In reality, this would be a complex numerical operation.
        # We simulate it by creating a new "mixed" track ID.
        t1_id = track1.split('_')[-1]
        t2_id = track2.split('_')[-1]
        # Sort to make addition commutative
        new_id = "_".join(sorted([t1_id, t2_id]))
        return f"MIX_{new_id}"

    return logic


@registry.register("action_logics", "audio_invert_phase_logic")
def audio_invert_phase_logic() -> callable:
    """Conceptual logic for inverting audio phase."""

    def logic(track: str) -> str:
        if "INV" in track:
            return track.replace("INV_", "")
        if "SILENT" in track:
            return track
        return f"INV_{track}"

    return logic


@registry.register("action_logics", "audio_gain_logic")
def audio_gain_logic() -> callable:
    """Conceptual logic for applying gain to an audio track."""

    def logic(scalar_str: str, vector_str: str) -> str:
        gain = float(scalar_str)
        if gain == 0: return "SILENT_TRACK"
        if gain == 1: return vector_str
        return f"GAIN({gain:.2f})_ON_{vector_str}"

    return logic


@registry.register("set_samplers", "key_value_sampler")
def key_value_sampler(keys: List[str],
                      min_val: float,
                      max_val: float,
                      min_keys: int = 1,
                      max_keys: int = -1,
                      val_is_int: bool = False) -> callable:
    """
    Creates a sampler for key-value pair strings like 'key1:val1,key2:val2'.
    """
    if max_keys == -1 or max_keys > len(keys):
        max_keys = len(keys)

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            num_keys_to_sample = random.randint(min_keys, max_keys)
            sampled_keys = random.sample(keys, num_keys_to_sample)

            sample_dict = {}
            for key in sampled_keys:
                if val_is_int:
                    value = random.randint(int(min_val), int(max_val))
                else:
                    value = random.uniform(min_val, max_val)
                # Avoid generating zero values which are identities
                while value == 0:
                    if val_is_int:
                        value = random.randint(int(min_val), int(max_val))
                    else:
                        value = random.uniform(min_val, max_val)

                # For cleaner output, format floats to a couple decimal places
                sample_dict[key] = f"{value:.2f}" if not val_is_int else str(
                    value)

            # Use the existing helper to create a canonical string representation
            samples.append(",".join(f"{k}:{sample_dict[k]}"
                                    for k in sorted(sample_dict.keys())))
        return samples

    return sampler


@registry.register("set_samplers", "subset_sampler")
def subset_sampler(elements: List[str],
                   min_subset_size: int = 0,
                   max_subset_size: int = -1,
                   separator: str = ',') -> callable:
    """
    Creates a sampler that generates random subsets of a given list of elements.
    The subsets are returned as formatted strings.
    """
    if max_subset_size == -1 or max_subset_size > len(elements):
        max_subset_size = len(elements)

    min_s = int(min_subset_size)
    max_s = int(max_subset_size)

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            # Choose a random size for the subset
            size = random.randint(min_s, max_s)
            # Choose 'size' unique elements from the list
            subset = random.sample(elements, size)
            # Format the subset into a sorted, delimited string
            samples.append(separator.join(sorted(subset)))
        return samples

    return sampler


@registry.register("set_samplers", "audio_track_sampler")
def audio_track_sampler() -> callable:
    """
    Creates a sampler that generates plausible-looking audio track IDs.
    """

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            track_type = random.choice(["VOCAL", "DRUMS", "BASS", "GUITAR"])
            take_num = random.randint(1, 5)
            # 10% chance of being silent or an inverse track for variety
            if random.random() < 0.05:
                samples.append("SILENT_TRACK")
            elif random.random() < 0.05:
                samples.append(f"INV_{track_type}_{take_num}")
            else:
                samples.append(f"{track_type}_{take_num}")
        return samples

    return sampler


@registry.register("set_validators", "sorted_pair_list_validator")
def sorted_pair_list_validator() -> callable:
    """
    Validates strings representing a sorted list of canonically ordered pairs.
    Example of a valid string: "ann-bob,carl-dave"
    - In "ann-bob", "ann" comes before "bob" alphabetically.
    - The pair "ann-bob" comes before "carl-dave" alphabetically.
    """

    def validator(s: str) -> bool:
        # The empty string is the identity element, so it's valid.
        if s == "":
            return True

        pairs = s.split(',')
        last_pair = ""  # To check for list-level sorting

        for pair_str in pairs:
            # Rule 1: Each part must be in the format "name1-name2"
            parts = pair_str.split('-')
            if len(parts) != 2:
                return False  # Malformed pair
            name1, name2 = parts

            # Rule 2: Names must not be empty and must be simple strings
            if not name1 or not name2 or not (name1.isalpha()
                                              and name2.isalpha()):
                return False

            # Rule 3: Internal sorting (canonical representation of a pair)
            if name1 >= name2:
                # e.g., "bob-ann" is invalid, must be "ann-bob"
                return False

            # Rule 4: External sorting (list of pairs must be sorted)
            if pair_str < last_pair:
                # e.g., "carl-dave,ann-bob" is invalid
                return False

            last_pair = pair_str

        return True  # All checks passed

    return validator


@registry.register("set_samplers", "friend_pair_sampler")
def friend_pair_sampler(names: List[str],
                        min_pairs: int = 0,
                        max_pairs: int = 3) -> callable:
    """
    Generates random, validly formatted friend-pair strings.
    """
    if len(names) < 2:
        raise ValueError(
            "Friend pair sampler requires at least two names in the pool.")

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            num_pairs = random.randint(min_pairs, max_pairs)
            if num_pairs == 0:
                samples.append("")
                continue

            generated_pairs = set()
            # We might need a few attempts to get unique pairs
            for _ in range(num_pairs * 2):  # Safety break
                if len(generated_pairs) >= num_pairs:
                    break

                # Sample two DISTINCT names
                p1, p2 = random.sample(names, 2)

                # Ensure canonical order (name1 < name2)
                if p1 > p2:
                    p1, p2 = p2, p1

                generated_pairs.add(f"{p1}-{p2}")

            # Final formatting: sort the pairs and join
            sorted_list = sorted(list(generated_pairs))
            samples.append(",".join(sorted_list))

        return samples

    return sampler


@registry.register("set_validators", "key_value_string_validator")
def key_value_string_validator(allowed_keys: List[str],
                               val_type: str = 'float') -> callable:
    """
    Validates key-value strings like 'key1:val1,key2:val2'.
    It checks if keys are from an allowed list and if values have the correct type.
    """
    allowed_keys_set = set(allowed_keys)

    def validator(s: str) -> bool:
        if s == "":
            return True  # Identity element is valid

        parts = s.split(',')
        if not parts:
            return False

        seen_keys = set()
        for part in parts:
            kv = part.split(':')
            if len(kv) != 2:
                return False  # Malformed pair
            key, val_str = kv

            # Check if the key is valid
            if key not in allowed_keys_set:
                return False

            # Check for duplicate keys
            if key in seen_keys:
                return False
            seen_keys.add(key)

            # Check if the value can be converted to the required type
            try:
                if val_type == 'float':
                    float(val_str)
                elif val_type == 'int':
                    int(val_str)
                else:
                    # Just checking it's not empty
                    if not val_str: return False
            except ValueError:
                return False  # Value has wrong type

        return True

    return validator


# In components.py, find and REPLACE this helper function.
def _parse_kv_string(s: str, val_type_func=float) -> Dict[str, Any]:
    """Parses 'k1:v1,k2:v2' into a dictionary, using a provided type function."""
    if not s:
        return {}
    parts = s.split(',')
    res_dict = {}
    for part in parts:
        try:
            key, val_str = part.split(':')
            res_dict[key.strip()] = val_type_func(val_str.strip())
        except (ValueError, IndexError):
            continue
    return res_dict


# In components.py, find and REPLACE this helper function.
def _format_kv_dict(d: Dict[str, Any]) -> str:
    """Formats a dictionary into a canonical 'k1:v1,k2:v2' string, sorted by key.
    Intelligently formats numbers to avoid '.0' for integers.
    """
    if not d:
        return ""
    # Filter out zero-value items before formatting
    d = {k: v for k, v in d.items() if v != 0}
    if not d:
        return ""

    parts = []
    for k in sorted(d.keys()):
        v = d[k]
        # Smart formatting: if a float is a whole number, format as int.
        if isinstance(v, float) and v == int(v):
            parts.append(f"{k}:{int(v)}")
        else:
            parts.append(f"{k}:{v}")
    return ",".join(parts)


# In components.py, find and REPLACE this logic component.
@registry.register("action_logics", "key_value_additive_merge_logic")
def key_value_additive_merge_logic(val_type: str = 'float') -> callable:
    """Merges two key-value strings, adding values for common keys.
    Respects whether values should be treated as 'int' or 'float'.
    """
    type_func = int if val_type == 'int' else float

    def logic(s1: str, s2: str) -> str:
        d1 = _parse_kv_string(s1, type_func)
        d2 = _parse_kv_string(s2, type_func)
        res_dict = d1.copy()
        for key, val in d2.items():
            res_dict[key] = res_dict.get(key, 0) + val
        return _format_kv_dict(res_dict)

    return logic


# In components.py, find and REPLACE this logic component.
@registry.register("action_logics", "key_value_additive_inverse_logic")
def key_value_additive_inverse_logic(val_type: str = 'float') -> callable:
    """Inverts a key-value string by negating all its values.
    Respects whether values should be treated as 'int' or 'float'.
    """
    type_func = int if val_type == 'int' else float

    def logic(s: str) -> str:
        d = _parse_kv_string(s, type_func)
        inv_dict = {key: -val for key, val in d.items()}
        return _format_kv_dict(inv_dict)

    return logic


# Add these helper functions and components to components.py


def _parse_timezone_offset(tz_str: str) -> int:
    """Parses 'UTC+8' into 8, 'UTC-5' into -5."""
    match = re.fullmatch(r"UTC([+-]\d+)", tz_str)
    if not match:
        raise ValueError(f"Invalid timezone format: {tz_str}")
    return int(match.group(1))


def _format_timezone_offset(offset: int) -> str:
    """Formats 8 into 'UTC+8', -5 into 'UTC-5'."""
    return f"UTC{offset:+}"


@registry.register("set_samplers", "timezone_sampler")
def timezone_sampler(min_offset: int = -12, max_offset: int = 14) -> callable:
    """
    Creates a sampler that generates random timezone strings like 'UTC+8'.
    """

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            offset = random.randint(min_offset, max_offset)
            samples.append(_format_timezone_offset(offset))
        return samples

    return sampler


# In components.py, find and REPLACE this logic component.
@registry.register("action_logics", "timezone_offset_compose_logic")
def timezone_offset_compose_logic() -> callable:
    """
    Creates logic for composing timezone offsets using modulo 24 arithmetic,
    mapping results to the range [-11, +12]. This forms a finite cyclic group Z_24.
    """

    def logic(tz1_str: str, tz2_str: str) -> str:
        offset1 = _parse_timezone_offset(tz1_str)
        offset2 = _parse_timezone_offset(tz2_str)

        # Perform addition modulo 24.
        # The formula (a + b + 11) % 24 - 11 maps the result to the range [-11, 12].
        # Let's test: 10 + 5 = 15. (15 + 11)%24 - 11 = 26%24 - 11 = 2 - 11 = -9. Correct.
        # Let's test: -10 + -5 = -15. (-15 + 11)%24 - 11 = -4%24 - 11 = 20 - 11 = 9. Correct.
        # Let's test the error case: 14 + 7 = 21 -> Let's assume inputs are in range, e.g. 12 + 10 = 22
        # 12 + 10 = 22. (22+11)%24-11 = 33%24-11 = 9-11 = -2. Correct.

        # A simpler way might be to map to [0, 23] first then adjust.
        # Let's map our range [-11, 12] to [0, 23]. offset_mapped = offset + 11
        # op1_mapped = offset1 + 11
        # op2_mapped = offset2 + 11
        # res_mapped = (op1_mapped + op2_mapped) % 24  -- This is wrong for a group isomorphism.
        # The operation must be a simple sum.

        new_offset = offset1 + offset2

        # Map to a positive equivalent range [0, 23]
        # Then map back to a balanced range [-11, 12] for display
        final_offset = (new_offset + 11) % 24 - 11

        return _format_timezone_offset(final_offset)

    return logic


# In components.py, find and REPLACE this logic component.
@registry.register("action_logics", "timezone_offset_inverse_logic")
def timezone_offset_inverse_logic() -> callable:
    """
    Creates logic for finding the inverse of a timezone offset in Z_24.
    The inverse of 'a' is '-a'.
    """

    def logic(tz_str: str) -> str:
        offset = _parse_timezone_offset(tz_str)
        # The inverse of 'a' in Z_n is 'n-a', or simply '-a' if we allow negative results.
        inverse_offset = -offset

        # We must ensure the inverse is also in the canonical range [-11, 12]
        final_inverse = (inverse_offset + 11) % 24 - 11

        return _format_timezone_offset(final_inverse)

    return logic


# Add these helper functions and components to components.py


def _parse_transaction(trans_str: str) -> float:
    """
    Parses a transaction string like 'DEPOSIT_100' or 'WITHDRAW_50' into a numeric value.
    DEPOSIT/INCOME are positive, WITHDRAW/PAY are negative.
    'NO_OP_0' is 0.
    Returns the net numeric change.
    """
    if trans_str.startswith("NO_OP"):
        return 0.0

    parts = trans_str.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid transaction format: {trans_str}")

    action, amount_str = parts
    try:
        amount = float(amount_str)
    except ValueError:
        raise ValueError(f"Invalid amount in transaction: {amount_str}")

    if action in ["DEPOSIT", "INCOME", "RECEIVE"]:
        return amount
    elif action in ["WITHDRAW", "PAY", "SPEND"]:
        return -amount
    else:
        raise ValueError(f"Unknown transaction action: {action}")


def _format_transaction(net_change: float) -> str:
    """
    Formats a numeric value into a canonical transaction string.
    Positive values become DEPOSIT, negative become WITHDRAW.
    """
    if abs(net_change) < 1e-9:  # Handle floating point inaccuracies for zero
        return "NO_OP_0"

    if net_change > 0:
        # Format to 2 decimal places for currency
        return f"DEPOSIT_{net_change:.2f}"
    else:
        # -net_change makes the amount positive for WITHDRAW
        return f"WITHDRAW_{-net_change:.2f}"


@registry.register("set_validators", "transaction_validator")
def transaction_validator() -> callable:
    """
    Validates transaction strings like 'DEPOSIT_100.50'.
    """

    def validator(s: str) -> bool:
        try:
            _parse_transaction(s)
            return True
        except ValueError:
            return False

    return validator


@registry.register("set_samplers", "transaction_sampler")
def transaction_sampler(min_amount: float = 1.0,
                        max_amount: float = 1000.0) -> callable:
    """
    Generates random transaction strings.
    """

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            amount = random.uniform(min_amount, max_amount)
            action = random.choice(["DEPOSIT", "WITHDRAW"])

            # 10% chance of generating the identity element
            if random.random() < 0.1:
                samples.append("NO_OP_0")
            else:
                samples.append(f"{action}_{amount:.2f}")
        return samples

    return sampler


@registry.register("action_logics", "transaction_compose_logic")
def transaction_compose_logic() -> callable:
    """
    Creates logic for composing two transactions by summing their net effect.
    """

    def logic(t1_str: str, t2_str: str) -> str:
        net1 = _parse_transaction(t1_str)
        net2 = _parse_transaction(t2_str)
        total_net = net1 + net2
        return _format_transaction(total_net)

    return logic


@registry.register("action_logics", "transaction_inverse_logic")
def transaction_inverse_logic() -> callable:
    """

    Creates logic for finding the inverse of a transaction.
    """

    def logic(t_str: str) -> str:
        net = _parse_transaction(t_str)
        return _format_transaction(-net)

    return logic


@registry.register("set_samplers", "command_sampler")
def command_sampler() -> callable:
    """
    Generates random, plausible-looking command sequences.
    """

    def sampler(k: int) -> List[str]:
        samples = []
        commands = [
            "INSERT", "DELETE", "MOVE_CURSOR", "SELECT", "COPY", "PASTE"
        ]
        args = ["'a'", "'word'", "5", "-3", "L10", "ALL"]

        for _ in range(k):
            num_cmds = random.randint(0, 4)
            if num_cmds == 0:
                samples.append("")
                continue

            cmd_list = []
            for _ in range(num_cmds):
                cmd = random.choice(commands)
                arg = random.choice(args)
                cmd_list.append(f"{cmd}({arg})")
            samples.append(";".join(cmd_list))
        return samples


@registry.register("action_logics", "string_concat_logic")
def string_concat_logic(separator: str = "") -> callable:
    """
    A simple logic that concatenates two strings with an optional separator.
    This version does NOT perform simplification, suitable for a "free semigroup".
    """

    def logic(s1: str, s2: str) -> str:
        if not s1: return s2
        if not s2: return s1
        return f"{s1}{separator}{s2}"

    return logic


@registry.register("action_logics", "command_list_inverse_logic")
def command_list_inverse_logic(inverse_map: Dict[str, str]) -> callable:
    """
    Finds the inverse of a command sequence.
    e.g., inv("CMD1(args1);CMD2(args2)") -> "INV_CMD2(args2);INV_CMD1(args1)"
    Argument inversion (e.g., MOVE(5) -> MOVE(-5)) must be handled by the inverse command name itself.
    This simplified version just adds a prefix "INV_".
    """

    def logic(s: str) -> str:
        if not s:
            return ""

        commands = s.split(';')
        inverse_commands = []

        for cmd_str in reversed(commands):
            match = re.match(r"([A-Z_]+)\((.*)\)", cmd_str)
            if match:
                cmd_name, args = match.groups()
                # Find the inverse command name from the map, or create a generic one
                inv_cmd_name = inverse_map.get(cmd_name, f"INV_{cmd_name}")
                inverse_commands.append(f"{inv_cmd_name}({args})")
            else:
                # Handle malformed or unknown commands gracefully
                inverse_commands.append(f"INV({cmd_str})")

        return ";".join(inverse_commands)

    return logic


def _parse_query_string(q_str: str) -> Dict[str, str]:
    """Parses 'k1=v1&k2=v2' into an ordered dictionary."""
    if not q_str:
        return {}
    params = {}
    pairs = q_str.split('&')
    for pair in pairs:
        if '=' in pair:
            key, val = pair.split('=', 1)
            params[key] = val
    return params


def _format_query_dict(q_dict: Dict[str, str]) -> str:
    """Formats a dictionary into a canonical 'k1=v1&k2=v2' string, sorted by key."""
    if not q_dict:
        return ""
    return "&".join(f"{k}={q_dict[k]}" for k in sorted(q_dict.keys()))


@registry.register("set_validators", "url_query_validator")
def url_query_validator() -> callable:
    """
    Validates that a string is a well-formed, canonically sorted URL query string.
    """

    def validator(s: str) -> bool:
        if s == "":
            return True
        try:
            # The core test is whether the string remains the same after parsing and re-formatting.
            # This elegantly checks for both valid format and canonical sorting.
            parsed = _parse_query_string(s)
            formatted = _format_query_dict(parsed)
            return s == formatted
        except Exception:
            return False

    return validator


@registry.register("set_samplers", "url_query_sampler")
def url_query_sampler(keys: List[str],
                      values: List[str],
                      min_params: int = 0,
                      max_params: int = 3) -> callable:
    """
    Generates random, canonically sorted URL query strings.
    """

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            num_params = random.randint(min_params, max_params)
            if num_params == 0:
                samples.append("")
                continue

            # Ensure we sample unique keys for each query string
            sampled_keys = random.sample(keys, min(num_params, len(keys)))

            query_dict = {}
            for key in sampled_keys:
                value = random.choice(values)
                query_dict[key] = value

            # Use the helper to ensure the output is always in canonical form
            samples.append(_format_query_dict(query_dict))
        return samples

    return sampler


@registry.register("action_logics", "query_xor_merge_logic")
def query_xor_merge_logic() -> callable:
    """
    Merges two query strings using symmetric difference (XOR) logic.
    If a key-value pair exists in both, it's removed. Otherwise, it's included.
    """

    def logic(q1_str: str, q2_str: str) -> str:
        # We can model this perfectly by treating the "key=value" strings as set elements.
        set1 = set(q1_str.split('&')) if q1_str else set()
        set2 = set(q2_str.split('&')) if q2_str else set()

        result_set = set1.symmetric_difference(set2)

        # We must re-parse and re-format to ensure the final output is canonically sorted by key.
        # For example, if result_set is {'c=3', 'a=1'}, it needs to be formatted as 'a=1&c=3'.
        temp_str = "&".join(sorted(list(result_set)))
        final_dict = _parse_query_string(temp_str)

        return _format_query_dict(final_dict)

    return logic


@registry.register("action_logics", "token_free_group_compose_logic")
def token_free_group_compose_logic(inverse_map: Dict[str, str],
                                   separator: str = ';') -> callable:
    """
    A more powerful free group composer that works on string tokens separated by a delimiter.
    """
    full_inverses = inverse_map.copy()
    for key, val in inverse_map.items():
        full_inverses[val] = key

    def logic(s1: str, s2: str) -> str:
        if not s1: return s2
        if not s2: return s1

        p1_parts = s1.split(separator) if s1 else []
        p2_parts = s2.split(separator) if s2 else []
        combined = p1_parts + p2_parts

        i = 0
        while i < len(combined) - 1:
            current_token = combined[i]
            next_token = combined[i + 1]
            if full_inverses.get(current_token) == next_token:
                combined.pop(i)
                combined.pop(i)
                if i > 0:
                    i -= 1
            else:
                i += 1
        return separator.join(combined)

    return logic


@registry.register("action_logics", "token_free_group_inverse_logic")
def token_free_group_inverse_logic(inverse_map: Dict[str, str],
                                   separator: str = ';') -> callable:
    """
    Finds the inverse of a token sequence.
    """
    full_inverses = inverse_map.copy()
    for key, val in inverse_map.items():
        full_inverses[val] = key

    def logic(s: str) -> str:
        if not s:
            return ""
        parts = s.split(separator)
        inv_parts = [
            full_inverses.get(p, f"INV({p})") for p in reversed(parts)
        ]
        return separator.join(inv_parts)

    return logic


@registry.register("set_samplers", "token_sequence_sampler")
def token_sequence_sampler(tokens: List[str],
                           inverse_map: Dict[str, str],
                           min_len: int = 0,
                           max_len: int = 4,
                           separator: str = ';') -> callable:
    """
    Generates random sequences of tokens.
    """

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            length = random.randint(min_len, max_len)
            if length == 0:
                samples.append("")
                continue

            sequence = random.choices(tokens, k=length)
            combined = sequence
            i = 0

            full_inverses = inverse_map.copy()
            while i < len(combined) - 1:
                current_token = combined[i]
                next_token = combined[i + 1]
                if full_inverses.get(current_token) == next_token:
                    combined.pop(i)
                    combined.pop(i)
                    if i > 0:
                        i -= 1
                else:
                    i += 1
            samples.append(separator.join(combined))

        return samples

    return sampler


# Add these components to components.py


def _str_to_perm_map(perm_str: str, alphabet: str) -> Dict[str, str]:
    """Converts a permutation string like 'bac' into a map {'a':'b', 'b':'a', 'c':'c'}."""
    return {
        original_char: perm_str[i]
        for i, original_char in enumerate(alphabet)
    }


def _perm_map_to_str(perm_map: Dict[str, str], alphabet: str) -> str:
    """Converts a permutation map back to its string representation."""
    return "".join(perm_map[c] for c in alphabet)


@registry.register("set_validators", "permutation_validator")
def permutation_validator(alphabet: str) -> callable:
    """
    Validates that a string is a valid permutation of a given alphabet.
    (i.e., it has the same length and contains exactly the same characters).
    """
    sorted_alphabet_str = "".join(sorted(alphabet))
    alphabet_len = len(alphabet)

    def validator(s: str) -> bool:
        if len(s) != alphabet_len:
            return False
        return "".join(sorted(s)) == sorted_alphabet_str

    return validator


@registry.register("set_samplers", "permutation_sampler")
def permutation_sampler(alphabet: str) -> callable:
    """
    Generates random permutations of a given alphabet string.
    """
    alphabet_list = list(alphabet)

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            random.shuffle(alphabet_list)
            samples.append("".join(alphabet_list))
        return samples

    return sampler


@registry.register("action_logics", "permutation_compose_logic")
def permutation_compose_logic(alphabet: str) -> callable:
    """
    Creates logic for composing two permutations, P1 o P2.
    This means applying P2 first, then P1.
    """

    def logic(p1_str: str, p2_str: str) -> str:
        # Convert permutations to explicit maps
        p1_map = _str_to_perm_map(p1_str, alphabet)
        p2_map = _str_to_perm_map(p2_str, alphabet)

        # Compose the maps: final_map[char] = p1_map[p2_map[char]]
        final_map = {}
        for char in alphabet:
            intermediate_char = p2_map[char]
            final_char = p1_map[intermediate_char]
            final_map[char] = final_char

        return _perm_map_to_str(final_map, alphabet)

    return logic


@registry.register("action_logics", "permutation_inverse_logic")
def permutation_inverse_logic(alphabet: str) -> callable:
    """
    Creates logic for finding the inverse of a permutation.
    """

    def logic(p_str: str) -> str:
        p_map = _str_to_perm_map(p_str, alphabet)

        # Invert the map: if p_map[a] = b, then inv_map[b] = a
        inv_map = {v: k for k, v in p_map.items()}

        return _perm_map_to_str(inv_map, alphabet)

    return logic


@registry.register("param_generators", "map_generator")
def map_generator(keys_template: str, values_template: str,
                  items: List[Any]) -> Dict[str, str]:
    """
    Dynamically generates a dictionary based on templates and a list of items.
    Example:
      keys_template: "PUSH({item})"
      values_template: "POP({item})"
      items: ["a", "b"]
    Returns:
      {"PUSH(a)": "POP(a)", "PUSH(b)": "POP(b)"}
    """
    gen_map = {}
    for item in items:
        key = keys_template.format(item=item)
        value = values_template.format(item=item)
        gen_map[key] = value
    return gen_map


@registry.register("set_validators", "stack_op_validator")
def stack_op_validator() -> callable:
    """
    Validates a semicolon-separated sequence of stack operations.
    This version is more robust and allows a wider range of characters inside the parentheses.
    """

    def validator(s: str) -> bool:
        if s == "":  # Identity element
            return True

        parts = s.split(';')
        if not parts:
            return False

        # Regex to match one valid operation like PUSH(anything_but_parentheses)
        # It allows spaces, quotes, brackets, etc., inside the args.
        # [^)]* matches zero or more characters that are NOT a closing parenthesis.
        op_pattern = re.compile(r"^(PUSH|POP)\([^)]*\)$")

        for part in parts:
            if not op_pattern.fullmatch(part):
                return False  # The structure PUSH(...) or POP(...) is wrong

        return True

    return validator


@registry.register("set_samplers", "stack_op_sampler")
def stack_op_sampler(alphabet: List[str], max_len: int = 5) -> callable:
    """
    Generates random sequences of stack operations.
    """

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            length = random.randint(0, max_len)
            if length == 0:
                samples.append("")
                continue

            sequence = []
            for _ in range(length):
                op_type = random.choice(["PUSH", "POP"])
                char = random.choice(alphabet)
                sequence.append(f"{op_type}({char})")

            samples.append(";".join(sequence))
        return samples

    return sampler


@registry.register("set_validators", "semicolon_list_validator")
def semicolon_list_validator(allowed_tokens: List[str]) -> callable:
    """
    Validates a semicolon-separated list of tokens, ensuring each token
    is in the allowed list.
    """
    allowed_set = set(allowed_tokens)

    def validator(s: str) -> bool:
        if s == "":  # Identity
            return True
        parts = s.split(';')
        # Check if all parts are in the allowed set
        return all(part in allowed_set for part in parts)

    return validator


# Add this component to components.py


@registry.register("action_logics", "rgb_difference_blend_logic")
def rgb_difference_blend_logic() -> callable:
    """
    Creates logic for the "Difference" blend mode in graphics software.
    The formula for each channel is: Result = |Blend - Base|.
    This forms a group where every element is its own inverse.
    """

    def logic(c1_str: str, c2_str: str) -> str:
        try:
            r1, g1, b1 = map(int, c1_str.split(','))
            r2, g2, b2 = map(int, c2_str.split(','))
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid RGB color format for '{c1_str}' or '{c2_str}'.")

        # Calculate the absolute difference for each channel
        res_r = abs(r1 - r2)
        res_g = abs(g1 - g2)
        res_b = abs(b1 - b2)

        return f"{res_r},{res_g},{res_b}"

    return logic


# Add these components to components.py


# --- Helper class for Video Editing Simulation ---
class Timeline:

    def __init__(self, start=0.0, end=100.0, speed=1.0, reversed=False):
        self.start = float(start)
        self.end = float(end)
        self.speed = float(speed)
        self.reversed = bool(reversed)

    @property
    def duration(self):
        return (self.end - self.start) / self.speed

    def __repr__(self):
        return f"Timeline(start={self.start:.2f}, end={self.end:.2f}, speed={self.speed:.2f}, reversed={self.reversed})"


def _parse_op(op_str: str) -> tuple[str, list[float]]:
    """
    Parses an operation string. Handles both forms: 'CMD(args)' and 'CMD'.
    """
    # Try to match 'CMD(args)' format first
    match_with_args = re.fullmatch(r"([A-Z_]+)\(([^)]*)\)", op_str)
    if match_with_args:
        name, args_str = match_with_args.groups()
        args = [float(a.strip())
                for a in args_str.split(',')] if args_str else []
        return name, args

    # If that fails, try to match 'CMD' format
    match_no_args = re.fullmatch(r"([A-Z_]+)", op_str)
    if match_no_args:
        name = match_no_args.group(1)
        return name, []  # Return empty list for args

    # If both fail, the format is invalid
    raise ValueError(f"Invalid op format: {op_str}")


registry.register("set_validators", "video_op_validator")


def video_op_validator() -> callable:
    # UPDATED: Now accepts UNTRIM as a valid operation.
    op_pattern = re.compile(
        r"^(TRIM\(\d+(\.\d*)?,\d+(\.\d*)?\)|UNTRIM\(\d+(\.\d*)?,\d+(\.\d*)?\)|SPEED\(\d+(\.\d*)?\)|REVERSE)$"
    )

    def validator(s: str) -> bool:
        if s == "": return True
        ops = s.split(';')
        return all(op_pattern.fullmatch(op) for op in ops)

    return validator


# In components.py, find and REPLACE video_op_inverse_logic.
@registry.register("action_logics", "video_op_inverse_logic")
def video_op_inverse_logic() -> callable:
    """
    Finds the inverse of a video operation sequence.
    inv(A;B) = inv(B);inv(A)
    The inverse of TRIM is a conceptual UNTRIM operation.
    """

    def logic(s: str) -> str:
        if not s: return ""
        ops = s.split(';')
        inv_ops = []
        for op_str in reversed(ops):
            op_name, args = _parse_op(op_str)

            if op_name == "SPEED":
                inv_op = f"SPEED({1.0/args[0]:.2f})" if args and args[
                    0] != 0 else "SPEED(inf)"
            elif op_name == "REVERSE":
                inv_op = "REVERSE"
            elif op_name == "TRIM":
                # The inverse is a new type of operation, UNTRIM.
                inv_op = f"UNTRIM({args[0]},{args[1]})"
            elif op_name == "UNTRIM":
                # The inverse of UNTRIM is TRIM.
                inv_op = f"TRIM({args[0]},{args[1]})"
            else:
                inv_op = f"INV_{op_name}"
                if args:
                    inv_op += f"({','.join(map(str,args))})"

            inv_ops.append(inv_op)
        return ";".join(inv_ops)

    return logic


def _apply_op_to_timeline(timeline: Timeline, op_name: str,
                          args: list[float]) -> Timeline:
    if op_name == "TRIM":
        # Trim operates on the *original* timeline, not the sped-up one
        original_start = timeline.start + args[0] * timeline.speed
        original_end = timeline.start + args[1] * timeline.speed
        timeline.start, timeline.end = max(timeline.start,
                                           original_start), min(
                                               timeline.end, original_end)
    elif op_name == "SPEED":
        timeline.speed *= args[0]
    elif op_name == "REVERSE":
        timeline.reversed = not timeline.reversed
    return timeline


# --- Registry Components ---


@registry.register("set_validators", "video_op_validator")
def video_op_validator() -> callable:
    op_pattern = re.compile(
        r"^(TRIM\(\d+(\.\d*)?,\d+(\.\d*)?\)|SPEED\(\d+(\.\d*)?\)|REVERSE)$")

    def validator(s: str) -> bool:
        if s == "": return True
        ops = s.split(';')
        return all(op_pattern.fullmatch(op) for op in ops)

    return validator


@registry.register("set_samplers", "video_op_sampler")
def video_op_sampler(max_len: int = 3) -> callable:

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            length = random.randint(0, max_len)
            if length == 0:
                samples.append("")
                continue

            op_list = []
            for _ in range(length):
                op_type = random.choice(["TRIM", "SPEED", "REVERSE"])
                if op_type == "TRIM":
                    t1 = random.uniform(0, 50)
                    t2 = random.uniform(t1 + 10, 100)
                    op_list.append(f"TRIM({t1:.1f},{t2:.1f})")
                elif op_type == "SPEED":
                    speed = random.choice([0.5, 2.0, 4.0])
                    op_list.append(f"SPEED({speed})")
                else:  # REVERSE
                    op_list.append("REVERSE")
            samples.append(";".join(op_list))
        return samples

    return sampler


@registry.register("action_logics", "video_op_compose_logic")
def video_op_compose_logic() -> callable:
    """
    Connects two video operation sequences.
    This is a simplified model, it does not perform complex algebraic simplification.
    It's a free monoid, not a group, but demonstrates non-commutativity.
    """

    def logic(s1: str, s2: str) -> str:
        if not s1: return s2
        if not s2: return s1
        return f"{s1};{s2}"

    return logic


# Add these components to components.py


def _parse_firewall_ops(s: str) -> list[tuple[str, list[str]]]:
    """Parses 'ADD(0,ruleA);DEL(1)' into [('ADD', ['0', 'ruleA']), ('DEL', ['1'])]"""
    if not s:
        return []
    ops = []
    op_parts = s.split(';')
    for part in op_parts:
        match = re.fullmatch(r"([A-Z]+)\(([^)]*)\)", part)
        if match:
            name, args_str = match.groups()
            args = [a.strip() for a in args_str.split(',')]
            ops.append((name, args))
        else:
            raise ValueError(f"Invalid firewall op format: {part}")
    return ops


def _format_firewall_ops(ops: list[tuple[str, list[str]]]) -> str:
    """Formats a list of parsed ops back into a string."""
    return ";".join(f"{name}({','.join(args)})" for name, args in ops)


# In components.py, find and REPLACE this helper function.
def _parse_firewall_ops(s: str) -> list[tuple[str, list[str]]]:
    """Parses 'ADD(0,ruleA);INV_DEL(1)' into [('ADD', ['0', 'ruleA']), ('INV_DEL', ['1'])]"""
    if not s:
        return []
    ops = []
    op_parts = s.split(';')
    for part in op_parts:
        # UPDATED REGEX: Now optionally matches "INV_" prefix.
        # It captures the prefix (or empty string) and the main command name separately.
        match = re.fullmatch(r"((?:INV_)?[A-Z]+)\(([^)]*)\)", part)
        if match:
            name, args_str = match.groups()
            args = [a.strip() for a in args_str.split(',')]
            ops.append((name, args))
        else:
            # This will now correctly raise an error for truly malformed strings,
            # but will succeed for strings like 'INV_ADD(1,ruleW)'.
            raise ValueError(f"Invalid firewall op format: {part}")
    return ops


# In components.py, find and REPLACE this logic component.
# The previous version had a bug in its last line.
@registry.register("action_logics", "firewall_op_inverse_logic")
def firewall_op_inverse_logic() -> callable:
    """
    Calculates the conceptual inverse of a sequence of firewall operations.
    inv(A;B) = inv(B);inv(A).
    """

    def logic(s: str) -> str:
        if not s: return ""
        ops = _parse_firewall_ops(s)
        inv_ops_parsed = []

        for op_name, args in reversed(ops):
            # This part of the logic remains the same.
            if op_name == "MOVE":
                inv_op_name = "MOVE"
                inv_args = [args[1], args[0]]  # Swap from and to
            elif op_name.startswith("INV_"):
                inv_op_name = op_name[4:]  # Remove "INV_"
                inv_args = args
            else:  # For ADD, DEL, and any other command
                inv_op_name = f"INV_{op_name}"
                inv_args = args

            inv_ops_parsed.append((inv_op_name, inv_args))

        # The FIX is here: We now have a list of parsed tuples,
        # which _format_firewall_ops can correctly handle.
        return _format_firewall_ops(inv_ops_parsed)

    return logic


@registry.register("set_samplers", "firewall_op_sampler")
def firewall_op_sampler(max_rules: int = 5, max_len: int = 3) -> callable:
    """Generates random sequences of firewall operations."""
    rule_names = [f"rule{chr(ord('A')+i)}" for i in range(26)]

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            length = random.randint(0, max_len)
            if length == 0:
                samples.append("")
                continue

            op_list = []
            # We simulate a list to know valid indices
            num_current_rules = random.randint(1, max_rules)
            for _ in range(length):
                op_type = random.choice(["ADD", "DEL", "MOVE"])
                if op_type == "ADD" and num_current_rules < max_rules * 2:
                    idx = random.randint(0, num_current_rules)
                    rule = random.choice(rule_names)
                    op_list.append(f"ADD({idx},{rule})")
                    num_current_rules += 1
                elif op_type == "DEL" and num_current_rules > 0:
                    idx = random.randint(0, num_current_rules - 1)
                    op_list.append(f"DEL({idx})")
                    num_current_rules -= 1
                elif op_type == "MOVE" and num_current_rules > 1:
                    idx_from = random.randint(0, num_current_rules - 1)
                    idx_to = random.randint(0, num_current_rules - 1)
                    if idx_from != idx_to:
                        op_list.append(f"MOVE({idx_from},{idx_to})")

            if not op_list:  # In case only invalid ops were chosen
                samples.append("")
            else:
                samples.append(";".join(op_list))
        return samples

    return sampler


@registry.register("set_validators", "firewall_op_validator")
def firewall_op_validator() -> callable:
    # UPDATED to include conceptual inverses
    op_pattern = re.compile(r"^(INV_)?(ADD|DEL|MOVE)\(.*\)$")

    def validator(s: str) -> bool:
        if s == "": return True
        ops = s.split(';')
        return all(op_pattern.fullmatch(op) for op in ops)

    return validator


# Add these helper functions and components to components.py

# A configuration dictionary for cron fields.
# Format: { 'char': (modulus, name) }
CRON_FIELDS_CONFIG = {
    'M': (60, "Minute"),
    'H': (24, "Hour"),
    'd': (31, "DayOfMonth"),  # 在此系统中简化
    'm': (12, "Month"),
    'w': (7, "DayOfWeek")
}

# 单位元（零偏移）的唯一规范字符串表示
CANONICAL_IDENTITY_STRING = "_".join(f"{char}+0"
                                     for char in CRON_FIELDS_CONFIG.keys())


def _parse_cron_offset(offset_str: str) -> Dict[str, int]:
    """
    解析 cron 偏移字符串 (例如 'M+5_H-2') 为字典。
    值被规范化到标准的 [0, modulus-1] 区间。
    例如, 'H-2' (mod 24) 会变为 {'H': 22}。
    """
    if not offset_str:
        return {}

    offset_dict = {}
    parts = offset_str.split('_')
    for part in parts:
        if not part:
            continue

        field_char = part[0]
        if field_char in CRON_FIELDS_CONFIG:
            try:
                value = int(part[1:])
                modulus = CRON_FIELDS_CONFIG[field_char][0]
                # 将值规范化到 [0, modulus-1] 的标准区间
                offset_dict[field_char] = value % modulus
            except (ValueError, IndexError):
                raise ValueError(f"cron 偏移部分的值无效: {part}")
        else:
            raise ValueError(f"cron 偏移中的字段未知: {field_char}")

    return offset_dict


def _format_cron_offset(offset_dict: Dict[str, int]) -> str:
    """
    将偏移字典格式化为其唯一的规范字符串表示。
    示例: {'M': 5, 'H': 22} -> 'M+5_H+22'
    - 字段根据 CRON_FIELDS_CONFIG 的顺序排序。
    - 值为 0 的字段将被省略，除非结果是单位元。
    - 空字典或仅包含 0 值的字典将被格式化为单位元的规范字符串。
    """
    parts = []
    # 为了生成规范表示，按配置中的顺序排序
    for char, _ in CRON_FIELDS_CONFIG.items():
        value = offset_dict.get(char, 0)
        # 为了简洁，我们只在字符串中包含非零的部分
        if value != 0:
            parts.append(f"{char}{value:+}")

    # 如果所有部分都为零，则返回单位元的规范字符串；否则，连接所有部分。
    if not parts:
        return CANONICAL_IDENTITY_STRING
    else:
        return "_".join(parts)


@registry.register("set_validators", "cron_offset_validator")
def cron_offset_validator() -> callable:
    """
    验证一个字符串是否是 cron 偏移的规范表示。
    一个字符串是有效的，当且仅当它在解析和重新格式化后与自身完全相同。
    """

    def validator(s: str) -> bool:
        try:
            # 空字符串不是一个有效的规范元素
            if not s:
                return False
            # 一个字符串是有效的，如果它符合其规范形式。
            # 解析会规范化值，格式化会创建规范字符串。
            return s == _format_cron_offset(_parse_cron_offset(s))
        except ValueError:
            return False

    return validator


@registry.register("set_samplers", "cron_offset_sampler")
def cron_offset_sampler() -> callable:
    """生成随机、有效且规范的 cron 偏移字符串。"""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            offset_dict = {}
            # 对每个可能的字段，随机决定是否包含它
            for char, (modulus, _) in CRON_FIELDS_CONFIG.items():
                if random.choice([True, False]):
                    # 在规范的 [0, modulus-1] 区间内生成一个随机偏移
                    offset = random.randint(0, modulus - 1)
                    # 只需存储非零的偏移量
                    if offset != 0:
                        offset_dict[char] = offset

            samples.append(_format_cron_offset(offset_dict))
        return samples

    return sampler


@registry.register("action_logics", "cron_offset_compose_logic")
def cron_offset_compose_logic() -> callable:
    """
    创建用于组合两个 cron 偏移的逻辑。
    组合是通过将各分量在对应模数下相加来完成的。
    结果以规范的字符串形式返回。
    """

    def logic(s1: str, s2: str) -> str:
        d1 = _parse_cron_offset(s1)
        d2 = _parse_cron_offset(s2)

        res_dict = {}
        # 使用规范的字段顺序进行处理
        for char, (modulus, _) in CRON_FIELDS_CONFIG.items():
            val1 = d1.get(char, 0)
            val2 = d2.get(char, 0)

            # 如果某个字段在两个输入中都为零，则无需计算
            if val1 == 0 and val2 == 0:
                continue

            # 执行模 N 加法。结果已经位于 [0, modulus-1] 区间。
            res_val = (val1 + val2) % modulus
            res_dict[char] = res_val

        return _format_cron_offset(res_dict)

    return logic


@registry.register("action_logics", "cron_offset_inverse_logic")
def cron_offset_inverse_logic() -> callable:
    """
    寻找一个 cron 偏移的加法逆元。
    结果以规范的字符串形式返回。
    """

    def logic(s: str) -> str:
        d = _parse_cron_offset(s)
        inv_dict = {}

        for char, val in d.items():
            # 0 的逆元是 0，无需处理
            if val == 0:
                continue

            modulus = CRON_FIELDS_CONFIG[char][0]
            # Z_n 中 'a' 的逆元是 'n - a'，等价于 -a mod n。
            # Python 的 % 运算符能正确处理负数。
            inv_val = (-val) % modulus
            inv_dict[char] = inv_val

        return _format_cron_offset(inv_dict)

    return logic


# Add these helper functions and components to components.py

# A configuration for Hearthstone-like buff fields.
# Format: { 'name': 'type' }
BUFF_FIELDS_CONFIG = {
    'ATK': 'additive',
    'HP': 'additive',
    'TAUNT': 'toggle',
    'DIVINE_SHIELD': 'toggle',
    'POISONOUS': 'toggle'
}


def _parse_buff_string(s: str) -> Dict[str, Any]:
    """Parses 'ATK+2;TAUNT_TOGGLE' into {'ATK': 2, 'TAUNT': True}."""
    if not s:
        return {}

    buff_dict = {}
    parts = s.split(';')
    for part in parts:
        if not part: continue

        if part.endswith("_TOGGLE"):
            field_name = part[:-7]
            if BUFF_FIELDS_CONFIG.get(field_name) == 'toggle':
                buff_dict[
                    field_name] = True  # True signifies a toggle occurred
            else:
                raise ValueError(
                    f"Field '{field_name}' is not a toggleable type.")
        else:
            match = re.fullmatch(r"([A-Z]+)([+-]\d+)", part)
            if match:
                field_name, value_str = match.groups()
                if BUFF_FIELDS_CONFIG.get(field_name) == 'additive':
                    buff_dict[field_name] = int(value_str)
                else:
                    raise ValueError(
                        f"Field '{field_name}' is not an additive type.")
            else:
                raise ValueError(f"Invalid buff format: {part}")
    return buff_dict


def _format_buff_dict(buff_dict: Dict[str, Any]) -> str:
    """Formats {'ATK': 2, 'TAUNT': True} into 'ATK+2;TAUNT_TOGGLE'."""
    if not buff_dict:
        return ""

    parts = []
    # Sort by field name for a canonical representation
    for field in sorted(buff_dict.keys()):
        value = buff_dict[field]
        field_type = BUFF_FIELDS_CONFIG.get(field)

        if field_type == 'additive' and value != 0:
            parts.append(f"{field}{value:+}")
        elif field_type == 'toggle' and value:  # Only show if a toggle happened
            parts.append(f"{field}_TOGGLE")

    return ";".join(parts)


@registry.register("set_validators", "hearthstone_buff_validator")
def hearthstone_buff_validator() -> callable:
    """Validates Hearthstone-style buff strings."""

    def validator(s: str) -> bool:
        try:
            # The most robust check: can it be parsed?
            _parse_buff_string(s)
            return True
        except ValueError:
            return False

    return validator


@registry.register("set_samplers", "hearthstone_buff_sampler")
def hearthstone_buff_sampler() -> callable:
    """Generates random Hearthstone-style buff strings."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            buff_dict = {}
            # Additive fields
            for field in [
                    f for f, t in BUFF_FIELDS_CONFIG.items() if t == 'additive'
            ]:
                if random.random() < 0.4:  # 40% chance to include this buff
                    value = random.randint(-3, 5)
                    if value != 0:
                        buff_dict[field] = value
            # Toggle fields
            for field in [
                    f for f, t in BUFF_FIELDS_CONFIG.items() if t == 'toggle'
            ]:
                if random.random() < 0.3:  # 30% chance to toggle
                    buff_dict[field] = True

            if not buff_dict:
                # 10% chance to sample identity, otherwise retry
                if random.random() < 0.1:
                    samples.append("")
                else:  # To avoid too many empty strings, generate at least one buff
                    buff_dict['ATK'] = random.randint(1, 3)
                    samples.append(_format_buff_dict(buff_dict))
            else:
                samples.append(_format_buff_dict(buff_dict))
        return samples

    return sampler


@registry.register("action_logics", "hearthstone_buff_compose_logic")
def hearthstone_buff_compose_logic() -> callable:
    """Composes two buffs. Additive fields are summed, toggle fields are XORed."""

    def logic(s1: str, s2: str) -> str:
        d1 = _parse_buff_string(s1)
        d2 = _parse_buff_string(s2)
        res_dict = {}
        all_keys = set(d1.keys()).union(set(d2.keys()))

        for field in all_keys:
            field_type = BUFF_FIELDS_CONFIG.get(field)
            val1 = d1.get(field, 0 if field_type == 'additive' else False)
            val2 = d2.get(field, 0 if field_type == 'additive' else False)

            if field_type == 'additive':
                res_dict[field] = val1 + val2
            elif field_type == 'toggle':
                res_dict[field] = val1 ^ val2

        return _format_buff_dict(res_dict)

    return logic


@registry.register("action_logics", "hearthstone_buff_inverse_logic")
def hearthstone_buff_inverse_logic() -> callable:
    """Finds the inverse of a buff. Additive fields are negated, toggles are their own inverse."""

    def logic(s: str) -> str:
        d = _parse_buff_string(s)
        inv_dict = {}
        for field, value in d.items():
            field_type = BUFF_FIELDS_CONFIG.get(field)
            if field_type == 'additive':
                inv_dict[field] = -value
            elif field_type == 'toggle':
                inv_dict[field] = value  # A toggle is its own inverse
        return _format_buff_dict(inv_dict)

    return logic


# Add these components to components.py

# --- Helpers for Rubik's Cube Logic ---
MOVES = ["R", "L", "U", "D", "F", "B"]
PRIME_SUFFIX = "#"  # CORRECTED: Use standard prime notation
DOUBLE_SUFFIX = "2"


def _parse_rubiks_moves(s: str) -> list[str]:
    """Parses 'R U R' U2' into ['R', 'U', 'R'', 'U2']"""
    if not s:
        return []
    # Use regex to find all valid move tokens
    return re.findall(rf"[RLUDFB][{DOUBLE_SUFFIX}{PRIME_SUFFIX}]?", s)


def _get_inverse_move(move: str) -> str:
    """Gets the inverse of a single move, e.g., R -> R', R' -> R, R2 -> R2."""
    if move.endswith(PRIME_SUFFIX):
        return move[0]
    elif move.endswith(DOUBLE_SUFFIX):
        return move  # Double moves are their own inverse
    else:
        return move + PRIME_SUFFIX


def _simplify_moves(moves: list[str]) -> list[str]:
    """
    Simplifies a list of Rubik's cube moves into a canonical form.
    This involves three phases that repeat until no more changes can be made:
    1. Consolidate adjacent same-face moves (e.g., R R -> R2, R R' -> '').
    2. Reorder adjacent commuting moves (opposite faces) into a fixed order
       (e.g., L R -> R L).
    """
    # Define canonical order: R before L, U before D, F before B.
    # A lower index means it should come first in a commuting pair.
    CANONICAL_ORDER_INDEX = {'R': 0, 'L': 1, 'U': 2, 'D': 3, 'F': 4, 'B': 5}
    OPPOSITE_FACE = {
        'R': 'L',
        'L': 'R',
        'U': 'D',
        'D': 'U',
        'F': 'B',
        'B': 'F'
    }

    while True:
        made_simplification = False
        original_len = len(moves)

        # Phase 1: Consolidate same-face moves using a stack-like approach.
        # This handles all same-face combinations, including cancellations (R R').
        if moves:
            consolidated = [moves[0]]
            for move in moves[1:]:
                if consolidated and consolidated[-1][0] == move[0]:
                    last_move = consolidated.pop()

                    # Turn values: normal=1, double=2, prime=3 (-1 mod 4)
                    def get_turn_val(m):
                        if m.endswith(PRIME_SUFFIX): return 3
                        if m.endswith(DOUBLE_SUFFIX): return 2
                        return 1

                    val1 = get_turn_val(last_move)
                    val2 = get_turn_val(move)
                    total_val = (val1 + val2) % 4

                    base = move[0]
                    if total_val == 1: consolidated.append(base)
                    elif total_val == 2:
                        consolidated.append(base + DOUBLE_SUFFIX)
                    elif total_val == 3:
                        consolidated.append(base + PRIME_SUFFIX)
                    # if total_val is 0, they cancel, so we don't append anything.
                else:
                    consolidated.append(move)

            if len(consolidated) != len(moves):
                moves = consolidated
                made_simplification = True
                continue  # Restart the loop to re-evaluate from the beginning

        # Phase 2: Enforce canonical order for commuting moves.
        i = 0
        while i < len(moves) - 1:
            curr_base = moves[i][0]
            next_base = moves[i + 1][0]

            # Check if they are a commuting pair (opposite faces)
            if OPPOSITE_FACE.get(curr_base) == next_base:
                # Check if they are in the wrong canonical order
                if CANONICAL_ORDER_INDEX[curr_base] > CANONICAL_ORDER_INDEX[
                        next_base]:
                    moves[i], moves[i + 1] = moves[i + 1], moves[i]
                    made_simplification = True
                    # A swap might create a new consolidation, so restart the whole process.
                    break
            i += 1

        if made_simplification:
            continue

        # If we went through a full loop with no changes, the sequence is canonical.
        break

    return moves


# --- Registry Components ---


@registry.register("set_validators", "rubiks_move_validator")
def rubiks_move_validator() -> callable:
    """Validates a string of Rubik's Cube moves."""

    def validator(s: str) -> bool:
        if s == "": return True
        try:
            # Check if the parsed version, when joined, matches the original string
            # without spaces. This ensures no invalid characters are present.
            parsed = _parse_rubiks_moves(s)
            return "".join(parsed) == s.replace(" ", "")
        except:
            return False

    return validator


@registry.register("set_samplers", "rubiks_move_sampler")
def rubiks_move_sampler(max_len: int = 8) -> callable:
    """
    Generates random sequences of Rubik's Cube moves in their canonical form.
    FIXED: Ensures all generated samples are simplified.
    """
    all_move_types = MOVES + [m + PRIME_SUFFIX for m in MOVES
                              ] + [m + DOUBLE_SUFFIX for m in MOVES]

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            length = random.randint(0, max_len)
            if length == 0:
                samples.append("")
                continue

            sequence = random.choices(all_move_types, k=length)

            # Simplify the randomly generated sequence to its canonical form.
            simplified_sequence = _simplify_moves(sequence)

            samples.append(" ".join(simplified_sequence))
        return samples

    return sampler


@registry.register("action_logics", "rubiks_move_compose_logic")
def rubiks_move_compose_logic() -> callable:
    """
    Composes two move sequences and simplifies them to their canonical form.
    FIXED: Uses the robust, standalone _simplify_moves function.
    """

    def logic(s1: str, s2: str) -> str:
        moves1 = _parse_rubiks_moves(s1)
        moves2 = _parse_rubiks_moves(s2)
        combined = moves1 + moves2

        simplified_moves = _simplify_moves(combined)

        return " ".join(simplified_moves)

    return logic


@registry.register("action_logics", "rubiks_move_inverse_logic")
def rubiks_move_inverse_logic() -> callable:
    """
    Finds the inverse of a Rubik's Cube sequence. (A B C)' = C' B' A'.
    Note: The result of the inverse is not simplified here, but will be
    simplified upon composition with another sequence.
    """

    def logic(s: str) -> str:
        moves = _parse_rubiks_moves(s)
        inv_moves = [_get_inverse_move(m) for m in reversed(moves)]
        inv_moves = _simplify_moves(inv_moves)
        return " ".join(inv_moves)

    return logic


# Add these helper functions and components to components.py


def _parse_rotor_state(state_str: str) -> list[int]:
    """Parses 'A,B,C' into a list of integers [0, 1, 2]."""
    parts = state_str.split(',')
    if len(parts) != 3:
        raise ValueError(f"Invalid rotor state format: {state_str}")

    # ord('A') is 65. So, ord(char) - 65 maps 'A'->0, 'B'->1, ...
    return [ord(char.strip().upper()) - 65 for char in parts]


def _format_rotor_state(state_list: list[int]) -> str:
    """Formats a list of integers [0, 1, 2] into 'A,B,C'."""
    # chr(num + 65) maps 0->'A', 1->'B', ...
    return ",".join([chr(num + 65) for num in state_list])


@registry.register("set_samplers", "enigma_rotor_sampler")
def enigma_rotor_sampler() -> callable:
    """Generates random 3-rotor Enigma states like 'X,Y,Z'."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            # Generate 3 random integers between 0 and 25
            state_list = [random.randint(0, 25) for _ in range(3)]
            samples.append(_format_rotor_state(state_list))
        return samples

    return sampler


@registry.register("action_logics", "rotor_state_add_logic")
def rotor_state_add_logic() -> callable:
    """
    Composes two rotor states by adding their positions modulo 26 for each rotor.
    """

    def logic(s1_str: str, s2_str: str) -> str:
        state1 = _parse_rotor_state(s1_str)
        state2 = _parse_rotor_state(s2_str)

        # Add corresponding elements modulo 26
        new_state = [(state1[i] + state2[i]) % 26 for i in range(3)]

        return _format_rotor_state(new_state)

    return logic


@registry.register("action_logics", "rotor_state_inverse_logic")
def rotor_state_inverse_logic() -> callable:
    """
    Finds the inverse of a rotor state.
    The inverse of 'a' in Z_26 is '26 - a' (or -a mod 26).
    """

    def logic(s_str: str) -> str:
        state = _parse_rotor_state(s_str)

        # The inverse of each component is (-component) mod 26
        inv_state = [(-pos) % 26 for pos in state]

        return _format_rotor_state(inv_state)

    return logic


# Add these helper functions and components to components.py
import math


def _gcd(a, b):
    """Helper to compute the greatest common divisor."""
    return math.gcd(a, b)


def _modInverse(a, m):
    """
    Helper to compute modular multiplicative inverse of a under modulo m.
    Uses the Extended Euclidean Algorithm.
    Returns x such that (a * x) % m == 1.
    """
    # This is equivalent to pow(a, -1, m) in Python 3.8+
    # We write it out for clarity and compatibility.
    if _gcd(a, m) != 1:
        # The modular inverse does not exist
        return None

    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 = x1 + m0
    return x1


def _parse_affine_pair(s: str) -> tuple[int, int]:
    """Parses '(a,b)' into a tuple (a, b)."""
    match = re.fullmatch(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", s)
    if not match:
        raise ValueError(f"Invalid affine pair format: {s}")
    return int(match.group(1)), int(match.group(2))


@registry.register("set_validators", "affine_param_validator")
def affine_param_validator(m: int) -> callable:
    """Validates that a pair (a,b) is a valid affine transformation for modulo m."""

    def validator(s: str) -> bool:
        try:
            a, b = _parse_affine_pair(s)
            # The key condition for an affine cipher to be invertible is that
            # 'a' must be coprime to 'm'.
            return _gcd(a, m) == 1
        except ValueError:
            return False

    return validator


@registry.register("set_samplers", "affine_param_sampler")
def affine_param_sampler(m: int) -> callable:
    """Generates random, valid affine transformation pairs (a,b)."""
    # Pre-calculate all possible values for 'a' that are coprime to 'm'
    coprime_a = [a for a in range(1, m) if _gcd(a, m) == 1]
    if not coprime_a:
        raise ValueError(
            f"No numbers are coprime to m={m}, cannot form an affine group.")

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            a = random.choice(coprime_a)
            b = random.randint(0, m - 1)
            samples.append(f"({a},{b})")
        return samples

    return sampler


@registry.register("action_logics", "affine_compose_logic")
def affine_compose_logic(m: int) -> callable:
    """
    Creates logic for composing two affine transformations.
    T1(x) = a1*x + b1
    T2(x) = a2*x + b2
    T1 o T2 (apply T2 then T1) is T1(T2(x)) = a1*(a2*x + b2) + b1
                                          = (a1*a2)*x + (a1*b2 + b1)
    So, (a1,b1) o (a2,b2) -> (a1*a2, a1*b2 + b1)
    """

    def logic(s1: str, s2: str) -> str:
        a1, b1 = _parse_affine_pair(s1)
        a2, b2 = _parse_affine_pair(s2)

        a_res = (a1 * a2) % m
        b_res = (a1 * b2 + b1) % m

        return f"({a_res},{b_res})"

    return logic


@registry.register("action_logics", "affine_inverse_logic")
def affine_inverse_logic(m: int) -> callable:
    """
    Finds the inverse of an affine transformation.
    If y = (a*x + b) % m, we solve for x:
    x = a^-1 * (y - b) % m = (a^-1 * y - a^-1 * b) % m
    So the inverse transform is (a_inv, -a_inv * b).
    """

    def logic(s: str) -> str:
        a, b = _parse_affine_pair(s)

        # Calculate modular inverse of a
        a_inv = _modInverse(a, m)
        if a_inv is None:
            raise ValueError(
                f"Cannot invert {s}: 'a'={a} is not coprime to m={m}.")

        # Calculate the new 'b' component of the inverse
        b_inv = (-a_inv * b) % m

        return f"({a_inv},{b_inv})"

    return logic


# Add these helper functions and components to components.py

# A configuration dictionary for Flexbox cycle operations.
# Format: { 'name': (modulus, ordered_values) }
FLEXBOX_FIELDS_CONFIG = {
    'J_CYCLE': (6, [
        'flex-start', 'flex-end', 'center', 'space-between', 'space-around',
        'space-evenly'
    ]),
    'A_CYCLE': (5, ['stretch', 'flex-start', 'flex-end', 'center', 'baseline'])
}


def _parse_cycle_ops(s: str) -> Dict[str, int]:
    """Parses 'J_CYCLE(2);A_CYCLE(4)' into {'J_CYCLE': 2, 'A_CYCLE': 4}."""
    if not s:
        return {}

    op_dict = {}
    parts = s.split(';')
    for part in parts:
        if not part: continue
        match = re.fullmatch(r"([A-Z_]+)\((\d+)\)", part)
        if match:
            op_name, value_str = match.groups()
            if op_name in FLEXBOX_FIELDS_CONFIG:
                op_dict[op_name] = int(value_str)
            else:
                raise ValueError(f"Unknown cycle op name: {op_name}")
        else:
            raise ValueError(f"Invalid cycle op format: {part}")
    return op_dict


def _format_cycle_ops(op_dict: Dict[str, int]) -> str:
    """Formats {'J_CYCLE': 2, 'A_CYCLE': 4} into a canonical string."""
    if not op_dict:
        return ""

    parts = []
    # Sort by op name for a canonical representation
    for op_name in sorted(op_dict.keys()):
        value = op_dict[op_name]
        modulus = FLEXBOX_FIELDS_CONFIG[op_name][0]
        # Store only the canonical representative in [0, modulus-1]
        canonical_value = value % modulus
        if canonical_value != 0:
            parts.append(f"{op_name}({canonical_value})")

    return ";".join(parts)


@registry.register("set_validators", "flexbox_op_validator")
def flexbox_op_validator() -> callable:
    """Validates Flexbox cycle operation strings."""

    def validator(s: str) -> bool:
        try:
            parsed = _parse_cycle_ops(s)
            # A strong validation: does it re-format to itself?
            # This checks for canonical form (no zero values, correct sorting).
            formatted = _format_cycle_ops(parsed)
            return s == formatted
        except ValueError:
            return False

    return validator


@registry.register("set_samplers", "flexbox_op_sampler")
def flexbox_op_sampler() -> callable:
    """Generates random Flexbox cycle operation strings."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            op_dict = {}
            for op_name, (modulus, _) in FLEXBOX_FIELDS_CONFIG.items():
                if random.random() < 0.6:  # 60% chance to include this op
                    value = random.randint(1, modulus -
                                           1)  # Sample non-identity ops
                    op_dict[op_name] = value

            samples.append(_format_cycle_ops(op_dict))
        return samples

    return sampler


@registry.register("action_logics", "flexbox_op_compose_logic")
def flexbox_op_compose_logic() -> callable:
    """Composes two Flexbox cycle operations."""

    def logic(s1: str, s2: str) -> str:
        d1 = _parse_cycle_ops(s1)
        d2 = _parse_cycle_ops(s2)
        res_dict = {}
        all_keys = set(d1.keys()).union(set(d2.keys()))

        for op_name in all_keys:
            modulus = FLEXBOX_FIELDS_CONFIG[op_name][0]
            val1 = d1.get(op_name, 0)
            val2 = d2.get(op_name, 0)
            res_dict[op_name] = (val1 + val2) % modulus

        return _format_cycle_ops(res_dict)

    return logic


@registry.register("action_logics", "flexbox_op_inverse_logic")
def flexbox_op_inverse_logic() -> callable:
    """Finds the inverse of a Flexbox cycle operation."""

    def logic(s: str) -> str:
        d = _parse_cycle_ops(s)
        inv_dict = {}
        for op_name, value in d.items():
            modulus = FLEXBOX_FIELDS_CONFIG[op_name][0]
            inv_dict[op_name] = (-value) % modulus
        return _format_cycle_ops(inv_dict)

    return logic


# Add these helper functions and components to components.py


def _parse_2d_vector(v_str: str) -> tuple[int, int]:
    """Parses a string like '(+2,-1)' or '(2,-1)' into a tuple (2, -1)."""
    match = re.fullmatch(r"\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)", v_str)
    if not match:
        raise ValueError(f"Invalid 2D vector format: {v_str}")
    return int(match.group(1)), int(match.group(2))


def _format_2d_vector(v_tuple: tuple[int, int]) -> str:
    """Formats a tuple (2, -1) into a canonical string '(+2,-1)'."""
    x, y = v_tuple
    # Using '+' to show sign for positive numbers is a nice canonical choice
    return f"({x:+},{y:+})"


@registry.register("set_validators", "chess_vector_validator")
def chess_vector_validator() -> callable:
    """Validates 2D integer vector strings like '(+2,-1)'."""

    def validator(s: str) -> bool:
        try:
            # A simple parse check is sufficient.
            _parse_2d_vector(s)
            return True
        except ValueError:
            return False

    return validator


@registry.register("set_samplers", "chess_vector_sampler")
def chess_vector_sampler(max_abs_val: int = 8) -> callable:
    """Generates random 2D integer vectors."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            x = random.randint(-max_abs_val, max_abs_val)
            y = random.randint(-max_abs_val, max_abs_val)
            # Ensure we don't only sample the identity
            if x == 0 and y == 0 and random.random(
            ) < 0.9:  # 90% chance to re-roll if it's (0,0)
                x = random.randint(-max_abs_val, max_abs_val)
                y = random.randint(-max_abs_val, max_abs_val)
            samples.append(_format_2d_vector((x, y)))
        return samples

    return sampler


@registry.register("action_logics", "vector_add_logic")
def vector_add_logic() -> callable:
    """Creates logic for adding two 2D integer vectors."""

    def logic(v1_str: str, v2_str: str) -> str:
        x1, y1 = _parse_2d_vector(v1_str)
        x2, y2 = _parse_2d_vector(v2_str)

        res_x = x1 + x2
        res_y = y1 + y2

        return _format_2d_vector((res_x, res_y))

    return logic


@registry.register("action_logics", "vector_inverse_logic")
def vector_inverse_logic() -> callable:
    """Creates logic for finding the additive inverse of a 2D integer vector."""

    def logic(v_str: str) -> str:
        x, y = _parse_2d_vector(v_str)
        return _format_2d_vector((-x, -y))

    return logic


# Add these helper functions and components to components.py


def _parse_braid_ops(s: str) -> list[str]:
    """Parses a braid string like "s1 s2 s1'" into ['s1', 's2', 's1'']"""
    if not s:
        return []
    return re.findall(r"s\d+#?", s)


def _get_braid_op_inverse(op: str) -> str:
    """Gets the inverse of a single braid generator. s1 -> s1', s1' -> s1."""
    return op[:-1] if op.endswith("#") else op + "#"


def _create_braid_simplifier(n: int) -> callable:
    """
    A factory that creates a braid simplification function for a given number of strands 'n'.
    This function will reduce any braid word to its unique canonical form.
    """
    # --- Pre-generate all possible replacement patterns for this n ---
    yb_replacements = {}
    for i in range(1, n - 1):
        s_i, s_i_p = f"s{i}", f"s{i}#"
        s_j, s_j_p = f"s{i+1}", f"s{i+1}#"
        # Positive form
        p1, p2 = f"{s_i} {s_j} {s_i}", f"{s_j} {s_i} {s_j}"
        if p1 > p2: yb_replacements[p1] = p2
        else: yb_replacements[p2] = p1
        # Negative form
        n1, n2 = f"{s_i_p} {s_j_p} {s_i_p}", f"{s_j_p} {s_i_p} {s_j_p}"
        if n1 > n2: yb_replacements[n1] = n2
        else: yb_replacements[n2] = n1

    sorted_yb_keys = sorted(yb_replacements.keys(), key=len, reverse=True)

    def simplify(ops: list[str]) -> list[str]:
        """The actual simplification logic."""
        while True:
            initial_len = len(ops)
            # Phase 1: Inverse Cancellation
            i = 0
            temp_ops = list(ops)
            while i < len(temp_ops) - 1:
                if temp_ops[i] == _get_braid_op_inverse(temp_ops[i + 1]):
                    temp_ops.pop(i)
                    temp_ops.pop(i)
                    i = max(0, i - 1)
                else:
                    i += 1
            ops = temp_ops
            if len(ops) != initial_len: continue

            # Phase 2: Yang-Baxter & Far-Commutativity
            made_internal_change = False
            temp_str = " ".join(ops)
            for lhs in sorted_yb_keys:
                if lhs in temp_str:
                    rhs = yb_replacements[lhs]
                    temp_str = temp_str.replace(lhs, rhs, 1)
                    ops = temp_str.split()
                    made_internal_change = True
                    break
            if made_internal_change: continue

            for i in range(len(ops) - 1):
                idx1 = int(ops[i].replace("#", "")[1:])
                idx2 = int(ops[i + 1].replace("#", "")[1:])
                if abs(idx1 - idx2) > 1 and idx1 > idx2:
                    ops[i], ops[i + 1] = ops[i + 1], ops[i]
                    made_internal_change = True
                    break
            if made_internal_change: continue

            break  # No changes in a full pass, sequence is canonical
        return ops

    return simplify


@registry.register("set_validators", "braid_op_validator")
def braid_op_validator(n: int) -> callable:
    """Validates a braid operation sequence for n strands."""
    # Generators are s1, s2, ..., s(n-1)
    max_index = n - 1

    def validator(s: str) -> bool:
        if s == "": return True
        try:
            ops = _parse_braid_ops(s)
            for op in ops:
                base = op.replace("#", "")
                index = int(base[1:])
                if not (1 <= index <= max_index):
                    return False
            return True
        except (ValueError, IndexError):
            return False

    return validator


@registry.register("set_samplers", "braid_op_sampler")
def braid_op_sampler(n: int, max_len: int = 5) -> callable:
    """
    Generates random sequences of braid operations in their fully simplified,
    canonical form.
    """
    # Create a simplifier function once, specific to the number of strands 'n'.
    simplifier = _create_braid_simplifier(n)

    generators = []
    for i in range(1, n):
        generators.append(f"s{i}")
        generators.append(f"s{i}#")

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            # Generate a random, potentially unsimplified sequence.
            length = random.randint(0, max_len)
            sequence = random.choices(generators, k=length)

            # **[CRITICAL STEP]** Simplify the random sequence to its canonical form.
            simplified_sequence = simplifier(sequence)

            # The result is now guaranteed to be a canonical braid word.
            samples.append(" ".join(simplified_sequence))
        return samples

    return sampler


# In components.py, FIND and REPLACE the entire braid_compose_logic function block.


@registry.register("action_logics", "braid_compose_logic")
def braid_compose_logic(n: int) -> callable:
    """
    Composes two braid sequences and simplifies the result to its canonical form.
    """
    # Get the same simplifier function used by the sampler.
    simplifier = _create_braid_simplifier(n)

    def logic(s1: str, s2: str) -> str:
        ops1 = _parse_braid_ops(s1)
        ops2 = _parse_braid_ops(s2)
        combined = ops1 + ops2
        simplified_ops = simplifier(combined)
        return " ".join(simplified_ops)

    return logic


@registry.register("action_logics", "braid_inverse_logic")
def braid_inverse_logic() -> callable:
    """Finds the inverse of a braid sequence. (A B C)' = C' B' A'"""

    def logic(s: str) -> str:
        ops = _parse_braid_ops(s)
        inv_ops = [_get_braid_op_inverse(op) for op in reversed(ops)]
        return " ".join(inv_ops)

    return logic


# Add these components to components.py


@registry.register("set_validators", "list_move_op_validator")
def list_move_op_validator() -> callable:
    """Validates a sequence of list move operations like 'MOVE(i,j)'."""
    op_pattern = re.compile(r"^MOVE\(\d+,\d+\)$")

    def validator(s: str) -> bool:
        if s == "": return True
        ops = s.split(';')
        return all(op_pattern.fullmatch(op) for op in ops)

    return validator


@registry.register("set_samplers", "list_move_op_sampler")
def list_move_op_sampler(list_size: int = 10, max_len: int = 3) -> callable:
    """Generates random sequences of valid list move operations."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            length = random.randint(0, max_len)
            if length == 0:
                samples.append("")
                continue

            op_list = []
            # We don't need to simulate the list for sampling, just ensure indices are plausible
            for _ in range(length):
                # Sample two DISTINCT indices
                indices = random.sample(range(list_size), 2)
                idx_from, idx_to = indices[0], indices[1]
                op_list.append(f"MOVE({idx_from},{idx_to})")

            samples.append(";".join(op_list))
        return samples

    return sampler


@registry.register("action_logics", "list_move_inverse_logic")
def list_move_inverse_logic() -> callable:
    """
    Finds the inverse of a list move operation sequence.
    inv(A;B) = inv(B);inv(A).
    The inverse of MOVE(i,j) is more complex than MOVE(j,i) due to index shifts.
    This model uses a conceptual INV_MOVE for simplicity and robustness.
    """

    def logic(s: str) -> str:
        if not s: return ""

        ops = s.split(';')
        inv_ops = []

        for op_str in reversed(ops):
            match = re.fullmatch(r"MOVE\((\d+),(\d+)\)", op_str)
            if match:
                idx_from, idx_to = match.groups()
                # A simple conceptual inverse that swaps from and to.
                # This is not perfectly accurate for list indices but is algebraically sound
                # for a free group model.
                inv_ops.append(f"MOVE({idx_to},{idx_from})")
            else:
                # Handle potential INV_MOVE(i,j) if we were to extend the model
                inv_ops.append(f"INV({op_str})")

        return ";".join(inv_ops)

    return logic


# Add these helper functions and components to components.py


def _parse_history_nav(nav_str: str) -> int:
    """
    Parses a navigation string like 'BACK(3)' into -3, 'FWD(2)' into 2.
    An empty string is 0.
    """
    if not nav_str:
        return 0

    match = re.fullmatch(r"(BACK|FWD)\((\d+)\)", nav_str)
    if not match:
        raise ValueError(f"Invalid history navigation format: {nav_str}")

    action, amount_str = match.groups()
    amount = int(amount_str)

    if action == "BACK":
        return -amount
    else:  # FWD
        return amount


def _format_history_nav(net_change: int) -> str:
    """
    Formats a numeric value into a canonical navigation string.
    e.g., -3 -> 'BACK(3)', 2 -> 'FWD(2)', 0 -> ''
    """
    if net_change == 0:
        return ""

    if net_change < 0:
        return f"BACK({-net_change})"
    else:  # net_change > 0
        return f"FWD({net_change})"


@registry.register("set_validators", "history_nav_validator")
def history_nav_validator() -> callable:
    """Validates history navigation strings like 'BACK(3)'."""

    def validator(s: str) -> bool:
        if s == "": return True
        return bool(re.fullmatch(r"(BACK|FWD)\(\d+\)", s))

    return validator


@registry.register("set_samplers", "history_nav_sampler")
def history_nav_sampler(max_steps: int = 5) -> callable:
    """Generates random history navigation strings."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            steps = random.randint(-max_steps, max_steps)
            # 10% chance of sampling identity, otherwise re-roll if it's 0
            if steps == 0 and random.random() < 0.9:
                steps = random.randint(-max_steps, max_steps)
                if steps == 0: steps = 1  # ensure it's not 0

            samples.append(_format_history_nav(steps))
        return samples

    return sampler


@registry.register("action_logics", "history_nav_compose_logic")
def history_nav_compose_logic() -> callable:
    """
    Creates logic for composing two navigation operations by summing their net effect.
    """

    def logic(s1_str: str, s2_str: str) -> str:
        # NOTE: The template has a complex format 'BACK(3);FWD(1)'.
        # A true group element should represent the NET effect, e.g., 'BACK(2)'.
        # We will implement the group of NET effects.
        net1 = _parse_history_nav(s1_str)
        net2 = _parse_history_nav(s2_str)
        total_net = net1 + net2
        return _format_history_nav(total_net)

    return logic


@registry.register("action_logics", "history_nav_inverse_logic")
def history_nav_inverse_logic() -> callable:
    """
    Creates logic for finding the inverse of a navigation operation.
    """

    def logic(s_str: str) -> str:
        net = _parse_history_nav(s_str)
        return _format_history_nav(-net)

    return logic


########


def _find_permutation_cycles(p: tuple) -> str:
    """辅助函数，将置换元组如 (1, 2, 0) 转换为循环表示法字符串如 '(0 1 2)'。"""
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
            # 我们只关心长度大于1的循环
            if len(cycle) > 1:
                cycles.append(" ".join(map(str, cycle)))

    if not cycles:
        return "()"  # 这是单位元置换

    # 将所有循环连接起来，例如 (0 1)(2 3)
    return "".join(f"({c})" for c in cycles)


@registry.register("param_generators", "permutation_elements")
def generate_permutation_elements(n: int) -> List[str]:
    """为 S_n 中的所有置换生成人类可读的名称，使用循环表示法。"""
    # 获取 n 个元素的所有置换
    perms = list(permutations(range(n)))
    # 将每个置换元组转换为其循环表示法字符串
    return [_find_permutation_cycles(p) for p in perms]


@registry.register("param_generators", "permutation_composition_table")
def generate_permutation_composition_table(n: int) -> List[List[int]]:
    """为对称群 S_n 生成完整的凯莱表。"""
    # 按规范顺序列出所有置换
    perms_list = list(permutations(range(n)))

    # 创建一个从置换元组到其索引的反向映射，以便快速查找
    perm_to_index = {p: i for i, p in enumerate(perms_list)}

    num_perms = len(perms_list)
    table = [[0] * num_perms for _ in range(num_perms)]

    # 遍历所有置换对 (p1, p2)
    for i in range(num_perms):
        for j in range(num_perms):
            p1 = perms_list[i]
            p2 = perms_list[j]

            # 复合 p1 ∘ p2 意味着先应用 p2，再应用 p1
            # p_comp[k] = p1[p2[k]]
            p_comp = tuple(p1[p2[k]] for k in range(n))

            # 查找结果置换的索引
            result_index = perm_to_index[p_comp]
            table[i][j] = result_index

    return table


# ==============================================================================
# --- General Purpose / Reusable Components ---
# ==============================================================================


@registry.register("set_validators", "is_in_list")
def is_in_list_validator(elements: List[str]) -> callable:
    """A general validator that checks if an item is in a predefined list."""
    valid_set = set(elements)
    return lambda s: s in valid_set


@registry.register("set_samplers", "sampler_from_alphabet")
def sampler_from_alphabet(alphabet: List[str],
                          max_len: int,
                          sep: str = "") -> callable:
    """A general sampler that creates sequences from a given alphabet of symbols."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            length = random.randint(1, max_len)
            sequence = random.choices(alphabet, k=length)
            samples.append(sep.join(sequence))
        return samples

    return sampler


@registry.register("param_generators", "python_code_generator")
def python_code_generator(code: str) -> Any:
    """
    [Param Generator] Executes a line of Python code to generate a parameter list.
    WARNING: Uses eval(), and should only be used in a trusted environment.
    """
    # Provide a safe context for eval, allowing specific modules like itertools
    safe_context = {"__import__": __import__}
    return eval(code, safe_context)


# ==============================================================================
# --- Components for "Audio Volume Slider (R)" ---
# ==============================================================================


@registry.register("set_validators", "decibel_validator")
def decibel_validator() -> callable:
    """Validates a float string ending in 'dB'."""
    regex = re.compile(r"^-?\d+(\.\d+)?dB$")
    return lambda s: bool(regex.fullmatch(s))


@registry.register("action_logics", "decibel_inverse_logic")
def decibel_inverse_logic() -> callable:
    """Calculates the inverse of a decibel adjustment."""

    def parse_db(s: str) -> float:
        return float(s[:-2])

    def logic(db_str: str) -> str:
        val = parse_db(db_str)
        return f"{-val:.1f}dB"

    return logic


@registry.register("set_samplers", "decibel_sampler")
def decibel_sampler(min_db: float = -60.0, max_db: float = 6.0) -> callable:
    """Generates random decibel level strings."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            val = random.uniform(min_db, max_db)
            samples.append(f"{val:.1f}dB")
        return samples

    return sampler


@registry.register("action_logics", "decibel_add_logic")
def decibel_add_logic() -> callable:
    """Adds two decibel adjustment values."""

    def parse_db(s: str) -> float:
        return float(s[:-2])

    def logic(db1_str: str, db2_str: str) -> str:
        val1 = parse_db(db1_str)
        val2 = parse_db(db2_str)
        return f"{val1 + val2:.1f}dB"

    return logic


# ==============================================================================
# --- Components for "Combination Lock (Zn_k)" ---
# ==============================================================================


@registry.register("set_validators", "combination_lock_validator")
def combination_lock_validator(dials: int, sep: str = "-") -> callable:
    """Validates a combination like '8-1-5' for a 3-dial lock."""
    # Pattern: a digit, followed by (separator and digit) n-1 times.
    pattern = r"^\d" + (re.escape(sep) + r"\d") * (dials - 1) + "$"
    regex = re.compile(pattern)
    return lambda s: bool(regex.fullmatch(s))


@registry.register("action_logics", "lock_inverse_logic")
def lock_inverse_logic(sep: str = "-") -> callable:
    """Calculates the inverse of a lock combination."""

    def logic(combo_str: str) -> str:
        nums = [int(n) for n in combo_str.split(sep)]
        inv_nums = [(10 - n) % 10 for n in nums]
        return sep.join(map(str, inv_nums))

    return logic


@registry.register("set_samplers", "combination_lock_sampler")
def combination_lock_sampler(dials: int, sep: str = "-") -> callable:
    """Generates random lock combinations."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            combo = [str(random.randint(0, 9)) for _ in range(dials)]
            samples.append(sep.join(combo))
        return samples

    return sampler


@registry.register("action_logics", "lock_turn_dials_logic")
def lock_turn_dials_logic(sep: str = "-") -> callable:
    """Adds two combinations component-wise, simulating turning dials."""

    def logic(combo1_str: str, combo2_str: str) -> str:
        nums1 = [int(n) for n in combo1_str.split(sep)]
        nums2 = [int(n) for n in combo2_str.split(sep)]
        new_nums = [(n1 + n2) % 10 for n1, n2 in zip(nums1, nums2)]
        return sep.join(map(str, new_nums))

    return logic


# ==============================================================================
# --- Components for "Hex Tile Orientation (D6)" ---
# ==============================================================================


def _generate_dihedral_table(n: int) -> List[List[int]]:
    """A general Cayley table generator for any Dihedral group Dn."""
    size = 2 * n
    table = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            is_i_rot = (i < n)
            is_j_rot = (j < n)
            i_val = i % n
            j_val = j % n
            if is_i_rot and is_j_rot:  # R*R -> R
                table[i][j] = (i_val + j_val) % n
            elif is_i_rot and not is_j_rot:  # R*F -> F
                table[i][j] = ((i_val + j_val) % n) + n
            elif not is_i_rot and is_j_rot:  # F*R -> F
                table[i][j] = ((i_val - j_val + n) % n) + n
            else:  # F*F -> R
                table[i][j] = (i_val - j_val + n) % n
    return table


def _create_inverse_lookup(table: List[List[int]],
                           identity_index: int = 0) -> List[int]:
    """Helper function to create a 1D inverse map from a Cayley table."""
    size = len(table)
    inverses = [0] * size
    for i in range(size):
        try:
            # The inverse of i is j if table[i][j] == identity
            inverse_of_i = table[i].index(identity_index)
            inverses[i] = inverse_of_i
        except ValueError:
            # This should not happen in a valid group
            raise ValueError(f"Could not find an inverse for element {i}.")
    return inverses


@registry.register("action_logics", "dihedral_inverse_logic")
def dihedral_inverse_logic(n: int, elements: List[str]) -> callable:
    """Calculates the inverse of a Dn operation."""
    dn_table = _generate_dihedral_table(n)
    element_map = {el: i for i, el in enumerate(elements)}
    # Identity is always the first element (rotation 0)
    inverse_map = _create_inverse_lookup(dn_table, identity_index=0)

    def logic(op_str: str) -> str:
        idx = element_map[op_str]
        inv_idx = inverse_map[idx]
        return elements[inv_idx]

    return logic


@registry.register("action_logics", "dihedral_group_logic")
def dihedral_group_logic(n: int, elements: List[str]) -> callable:
    """A general logic for any Dn group, powered by a generated Cayley table."""
    element_map = {el: i for i, el in enumerate(elements)}
    dn_table = _generate_dihedral_table(n)

    def logic(op1_str: str, op2_str: str) -> str:
        idx1 = element_map[op1_str]
        idx2 = element_map[op2_str]
        result_idx = dn_table[idx1][idx2]
        return elements[result_idx]

    return logic


# ==============================================================================
# --- Components for "Music Playlist Shuffling (Sn)" ---
# ==============================================================================


@registry.register("param_generators", "permutation_elements_generator")
def permutation_elements_generator(elements: List[str],
                                   sep: str = ",") -> List[str]:
    """
    [Param Generator] Safely generates all permutations of a list of elements
    and formats them into strings like '(A,B,C)'. This replaces the unsafe eval-based generator.
    """
    # Generate all permutation tuples from the input list
    perms = permutations(elements)
    # Format each tuple into the desired string format

    return [f"({sep.join(p)})" for p in perms]


@registry.register("action_logics", "playlist_shuffle_logic")
def playlist_shuffle_logic(tracks: List[str], sep: str = ",") -> callable:
    """
    [Action Logic] Composes two playlist permutations.
    This logic applies the second permutation first, then the first one.
    """
    # Create a mapping from track name to its original index for quick lookups
    track_to_idx = {track: i for i, track in enumerate(tracks)}
    num_tracks = len(tracks)

    def parse_perm(perm_str: str) -> tuple:
        """Helper to parse a string like '(A,C,B)' into an index tuple (0, 2, 1)."""
        # Strip parentheses and split by the separator

        perm_tracks = perm_str.strip()[1:-1].split(sep)
        return tuple(track_to_idx[t] for t in perm_tracks)

    def format_playlist(perm_tuple: tuple) -> str:
        """Helper to format an index tuple (0, 2, 1) back to a string '(A,C,B)'."""
        ordered_tracks = [tracks[i] for i in perm_tuple]
        return f"({sep.join(ordered_tracks)})"

    def logic(perm1_str: str, perm2_str: str) -> str:
        # Convert the string representations into index-based permutations
        p1 = parse_perm(perm1_str)
        p2 = parse_perm(perm2_str)

        # Composition p1 o p2 means apply p2 then p1.
        # The track at the new position k is the track that was at the old position p2[k],
        # which is then moved by p1. So, new_perm[k] = p1[p2[k]].
        # Let's trace: If original is (0,1,2). p2=(0,2,1). Playlist becomes (A,C,B).
        # Now apply p1=(1,0,2) to (A,C,B). New list is (C,A,B). Indices are (2,0,1).
        # Let's check with formula:
        # k=0: p1[p2[0]] = p1[0] = 1. No, this is wrong.
        #
        # Correct logic: We are permuting the *elements*, not the positions.
        # Let the list of elements be E. p1 is a reordering of E. p2 is another.
        # The operation is on the permutations themselves.
        # Let p1=(1,0,2) and p2=(0,2,1).
        # p_comp[i] = p1[p2[i]]
        # p_comp[0] = p1[p2[0]] = p1[0] = 1
        # p_comp[1] = p1[p2[1]] = p1[2] = 2
        # p_comp[2] = p1[p2[2]] = p1[1] = 0
        # Result is (1,2,0)
        p_comp = list(tuple(p1[p2[i]] for i in range(num_tracks)))

        return format_playlist(p_comp)

    return logic


@registry.register("action_logics", "playlist_inverse_logic")
def playlist_inverse_logic(tracks: List[str], sep: str = ",") -> callable:
    """
    [Action Logic] Finds the inverse of a given playlist permutation.
    """
    track_to_idx = {track: i for i, track in enumerate(tracks)}
    num_tracks = len(tracks)

    def parse_perm(perm_str: str) -> tuple:
        """Helper to parse a string like '(A,C,B)' into an index tuple (0, 2, 1)."""
        perm_tracks = perm_str.strip()[1:-1].split(sep)
        return tuple(track_to_idx[t] for t in perm_tracks)

    def format_playlist(perm_tuple: tuple) -> str:
        """Helper to format an index tuple (0, 2, 1) back to a string '(A,C,B)'."""
        ordered_tracks = [tracks[i] for i in perm_tuple]
        return f"({sep.join(ordered_tracks)})"

    def logic(perm_str: str) -> str:
        p = parse_perm(perm_str)

        # To find the inverse, if p[i] = j, then inv_p[j] = i.
        inv_p_list = [0] * num_tracks
        for i in range(num_tracks):
            j = p[i]
            inv_p_list[j] = i

        inv_p = tuple(inv_p_list)
        return format_playlist(inv_p)

    return logic


# ==============================================================================
# --- Components for "Text Editor Cursor (Z^2)" ---
# ==============================================================================


@registry.register("set_validators", "grid_coord_validator")
def grid_coord_validator() -> callable:
    """Validates coordinates like '(3, -5)'."""
    regex = re.compile(r"^\(-?\d+,\s*-?\d+\)$")
    return lambda s: bool(regex.fullmatch(s))


@registry.register("set_samplers", "grid_coord_sampler")
def grid_coord_sampler(max_coord: int = 10) -> callable:
    """Generates random grid coordinates."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
            samples.append(f"({x},{y})")
        return samples

    return sampler


@registry.register("action_logics", "grid_vector_add_logic")
def grid_vector_add_logic() -> callable:
    """Adds two coordinate vectors."""

    def parse_coord(s: str) -> tuple[int, int]:
        parts = s.strip()[1:-1].split(',')
        return int(parts[0]), int(parts[1])

    def logic(coord1_str: str, coord2_str: str) -> str:
        x1, y1 = parse_coord(coord1_str)
        x2, y2 = parse_coord(coord2_str)
        return f"({x1 + x2},{y1 + y2})"

    return logic


# --- NEW LOGIC TO IMPLEMENT ---
@registry.register("action_logics", "grid_vector_inverse_logic")
def grid_vector_inverse_logic() -> callable:
    """
    [Action Logic] Calculates the inverse of a coordinate vector.
    The inverse of a move (r,c) is the move that cancels it: (-r,-c).
    """

    def parse_coord(s: str) -> tuple[int, int]:
        parts = s.strip()[1:-1].split(',')
        return int(parts[0]), int(parts[1])

    def logic(coord_str: str) -> str:
        r, c = parse_coord(coord_str)
        # Return the negated coordinate pair
        return f"({-r},{-c})"

    return logic


# ==============================================================================
# --- Components for "Text Editor History (Fn)" ---
# ==============================================================================
@registry.register("action_logics", "text_op_inverse_logic")
def text_op_inverse_logic() -> callable:
    """Calculates the inverse of a sequence of text operations."""

    def get_inv_op(op: str) -> str:
        if op.startswith("i"):
            return "b"
        elif op == "b":
            # The conceptual inverse of backspace is an insert of the deleted char.
            # In a free group, we can represent this as a generic 'Insert' operation.
            return "i(*)"  # Represents inserting an unknown character
        return ""

    def parse_ops(s: str) -> List[str]:
        return re.findall(r"i\(.\)|b", s)

    def logic(op_seq: str) -> str:
        ops = parse_ops(op_seq)
        inv_ops = [get_inv_op(op) for op in reversed(ops)]
        return "".join(inv_ops)

    return logic


@registry.register("action_logics", "text_op_compose_logic")
def text_op_compose_logic() -> callable:
    """Composes text editor operations, simplifying inverse pairs (e.g., insert/backspace)."""

    def parse_ops(s: str) -> List[str]:
        return re.findall(r"i\(.\)|b", s)

    def logic(op_seq1: str, op_seq2: str) -> str:
        full_ops = parse_ops(op_seq1) + parse_ops(op_seq2)
        made_change = True
        while made_change:
            made_change = False
            i = 0
            while i < len(full_ops) - 1:
                # Rule: an insert followed by a backspace cancels
                if full_ops[i].startswith("i") and full_ops[i + 1] == "b":
                    full_ops.pop(i)
                    full_ops.pop(i)
                    made_change = True
                    break  # Restart scan
                i += 1
        return "".join(full_ops)

    return logic


# ==============================================================================
# --- Components for "Traffic Light Cycle (Z3)" ---
# ==============================================================================


@registry.register("action_logics", "cycle_next_state_logic")
def cycle_next_state_logic(states: List[str]) -> callable:
    """Logic to advance to the next state in a fixed cycle."""
    state_map = {state: i for i, state in enumerate(states)}
    num_states = len(states)

    def logic(current_state: str, steps_str: str) -> str:

        current_idx = state_map[current_state]
        steps = state_map[steps_str]
        next_idx = (current_idx + steps) % num_states

        return states[next_idx]

    return logic


@registry.register("action_logics", "cycle_inverse_state_logic")
def cycle_inverse_state_logic(states: List[str]) -> callable:
    """Calculates the inverse state in a cycle."""
    state_map = {state: i for i, state in enumerate(states)}
    n = len(states)

    def logic(state_str: str) -> str:
        idx = state_map[state_str]
        inv_idx = (n - idx) % n
        # The inverse of state 0 is state 0 (the identity)
        if idx == 0:
            inv_idx = 0
        return states[inv_idx]

    return logic


# ==============================================================================
# --- NEW Components for "Tally Mark Text Editor Cursor" ---
# ==============================================================================


# --- Tally Mark Number System Helpers ---
def _tally_to_int(tally_str: str) -> int:
    """Converts a tally string like '>>>' or '<<' to an integer."""
    if not tally_str:
        return 0
    if tally_str[0] == '>':
        return len(tally_str)
    if tally_str[0] == '<':
        return -len(tally_str)
    raise ValueError(f"Invalid tally mark format: {tally_str}")


def _int_to_tally(n: int) -> str:
    """Converts an integer to a tally string."""
    if n == 0:
        return ""
    if n > 0:
        return '>' * n
    if n < 0:
        return '<' * abs(n)


def _parse_tally_coord(coord_str: str) -> tuple[int, int]:
    """Parses a tally coordinate string '(>>>,<<)' into an integer tuple (3, -2)."""
    # Strip parentheses and split on the comma
    parts = coord_str.strip()[1:-1].split(',')
    row_tally = parts[0]
    col_tally = parts[1]
    return _tally_to_int(row_tally), _tally_to_int(col_tally)


# --- Registered Components for the Tally Mark System ---


@registry.register("set_validators", "tally_coord_validator")
def tally_coord_validator() -> callable:
    """Validates coordinates using the tally mark system, e.g., '(>>,<)'."""
    # Pattern: An opening paren, followed by zero or more > or <, a comma,
    # zero or more > or <, and a closing paren.
    # It also ensures a part is not a mix of > and <.
    regex = re.compile(
        r"^\((>*)<*,(>*)<*\)$|^\((>*)<*,(<*)\)$|^\((<*),(>*)<*\)$|^\((<*),(<*)\)$"
    )

    def validator(s: str) -> bool:
        # A simpler check is to try parsing it. If it succeeds, it's valid.
        try:
            _parse_tally_coord(s)
            return True
        except (ValueError, IndexError):
            return False

    return validator


@registry.register("set_samplers", "tally_coord_sampler")
def tally_coord_sampler(max_abs_val: int = 5) -> callable:
    """Generates random coordinates using the tally mark system."""

    def sampler(k: int) -> List[str]:
        samples = []
        for _ in range(k):
            row_int = random.randint(-max_abs_val, max_abs_val)
            col_int = random.randint(-max_abs_val, max_abs_val)
            row_tally = _int_to_tally(row_int)
            col_tally = _int_to_tally(col_int)
            samples.append(f"({row_tally},{col_tally})")
        return samples

    return sampler


@registry.register("action_logics", "tally_vector_add_logic")
def tally_vector_add_logic() -> callable:
    """Adds two tally coordinate vectors."""

    def logic(coord1_str: str, coord2_str: str) -> str:
        r1, c1 = _parse_tally_coord(coord1_str)
        r2, c2 = _parse_tally_coord(coord2_str)

        # Perform standard integer addition on the parsed values
        new_r, new_c = r1 + r2, c1 + c2

        # Convert the result back to the tally mark format
        return f"({_int_to_tally(new_r)},{_int_to_tally(new_c)})"

    return logic


@registry.register("action_logics", "tally_vector_inverse_logic")
def tally_vector_inverse_logic() -> callable:
    """Calculates the inverse of a tally coordinate vector."""

    def logic(coord_str: str) -> str:
        r, c = _parse_tally_coord(coord_str)

        # The inverse is simply the negation of the integer values
        inv_r, inv_c = -r, -c

        # Convert back to the tally mark format
        return f"({_int_to_tally(inv_r)},{_int_to_tally(inv_c)})"

    return logic
