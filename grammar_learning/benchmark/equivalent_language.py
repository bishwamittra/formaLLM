
from collections import defaultdict, Counter
import nltk
from nltk.grammar import Nonterminal, ProbabilisticProduction
from pcfg import PCFG

def learn_trie_pcfg_nltk(samples, start_symbol="S", minimize=True):
    """
    Learn an acyclic PCFG from a finite multiset of token sequences.
    If minimize=True, compresses the trie by merging equivalent suffix subtries
    (minimal number of nodes), while keeping probabilities correct.

    samples: list of sequences, e.g. [["1","2"], ["1","2"], ["3","1"]]
    Returns: nltk.PCFG
    """

    freq = Counter(tuple(s) for s in samples)

    # ---------- Build weighted trie ----------
    next_id = 0
    root = next_id
    next_id += 1

    children = defaultdict(dict)          # children[u][tok] = v
    pass_count = defaultdict(int)         # total mass passing through node u
    end_count = defaultdict(int)          # total mass ending at node u
    edge_count = defaultdict(lambda: defaultdict(int))  # edge_count[u][tok] = mass taking that edge

    for seq, c in freq.items():
        u = root
        pass_count[u] += c
        for tok in seq:
            if tok not in children[u]:
                v = next_id
                next_id += 1
                children[u][tok] = v
            v = children[u][tok]
            edge_count[u][tok] += c       # <-- critical: flow on this specific edge
            u = v
            pass_count[u] += c
        end_count[u] += c

    # ---------- Minimize trie to minimal DAG ----------
    if minimize:
        # postorder nodes (leaves first)
        order = []
        stack = [root]
        seen = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for v in children[u].values():
                stack.append(v)
            order.append(u)
        order.reverse()

        rep = {}
        registry = {}
        rep_id_counter = 0

        def intern(sig):
            nonlocal rep_id_counter
            if sig in registry:
                return registry[sig]
            rid = rep_id_counter
            rep_id_counter += 1
            registry[sig] = rid
            return rid

        # signatures based on structure (final flag + token-labeled edges to reps)
        for u in order:
            edges = tuple(sorted((tok, rep[v]) for tok, v in children[u].items()))
            is_final = end_count.get(u, 0) > 0
            rep[u] = intern((is_final, edges))

        new_root = rep[root]
        new_next_id = rep_id_counter

        # aggregate node counts
        new_pass = defaultdict(int)
        new_end = defaultdict(int)
        for u in range(next_id):
            ru = rep[u]
            new_pass[ru] += pass_count.get(u, 0)
            new_end[ru] += end_count.get(u, 0)

        # aggregate edge counts (this is the key fix)
        new_children = defaultdict(dict)
        new_edge = defaultdict(lambda: defaultdict(int))
        for u in range(next_id):
            ru = rep[u]
            for tok, v in children[u].items():
                rv = rep[v]
                new_children[ru][tok] = rv
                new_edge[ru][tok] += edge_count[u][tok]

        # replace
        root = new_root
        next_id = new_next_id
        children = new_children
        pass_count = new_pass
        end_count = new_end
        edge_count = new_edge

    # ---------- Convert minimized DAG to PCFG ----------
    NT = {u: Nonterminal(f"A_{u}") for u in range(next_id)}
    S = Nonterminal(start_symbol)

    prods = [ProbabilisticProduction(S, (NT[root],), prob=1.0)]

    for u in range(next_id):
        total = pass_count.get(u, 0)
        if total == 0:
            continue

        # epsilon (end) probability
        eps_mass = end_count.get(u, 0)
        if eps_mass > 0:
            prods.append(ProbabilisticProduction(NT[u], (), prob=eps_mass / total))

        # outgoing token transitions: use edge_count, NOT pass_count[child]
        out_mass = 0
        for tok, v in children[u].items():
            mass = edge_count[u][tok]
            out_mass += mass
            prods.append(ProbabilisticProduction(NT[u], (tok, NT[v]), prob=mass / total))

        # Optional sanity check (floating error tolerance)
        # eps_mass + out_mass should equal total
        # (they are integers, so it should be exact)
        # assert eps_mass + out_mass == total, (u, eps_mass, out_mass, total)

    return PCFG(S, prods)



import pickle
import pandas as pd
import os


def get_data(grammar_name):
    ideal_training_size = 5000
    if os.path.isfile(f"../data/{grammar_name}/sequences_w_edit_distance_{grammar_name}_{ideal_training_size * 2}_5.pkl"):
        with open(f"../data/{grammar_name}/sequences_w_edit_distance_{grammar_name}_10000_5.pkl", 'rb') as f:
            data = pickle.load(f)
            assert "train_sequences" in data
            training_size_original = len(data["train_sequences"])
            for key in data:
                # print(key, len(data[key]), len(set(data[key])))
                for i, sequence in enumerate(data[key]):
                    assert sequence == data[key][i]
    else:
        raise ValueError(f"Data not found: ../data/{grammar_name}/sequences_w_edit_distance_{grammar_name}_{ideal_training_size * 2}_5.pkl")

    return data


def get_equivalent_grammar(grammar_name="pcfg_tiny_demo_extended_incorrect"):

    data = get_data(grammar_name)
    assert "train_sequences" in data
    assert "test_sequences" in data
    in_language_data = data['train_sequences'] + data['test_sequences']
    grammar = learn_trie_pcfg_nltk(in_language_data)
    return grammar