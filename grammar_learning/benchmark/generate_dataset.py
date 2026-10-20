"""
Generate benchmark data for a single grammar and save pickle files to
grammar_learning/data/<grammar_name>/.

Output files (same as data_generation.ipynb):
  sequences_w_edit_distance_<name>_<N>_<seed>.pkl  — all splits
  meta_data_<name>_<N>_<seed>.pkl                  — NT position maps (train/test)
  grammar_meta_data_<name>.pkl                      — grammar metadata
  meta_data_lexer_edit_<name>_<N>_<seed>.pkl        — edit-distance metadata
  meta_data_grammar_edit_<name>_<N>_<seed>.pkl      — grammar-edit metadata

Run from the repo root:
    python grammar_learning/benchmark/generate_dataset.py --grammar_name pcfg_cfg3b_eq_len_skewed_prob
"""

import argparse
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pcfg import PCFG
from pcsg import PCSG, from_pcsg_string
from utils_grammar import (
    compute_random_guess_metric,
    compute_terminal_freq,
    generate_similar_sequences,
    get_grammatical_sentences,
    get_nongrammatical_sentences_from_perturbed_grammar,
    get_perturbed_grammar,
    random_sentence_generator,
    train_test_split,
    find_sentences_applied_by_a_rule,
)


def get_grammar_string(grammar_name: str):
    from utils_grammar import get_grammar_string as _get
    return _get(grammar_name)


def get_stat(sequences):
    lengths = [len(s) for s in sequences]
    return {
        "num sequences": len(sequences),
        "unique sequences": len(set(sequences)),
        "max": max(lengths),
        "min": min(lengths),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar_name", required=True, help="Grammar name (must exist in example_grammar.py)")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--store_path", default="../data",
                        help="Path relative to the benchmark/ directory")
    args = parser.parse_args()

    grammar_name = args.grammar_name
    num_samples = args.num_samples
    seed = args.seed

    # Resolve store_path relative to this script's directory
    store_path = Path(__file__).resolve().parent / args.store_path
    store_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load grammar
    # ------------------------------------------------------------------ #
    pfsa_object = {}
    if grammar_name.startswith("pfsa"):
        pfsa_object = get_grammar_string(grammar_name)
        if pfsa_object is None:
            raise ValueError(f"Unknown PFSA grammar: {grammar_name}")
        grammar_string = pfsa_object["grammar_string"]
        if pfsa_object["entropy_analytic"] > 1000:
            raise ValueError(f"Grammar {grammar_name} has too-high entropy — skipping")
    else:
        grammar_string = get_grammar_string(grammar_name)

    if grammar_name.startswith("pcfg") or grammar_name.startswith("preg") or grammar_name.startswith("pfsa"):
        grammar = PCFG.fromstring(grammar_string)
    elif grammar_name.startswith("pcsg"):
        grammar = from_pcsg_string(grammar_string)
    else:
        raise ValueError(f"Cannot determine grammar type from name: {grammar_name}")

    print(f"Grammar: {grammar_name}")
    print(grammar_string)
    print(f"Random-guess loss: {compute_random_guess_metric(grammar):.4f}")

    # ------------------------------------------------------------------ #
    # Generate grammatical sequences
    # ------------------------------------------------------------------ #
    sequences, nt_position_map, sequence_freq, sequence_prob_dict, nt_position_map_reverse = \
        get_grammatical_sentences(grammar, num_samples, seed)

    # Remove empty sequences
    sequences = [s for s in sequences if len(s) > 0]
    sequence_freq = {s: f for s, f in sequence_freq.items() if len(s) > 0}
    sequence_prob_dict = {s: p for s, p in sequence_prob_dict.items() if len(s) > 0}
    nt_position_map = {s: m for s, m in nt_position_map.items() if len(s) > 0}
    nt_position_map_reverse = {s: m for s, m in nt_position_map_reverse.items() if len(s) > 0}

    stat = get_stat(sequences)
    print(f"Generated {stat['num sequences']} sequences "
          f"({stat['unique sequences']} unique), "
          f"len {stat['min']}–{stat['max']}, mean {stat['mean']:.1f}")

    # ------------------------------------------------------------------ #
    # Grammar metadata
    # ------------------------------------------------------------------ #
    grammar_meta_data = {
        "grammar_name": grammar_name,
        "grammar_string": grammar_string,
        "terminals": list(grammar._lexical_index.keys()) if hasattr(grammar, "_lexical_index") else [],
        "num_terminals": len(grammar._lexical_index) if hasattr(grammar, "_lexical_index") else 0,
        "num_nonterminals": len(set(p.lhs() for p in grammar.productions())) if isinstance(grammar, PCFG) else len(grammar.nonterminals),
        "expected_length": float(np.mean([len(s) for s in sequences if len(s) > 0])),
    }
    for key in ("rank", "entropy_analytic"):
        if key in pfsa_object:
            grammar_meta_data[key] = pfsa_object[key]

    for prob_source, label in [(sequence_freq, "entropy_freq_approximation"),
                                (sequence_prob_dict, "entropy_prob_approximation")]:
        vals = list(prob_source.values())
        total = sum(vals)
        normed = [v / total for v in vals if v > 0]
        grammar_meta_data[label] = float(-sum(p * math.log2(p) for p in normed if p > 0))

    # ------------------------------------------------------------------ #
    # Train / test split
    # ------------------------------------------------------------------ #
    train_size = num_samples // 2
    test_size = num_samples - train_size
    non_grammatical_size = test_size

    random.seed(seed)
    random.shuffle(sequences)
    sequences = sequences[:train_size + test_size]

    ratio = train_size / (train_size + test_size)
    train_sequences, test_sequences = train_test_split(sequences, seed, sequence_freq, ratio)

    data = {
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
    }
    print(f"Train: {len(train_sequences)}, Test: {len(test_sequences)}")

    meta_data = {
        "non_terminal_applied_position_map": nt_position_map,
        "sequence_freq": sequence_freq,
        "sequence_prob_dict": sequence_prob_dict,
    }

    # ------------------------------------------------------------------ #
    # Non-grammatical sequences (random)
    # ------------------------------------------------------------------ #
    len_sequences = [len(s) for s in sequences]
    non_grammatical_sequences = random_sentence_generator(
        grammar=grammar,
        num_samples=non_grammatical_size,
        min_length=min(len_sequences),
        max_length=max(len_sequences),
        sampled_sequences=set(sequence_freq.keys()),
        seed=seed,
        timeout=100,
        terminal_freq=compute_terminal_freq(train_sequences + test_sequences),
    )
    data["non_grammatical_sequences"] = non_grammatical_sequences
    print(f"Non-grammatical (random): {len(non_grammatical_sequences)}")

    # ------------------------------------------------------------------ #
    # Grammar perturbation (PCFG only)
    # ------------------------------------------------------------------ #
    grammar_edit_dict = {}
    if grammar_name.startswith("pcfg"):
        total_runs = 5
        num_per_run = 200

        for edit_level in [1, 2, 3, 4, 5, 6, 7]:
            for edit in [1]:
                for forced_action in ["all"]:
                    for run_id in range(total_runs):
                        try:
                            grammar_perturbed, perturbation_result = get_perturbed_grammar(
                                grammar, grammar_name,
                                level=edit_level, edit=edit,
                                forced_action=None,  # "all" means no forced action
                                seed=seed + run_id, verbose=False,
                            )
                            ge_seqs, ge_nt_map, ge_freq, ge_prob = \
                                get_nongrammatical_sentences_from_perturbed_grammar(
                                    base_grammar=grammar,
                                    perturbed_grammar=grammar_perturbed,
                                    num_samples=num_per_run,
                                    seed=seed,
                                )
                            if not ge_seqs:
                                continue

                            key = f"non_grammatical_test_sequences_grammar_edit_{edit_level}_{edit}_{forced_action}"
                            data.setdefault(key, [])
                            data[key] += ge_seqs

                            gkey = f"grammar_edit_{edit_level}_{edit}_{forced_action}"
                            grammar_edit_dict.setdefault(gkey, [])
                            grammar_edit_dict[gkey].append({
                                "base_grammar": grammar,
                                "perturbed_grammar": grammar_perturbed,
                                "perturbation_result": perturbation_result,
                                "sequences": ge_seqs,
                                "non_terminal_applied_position_map": ge_nt_map,
                                "sequence_freq": ge_freq,
                                "sequence_prob_dict": ge_prob,
                                "edit_level": edit_level,
                                "edit": edit,
                                "forced_action": forced_action,
                            })
                            print(f"  grammar_edit level={edit_level} run={run_id}: {len(ge_seqs)} seqs")
                        except Exception:
                            pass

    # ------------------------------------------------------------------ #
    # Edit-distance (lexer) perturbation
    # ------------------------------------------------------------------ #
    edit_distance_perturb_dict = {}
    edit_distance_nt_mapping = {}
    sequence_to_perturbed_map = {}

    for source_name, source_seqs in [("test", test_sequences), ("train", train_sequences)]:
        stat_src = get_stat(source_seqs)
        pmin, pmax = stat_src["min"], stat_src["max"]
        if pmin == pmax:
            pmin = 1
        for edit_distance in [1, 2, 3]:
            perturbed, perturb_pos_dict, perturbed_nt_map, sequence_to_perturbed_map = \
                generate_similar_sequences(
                    grammar, source_seqs, nt_position_map,
                    sequence_to_perturbed_map,
                    edit_distance, seed,
                    sampled_sequences=set(sequence_freq.keys()),
                    perturb_start_index=pmin,
                    perturb_end_index=pmax,
                )
            key = f"non_grammatical_{source_name}_sequences_edit_distance_{edit_distance}_{pmin}_{pmax}"
            data[key] = perturbed
            edit_distance_perturb_dict[key] = perturb_pos_dict
            edit_distance_nt_mapping[key] = perturbed_nt_map
            print(f"  edit_distance source={source_name} d={edit_distance}: {len(perturbed)} seqs")

    meta_data_edit_distance = {
        "non_terminal_applied_position_map": edit_distance_nt_mapping,
        "perturbation_result": edit_distance_perturb_dict,
        "sequence_to_perturbed_sequence_map": sequence_to_perturbed_map,
    }

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    out_dir = store_path / grammar_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  saved → {path}")

    _save(data,
          out_dir / f"sequences_w_edit_distance_{grammar_name}_{train_size + test_size}_{seed}.pkl")
    _save(meta_data,
          out_dir / f"meta_data_{grammar_name}_{num_samples}_{seed}.pkl")
    _save(grammar_meta_data,
          out_dir / f"grammar_meta_data_{grammar_name}.pkl")
    _save(meta_data_edit_distance,
          out_dir / f"meta_data_lexer_edit_{grammar_name}_{num_samples}_{seed}.pkl")
    if grammar_edit_dict:
        _save(grammar_edit_dict,
              out_dir / f"meta_data_grammar_edit_{grammar_name}_{num_samples}_{seed}.pkl")

    print(f"\nDone. Splits: {list(data.keys())}")


if __name__ == "__main__":
    main()
