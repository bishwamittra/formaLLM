"""
Convert formal-grammar pickle datasets to HuggingFace-compatible Parquet format.

Data sources:
  sequences_w_edit_distance_*.pkl
    Dict keyed by split name → list of token tuples, e.g. ('6','5','4',...).
    Splits: train_sequences, test_sequences, non_grammatical_sequences,
            non_grammatical_*_grammar_edit_*, non_grammatical_*_edit_distance_*

  meta_data_*.pkl  (train/test only)
    Dict with keys:
      non_terminal_applied_position_map: {sequence_tuple → {Nonterminal → [(start, end, (rule_idx, parent_NT, child_idx)), ...]}}
      sequence_freq:      {sequence_tuple → int}
      sequence_prob_dict: {sequence_tuple → float}

Output schema — train_sequences and test_sequences:
  grammar_name             : str
  split                    : str
  sequence                 : str   space-joined tokens
  non_terminal_positions   : str   JSON  {nonterminal: [[start, end, [rule_idx, parent, child_idx]], ...]}
  sequence_freq            : int   how often this sequence appears in training data
  sequence_prob            : float probability under the grammar

All other splits (non-grammatical, edit-distance, grammar-edit):
  grammar_name             : str
  split                    : str
  sequence                 : str
  non_terminal_positions   : null
  sequence_freq            : null
  sequence_prob            : null

Grammar rules are written to grammar_learning/data/<ds>_hf/grammar_meta.json.

Run from the repo root:
    python hf_upload/convert_to_parquet.py
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grammar_learning" / "benchmark"))

import pandas as pd

DATA_ROOT = Path("grammar_learning/data")

SPLITS_WITH_METADATA = {"train_sequences", "test_sequences"}


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def seq_to_str(seq) -> str:
    if isinstance(seq, str):
        return seq
    return " ".join(str(t) for t in seq)


def serialize_nt_positions(pos_map) -> str:
    """
    Serialize all parses to a JSON list of dicts.
    Each dict maps nonterminal string → [[start, end, [rule_idx, parent_NT, child_idx]], ...].
    context may be None for the root S node.
    Accepts both the new list-of-dicts format and the old single-dict format.
    """
    if isinstance(pos_map, dict):
        pos_map = [pos_map]
    result = []
    for parse in pos_map:
        out = {}
        for nt, spans in parse.items():
            serialized_spans = []
            for span in spans:
                start, end, context = span
                if context is None:
                    serialized_spans.append([start, end, None])
                else:
                    rule_idx, parent_nt, child_idx = context
                    serialized_spans.append([start, end, [rule_idx, str(parent_nt), child_idx]])
            out[str(nt)] = serialized_spans
        result.append(out)
    return json.dumps(result)


def convert_dataset(ds_name: str):
    ds_dir = DATA_ROOT / ds_name
    if not ds_dir.exists():
        print(f"  skip — not found: {ds_dir}")
        return

    # --- load all sequences ---
    seq_files = sorted(ds_dir.glob("sequences_w_edit_distance_*.pkl"))
    if not seq_files:
        print(f"  skip — no sequences_w_edit_distance file in {ds_dir}")
        return
    data = load_pkl(seq_files[0])

    # --- load train/test metadata ---
    meta_lookup = {}   # sequence_tuple → {freq, prob, nt_positions_json}
    for meta_file in ds_dir.glob("meta_data_*.pkl"):
        if "grammar_edit" in meta_file.name or "lexer_edit" in meta_file.name:
            continue
        meta = load_pkl(meta_file)
        nt_map   = meta.get("non_terminal_applied_position_map", {})
        freq_map = meta.get("sequence_freq", {})
        prob_map = meta.get("sequence_prob_dict", {})
        for seq_tuple, pos_map in nt_map.items():
            meta_lookup[seq_tuple] = {
                "non_terminal_positions": serialize_nt_positions(pos_map),
                "sequence_freq": freq_map.get(seq_tuple),
                "sequence_prob": prob_map.get(seq_tuple),
            }
        break  # only one meta_data file expected

    # --- one parquet file per split ---
    out_dir = DATA_ROOT / f"{ds_name}_hf"
    out_dir.mkdir(exist_ok=True)

    for split_name, seqs in data.items():
        if not isinstance(seqs, (list, tuple)):
            continue
        include_meta = split_name in SPLITS_WITH_METADATA
        rows = []
        for seq in seqs:
            row = {
                "sequence": seq_to_str(seq),
                "non_terminal_positions": None,
                "sequence_freq": None,
                "sequence_prob": None,
            }
            if include_meta and seq in meta_lookup:
                row.update(meta_lookup[seq])
            rows.append(row)

        split_dir = out_dir / split_name
        split_dir.mkdir(exist_ok=True)
        out_path = split_dir / "data.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"    {split_name}: {len(rows)} sequences"
              + (" [+metadata]" if include_meta else "")
              + f"  → {out_path}")

    # --- grammar rules sidecar ---
    for gmeta_file in ds_dir.glob("grammar_meta_data_*.pkl"):
        gdata = load_pkl(gmeta_file)
        info = {}
        if isinstance(gdata, dict):
            for k, v in gdata.items():
                info[k] = str(v)
        else:
            info["grammar"] = str(gdata)
        sidecar = out_dir / "grammar_meta.json"
        sidecar.write_text(json.dumps(info, indent=2))
        print(f"  → {sidecar}")


def main():
    # discover all grammar directories (those containing sequences_w_edit_distance_*.pkl)
    datasets = sorted(
        d.name
        for d in DATA_ROOT.iterdir()
        if d.is_dir()
        and not d.name.endswith("_hf")
        and any(d.glob("sequences_w_edit_distance_*.pkl"))
    )
    if not datasets:
        print("No grammar datasets found in grammar_learning/data/")
        return
    print(f"Found {len(datasets)} grammar(s): {datasets}\n")
    for ds_name in datasets:
        print(ds_name)
        convert_dataset(ds_name)
    print("\nDone. Run upload_to_hf.py next.")


if __name__ == "__main__":
    main()
