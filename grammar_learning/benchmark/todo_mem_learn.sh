cd grammar_learning/benchmark
for g in \
  pcfg_4_3_1_2_3_4_5_6_7_8_9 \
  pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob \
  pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_uniform_prob \
  pcfg_4_3_1_2_3_4_5_6_7_8_9_latin \
  pcfg_cfg3b_disjoint_terminals_latin \
  pcfg_cfg3b_eq_len_skewed_prob \
  pcfg_cfg3b_eq_len_uniform_prob \
  "pcfg_max-depth_4_max-breadth_4_rules_2_skewness_1_alphabet_0-1-2-3-4"; do
    python generate_dataset.py --grammar_name "$g"
done