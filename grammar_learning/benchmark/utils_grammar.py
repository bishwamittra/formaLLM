import numpy as np
from pcfg import PCFG
from pcsg import PCSG
from pfsa import get_random_pfsa
from nltk.parse import RecursiveDescentParser
from nltk.parse.chart import BottomUpLeftCornerChartParser
from tqdm import tqdm
from copy import deepcopy
from example_grammar import grammar_details_dict
from pcsg import from_pcsg_string
from nltk.grammar import Nonterminal
import random
from time import time
from example_grammar import grammar_dict, hierarchical_hcfg_double_branch, hierarchical_cfg
import string




def get_grammar_string(grammar_name):
    
    if grammar_name in grammar_dict:
        print("From manually defined grammar")
        return grammar_dict[grammar_name]
    
    
    elif grammar_name.startswith("preg"):
        grammar_name_split = grammar_name.split("_")
        num_terminals = int(grammar_name_split[1])
        num_non_terminals = int(grammar_name_split[2])
        max_breadth = int(grammar_name_split[3])
        min_rule = int(grammar_name_split[4])
        max_rule = int(grammar_name_split[5])
        assert int(grammar_name_split[6]) in [0, 1] 
        assert int(grammar_name_split[7]) in [0, 1]
        deterministic_breadth = int(grammar_name_split[6]) == 1
        is_numerical_terminal = int(grammar_name_split[7]) == 1

        if len(grammar_name_split) > 8:
            assert int(grammar_name_split[8]) in [0, 1]
            no_cycles = int(grammar_name_split[8]) == 1
        else:
            no_cycles = False

        print(f"Grammar name: {grammar_name}")
        print(f"Num terminals: {num_terminals}")
        print(f"Num non terminals: {num_non_terminals}")
        print(f"Max breadth: {max_breadth}")
        print(f"Min rule: {min_rule}")
        print(f"Max rule: {max_rule}")
        print(f"Is deterministic breadth: {deterministic_breadth}")
        print(f"Is numerical terminal: {is_numerical_terminal}")
        print(f"No cycles: {no_cycles}")
        print()

        return generate_regular_grammar(
            num_terminals=num_terminals,
            num_non_terminals=num_non_terminals,
            max_breadth=max_breadth,
            min_rule=min_rule,
            max_rule=max_rule,
            deterministic_breadth=deterministic_breadth,
            is_numerical_terminal=is_numerical_terminal,
            no_cycles=no_cycles
        )


    elif grammar_name.startswith("pfsa"):
        assert grammar_name.startswith("pfsa")
        grammar_name_split = grammar_name.split("_")
        assert grammar_name_split[1] == "states"
        assert grammar_name_split[3] == "symbols"
        assert grammar_name_split[5] == "index"
        assert grammar_name_split[7] == "alphabet"
        num_states = int(grammar_name_split[2])
        num_symbols = int(grammar_name_split[4])
        index_weight_matrix = int(grammar_name_split[6])
        alphabet = list(map(str, grammar_name_split[8].split("-")))

        print(f"Grammar name: {grammar_name}")
        print(f"Num states: {num_states}")
        print(f"Num symbols: {num_symbols}")
        print(f"Index weight matrix: {index_weight_matrix}")
        print(f"Alphabet: {alphabet}")
        print()


        return get_random_pfsa(num_states=num_states,
                                num_symbols=num_symbols,
                                index_weight_matrix=index_weight_matrix,
                                alphabet=alphabet
        )

    elif grammar_name.startswith("pcfg_double-branch"):

        grammar_name_split = grammar_name.split("_")
        assert grammar_name_split[1] == "double-branch"
        assert grammar_name_split[2] == "max-depth"
        assert grammar_name_split[4] == "max-breadth"
        assert grammar_name_split[6] == "rules"
        assert grammar_name_split[8] == "skewness"
        assert grammar_name_split[10] == "alphabet"

 
        depth = grammar_name_split[3]
        breadth = grammar_name_split[5]
        production_per_non_terminal = grammar_name_split[7]
        skewness = grammar_name_split[9]
        terminals = list(map(str, grammar_name_split[11].split("-")))
        return hierarchical_hcfg_double_branch(int(depth), 
                                               int(breadth),
                                               int(production_per_non_terminal),
                                               float(skewness),
                                               terminals=terminals)
    
    

    elif grammar_name.startswith("pcfg"):
        print("Hierarchical CFG")
        grammar_name_split = grammar_name.split("_")
        assert grammar_name_split[1] == "max-depth"
        assert grammar_name_split[3] == "max-breadth"
        assert grammar_name_split[5] == "rules"
        assert grammar_name_split[7] == "skewness"
        assert grammar_name_split[9] == "alphabet"
        

        max_depth = int(grammar_name_split[2])
        max_breadth = int(grammar_name_split[4])
        production_per_non_terminal = int(grammar_name_split[6])
        skewness = float(grammar_name_split[8])
        terminals = list(map(str, grammar_name_split[10].split("-")))
        return hierarchical_cfg(max_depth, 
                                max_breadth, 
                                production_per_non_terminal, 
                                skewness,
                                terminals)

    else:
        raise ValueError(grammar_name)   



def compute_random_guess_metric(grammar, metric="loss"):
    if isinstance(grammar, str):
        if grammar.startswith("pcfg") or grammar.startswith("preg"):
            grammar = PCFG.fromstring(get_grammar_string(grammar))
        elif grammar.startswith("pcsg"):
            grammar = from_pcsg_string(grammar)
        else:
            raise ValueError(grammar)
    print(len(grammar._lexical_index))
    # log of the number of terminals
    if metric == "loss":
        return np.log(len(grammar._lexical_index))
    elif metric == "entropy":
        return np.log(len(grammar._lexical_index))
    elif metric == "accuracy":
        return 1.0/len(grammar._lexical_index) 
    elif metric in ["target_token_prob", "predicted_token_prob"]:
        return 1.0/len(grammar._lexical_index) 
    elif metric == "total_prob_mass":
        return 1.0
    else:
        raise NotImplementedError(f"Metric {metric} is not supported")
    




def generate_regular_grammar(
            num_terminals: int, 
            num_non_terminals: int,
            max_breadth = 1,
            min_rule: int = 1,
            max_rule:int = 2,
            deterministic_breadth = True,
            is_prob = True,
            is_numerical_terminal = True,
            no_cycles = False,
            seed=5,
    ):

    random.seed(seed)


    if num_terminals < 1 or num_non_terminals < 1:
        raise ValueError("The number of terminals and non-terminals must be at least 1.")


    assert min_rule >= 1, "The minimum number of rules must be at least 1."
    assert max_rule >= min_rule, "The maximum number of rules must be at least the minimum number of rules."
    assert max_breadth >= 2, "The breadth must be at least 1."
    
    # terminals = random.sample(string.ascii_lowercase, num_terminals)
    if is_numerical_terminal:
        assert num_terminals < 10
        terminals = list(range(1, 10))[:num_terminals]
    else:
        terminals = string.ascii_lowercase[:num_terminals]
    non_terminals = ["S"] + [f"A{i}" for i in range(num_non_terminals-1)]
    
    start_symbol = non_terminals[0]  # Usually the first non-terminal is the start symbol
    rules = {}

    
    
    for idx_nt, nt in enumerate(non_terminals):
        num_rules = random.randint(min_rule, max_rule)  # Each non-terminal will have 1 or 2 rules
        rules[nt] = []
        if no_cycles:
            if len(non_terminals[idx_nt + 1:]) > 0:
                next_nt_list = iter(random.choices(non_terminals[idx_nt + 1:], k=num_rules))
            else:
                next_nt_list = iter([])
        else:
            next_nt_list = iter(random.choices(non_terminals, k=num_rules))
        for num_rule in range(num_rules):
            if deterministic_breadth:
                breadth = max_breadth - 1
            else:
                breadth = random.randint(1, max_breadth-1)
            terminal_str = ' '.join([f"'{terminal}'" for terminal in random.choices(terminals, k=breadth)])
            
            if random.random() < 0.9 and nt != non_terminals[-1]:  # Ensure last non-terminal has terminal-only rules
                next_nt = next_nt_list.__next__()
                if is_prob:
                    rules[nt].append(f"{terminal_str} {next_nt} [{round(1/num_rules, 4)}]")
                else:
                    rules[nt].append(f"{terminal_str} {next_nt}")
            else:
                if is_prob:
                    rules[nt].append(f"{terminal_str} [{round(1/num_rules, 4)}]")
                else:
                    rules[nt].append(f"{terminal_str}")
    
    return get_regular_grammar_string(start_symbol, rules)

def get_regular_grammar_string(start_symbol, rules):
    # print("Generated Regular Grammar:")
    # print(f"Start Symbol: {start_symbol}")
    s = ""
    for nt, productions in rules.items():
        # print(f"{nt} -> {' | '.join(productions)}")
        for production in productions:
            s += f"{nt} -> {production}" + "\n"

    return s





def check_sequence_exists(grammar_parser, sequence):
    assert isinstance(sequence, list)
    parse_count = 0
    for t in grammar_parser.parse(sequence):
        parse_count += 1
    return parse_count > 0



def random_sentence_generator(grammar, 
                              num_samples, 
                              min_length, 
                              max_length, 
                              seed,
                              sampled_sequences=None, 
                              terminal_freq=None,
                              timeout=100,
                              verbose=False):
    
    membership_query_time = 0
    # grammar_parser = RecursiveDescentParser(grammar)
    grammar_parser = BottomUpLeftCornerChartParser(grammar)
    if(terminal_freq is None):
        terminals = list(grammar._lexical_index.keys())
        prob = np.ones(len(terminals))/len(terminals)
    else:
        terminals = list(terminal_freq.keys())
        prob = np.array([terminal_freq[t] for t in terminals])
        prob = prob/np.sum(prob)
    assert min_length <= max_length
    if(min_length == max_length):
        max_length = min_length + 1

    if isinstance(grammar, PCSG):
        assert sampled_sequences is not None
    
    if(verbose):
        print(f"Terminals: {terminals}")
    
    np.random.seed(seed)
    sequences = []
    found_sequences = 0
    match_with_sampled_sequence = 0
    start_time = time()
    for _ in tqdm(range(num_samples*5)):
        if time() - start_time > timeout:
            print("Time out!")
            break
        sequence = []
        length = np.random.randint(min_length, max_length)
        for _ in range(length):
            sequence.append(np.random.choice(terminals, p=prob))
        start = time()
        if isinstance(grammar, PCSG):
            sequence = tuple(sequence)
            # print(sequence)
            if sequence not in sampled_sequences:
                sequences.append(sequence)
                found_sequences += 1
            else:
                match_with_sampled_sequence += 1
        else:
            if(not check_sequence_exists(grammar_parser, sequence)):
                sequences.append(tuple(sequence))
                found_sequences += 1
        membership_query_time += time() - start
        if(found_sequences == num_samples):
            break

    print(f"Membership query time: {membership_query_time:.2f} s")
    print(f"Match with sampled sequence: {match_with_sampled_sequence}")
    return sequences


def generate_similar_sequences(grammar, 
                               sequences,
                               sentence_to_non_terminal_applied_position_map, 
                               edit_distance, 
                               seed,
                               sampled_sequences=None, 
                               perturb_start_index=0, 
                               perturb_end_index=100000
    ):
    assert perturb_start_index < perturb_end_index
    if isinstance(grammar, PCSG):
        assert sampled_sequences is not None
    

    # grammar parser
    # grammar_parser = RecursiveDescentParser(grammar)
    grammar_parser = BottomUpLeftCornerChartParser(grammar)

    # non-terminal symbols
    terminals = list(grammar._lexical_index.keys())
    terminals_dict_wo = {}
    for terminal in terminals:
        # all terminals except the current terminal
        terminals_dict_wo[terminal] = [t for t in terminals if t != terminal]

    random_positions = []
    np.random.seed(seed)
    new_sequences = []
    perturbation = ['delete', 'insert', 'replace']
    perturb_position_dict = {}
    modified_sentence_to_non_terminal_applied_position_map = {}
    match_with_sampled_sequence = 0
    timeout = 100
    start_time = time()
    for sequence in tqdm(sequences):
        if time() - start_time > timeout:
            print("Time out!")
            break
        perturb_positions = []
        new_non_terminal_map = {non_terminal: value.copy() for (non_terminal, value) in sentence_to_non_terminal_applied_position_map[sequence].items()}
        new_sequence = list(sequence)
        # print()
        # print(new_non_terminal_map)
        for _ in range(edit_distance):
            if(len(new_sequence) == 0):
                break
            if(len(new_sequence) <= min(perturb_start_index, perturb_end_index)):
                break
            action = np.random.choice(perturbation)
            random_position = np.random.randint(low=min(perturb_start_index, len(new_sequence)), 
                                                    high=min(perturb_end_index, len(new_sequence)))
            random_positions.append(random_position)
            if action == 'delete':
                if len(new_sequence) > 0:
                    deleted_terminal = new_sequence[random_position]
                    del new_sequence[random_position]
                    for i in range(len(perturb_positions)):
                        # decrement all positions after the deleted position
                        if perturb_positions[i][0] > random_position:
                            perturb_positions[i] = (perturb_positions[i][0] - 1, perturb_positions[i][1], perturb_positions[i][2])
                    perturb_positions.append((random_position, "delete", deleted_terminal))
                    for non_terminal in new_non_terminal_map.keys():
                        deleted_map_index = []
                        for index, (start_position, end_position, _) in enumerate(new_non_terminal_map[non_terminal]):
                            if start_position == random_position:
                                if end_position > random_position:
                                    new_non_terminal_map[non_terminal][index] = (start_position, end_position - 1, _)
                                else:
                                    deleted_map_index.append(index)
                            elif start_position > random_position:
                                new_non_terminal_map[non_terminal][index] = (start_position - 1, end_position - 1, _)
                            elif end_position >= random_position:
                                new_non_terminal_map[non_terminal][index] = (start_position, end_position - 1, _)
                            else:
                                pass
                        for index in deleted_map_index:
                            del new_non_terminal_map[non_terminal][index]
            elif action == 'insert':
                new_sequence.insert(random_position, np.random.choice(terminals))
                for i in range(len(perturb_positions)):
                    # increment all positions after the inserted position
                    if perturb_positions[i][0] > random_position:
                        perturb_positions[i] = (perturb_positions[i][0] + 1, perturb_positions[i][1], perturb_positions[i][2])
                perturb_positions.append((random_position, "insert", new_sequence[random_position]))
                for non_terminal in new_non_terminal_map.keys():
                    for index, (start_position, end_position, _) in enumerate(new_non_terminal_map[non_terminal]):
                        if start_position == random_position:
                            new_non_terminal_map[non_terminal][index] = (start_position, end_position + 1, _)                        
                        elif start_position > random_position:
                            new_non_terminal_map[non_terminal][index] = (start_position + 1, end_position + 1, _)
                        elif end_position >= random_position:
                            new_non_terminal_map[non_terminal][index] = (start_position, end_position + 1, _)
                        else:
                            pass
            else:
                if len(new_sequence) > 0:
                    perturb_positions.append((random_position, "replace", new_sequence[random_position])) # store earlier value
                    new_sequence[random_position] = np.random.choice(terminals_dict_wo[new_sequence[random_position]])
        
        # drop empty sequences and sequences that are accepted by the grammar
        if(len(new_sequence) > 0 and isinstance(grammar, PCSG)):
            if tuple(new_sequence) not in sampled_sequences:
                new_sequences.append(tuple(new_sequence))
                perturb_position_dict[tuple(new_sequence)] = tuple(perturb_positions)
                modified_sentence_to_non_terminal_applied_position_map[tuple(new_sequence)] = new_non_terminal_map
            else:
                # print("Sequence already sampled", tuple(new_sequence))
                match_with_sampled_sequence += 1

        elif(len(new_sequence) > 0 and not check_sequence_exists(grammar_parser, new_sequence)):
            # print(new_non_terminal_map)
            new_sequences.append(tuple(new_sequence))
            perturb_position_dict[tuple(new_sequence)] = tuple(perturb_positions)
            modified_sentence_to_non_terminal_applied_position_map[tuple(new_sequence)] = new_non_terminal_map
    
    
    temp_sentence_to_non_terminal_applied_position_map = None
    random_positions = np.array(random_positions)
    print(random_positions.mean(), random_positions.std())
    print(f"Match with sampled sequence: {match_with_sampled_sequence}")
    return new_sequences, perturb_position_dict, modified_sentence_to_non_terminal_applied_position_map


def get_subgrammar_string(grammar_string, 
                          target_nonterminal):
    if(target_nonterminal == "S"):
        return None

    subgrammar_string = []
    include_rule = False
    for rule in grammar_string.strip().split("\n"):
        rule = rule.strip()
        if rule.startswith(target_nonterminal) and not include_rule:
        # if target_nonterminal in rule and not include_rule:
            include_rule = True
        if include_rule:
            subgrammar_string.append(rule)
    if(len(subgrammar_string) != 0):
        if subgrammar_string[0].startswith("S ->"):
            #  nonterminal is the start symbol
            return None
        subgrammar_string = [f"S -> {target_nonterminal} [1]"] + subgrammar_string
    return "\n".join(subgrammar_string)


def refine_non_terminal_applied_position(non_terminal_position_map):
    new_non_terminal_position_map = {}
    for non_terminal in non_terminal_position_map.keys():
        len_non_terminal_position_map = len(non_terminal_position_map[non_terminal])
        keep = [True for _ in range(len_non_terminal_position_map)]
        for i in range(len_non_terminal_position_map):
            for j in range(len_non_terminal_position_map):
                if(i == j):
                    continue
                # check containment
                if non_terminal_position_map[non_terminal][i][0] >= non_terminal_position_map[non_terminal][j][0] and \
                   non_terminal_position_map[non_terminal][i][1] <= non_terminal_position_map[non_terminal][j][1]:
                    keep[i] = False
                    break
        # print(non_terminal.symbol())
        # print(non_terminal_position_map[non_terminal])
        # print(keep)
        # print()
        new_non_terminal_position_map[non_terminal] = [non_terminal_position_map[non_terminal][i] for i in range(len_non_terminal_position_map) if keep[i]]
    return new_non_terminal_position_map            

def get_grammatical_sentences(grammar, 
                              num_samples, 
                              seed):
    """
    Generates grammatically correct sentences using a given grammar.

    Args:
        grammar (Grammar): The grammar to use for generating sentences.
        num_samples (int): The number of sentences to generate.
        seed (int): The random seed to use for generating sentences.

    Returns:
        List[Tuple[str]]: A list of tuples, where each tuple represents a sentence.
    """
    sentence_prob_dict = {}
    sentence_freq = {}
    sentence_to_non_terminal_applied_position_map = {}
    sentences = []
    warning_printed = False
    num_samples_effective = 1 * num_samples
    for (sentence, prob), non_terminal_applied_position in tqdm(grammar.generate(num_samples_effective, seed=seed), disable=False):
            # if len(sentence) == 0:
            #     continue
            sentence = tuple(sentence)
            if(sentence not in sentence_prob_dict):
                    sentence_prob_dict[sentence] = prob
                    sentence_to_non_terminal_applied_position_map[sentence] = refine_non_terminal_applied_position(non_terminal_applied_position) 
            else:
                    if not np.isclose(sentence_prob_dict[sentence], prob):
                        # ambiguous grammar. A loose comparison is used here
                        if not warning_printed:
                            print(f"Found sentences with different probabilities! {sentence_prob_dict[sentence]} vs {prob}")
                            warning_printed = True
                        sentence_prob_dict[sentence] += prob

            if(sentence not in sentence_freq):
                    sentence_freq[sentence] = 1
            else:
                    sentence_freq[sentence] += 1
            sentences.append(sentence)
            # print(sentence)
            # print(prob)
            # print(non_terminal_applied_position)
            # print(refine_non_terminal_applied_position(non_terminal_applied_position))
            # print()

    return sentences, sentence_to_non_terminal_applied_position_map, sentence_freq, sentence_prob_dict


def get_nongrammatical_sentences_from_perturbed_grammar(base_grammar,
                                                    perturbed_grammar,                        
                                                    num_samples, 
                                                    seed):
    
    # base_grammar_parser = RecursiveDescentParser(base_grammar)
    base_grammar_parser = BottomUpLeftCornerChartParser(base_grammar)
    sentence_prob_dict = {}
    sentence_freq = {}
    sentence_to_non_terminal_applied_position_map = {}
    sentences = []
    num_samples_effective = 5 * num_samples
    found_samples = 0
    for (sentence, prob), non_terminal_applied_position in tqdm(perturbed_grammar.generate(num_samples_effective, seed=seed), disable=False):
            if check_sequence_exists(base_grammar_parser, sentence):
                continue
            found_samples += 1
            sentence = tuple(sentence)
            if(sentence not in sentence_prob_dict):
                    sentence_prob_dict[sentence] = prob
                    sentence_to_non_terminal_applied_position_map[sentence] = refine_non_terminal_applied_position(non_terminal_applied_position) 
            else:
                    if not np.isclose(sentence_prob_dict[sentence], prob):
                        # ambiguous grammar. A loose comparison is used here
                        print(f"Found sentences with different probabilities! {sentence_prob_dict[sentence]} vs {prob}")
                        sentence_prob_dict[sentence] += prob

            if(sentence not in sentence_freq):
                    sentence_freq[sentence] = 1
            else:
                    sentence_freq[sentence] += 1
            sentences.append(sentence)
            # print(sentence)
            # print(prob)
            # print(non_terminal_applied_position)
            # print(refine_non_terminal_applied_position(non_terminal_applied_position))
            # print()

            if found_samples >= num_samples:
                break

    return sentences, sentence_to_non_terminal_applied_position_map, sentence_freq, sentence_prob_dict


def train_test_split(sequences, 
                     seed,
                     sequence_freq,
                     train_test_ratio=0.8,
                     ):
    test = []
    train = []
    num_test = 0
    num_train = 0
    seen_sequences = {}
    max_num_test = len(sequences) * (1 - train_test_ratio)
    max_num_train = len(sequences) * train_test_ratio
    
    np.random.seed(seed)
    for sequence in sequences:
            if(sequence in seen_sequences):
                continue
            else:
                seen_sequences[sequence] = True
            freq = sequence_freq[sequence]            
            if(bool(random.getrandbits(1))):
                # First try on the training set
                if num_train < max_num_train:
                    for _ in range(freq):
                        train.append(sequence)
                        num_train += 1
                else:
                    for _ in range(freq):
                        test.append(sequence)
                        num_test += 1
            else:
                # First try on the test set
                if num_test < max_num_test:
                    for _ in range(freq):
                        test.append(sequence)
                        num_test += 1
                else:
                    for _ in range(freq):
                        train.append(sequence)
                        num_train += 1

    
    np.random.shuffle(train)
    np.random.shuffle(test)

    print(f"Train: {len(train)}")
    print(f"Test: {len(test)}")

    return train, test

def bucket_sequences_by_length(sequences, num_buckets):
    len_sequences = [len(sequence) for sequence in sequences]
    min_len = min(len_sequences)
    max_len = max(len_sequences) + 1
    sequences_bucket = [[] for _ in range(num_buckets)]
    bucket_stat = [(max_len+1, min_len-1) for _ in range(num_buckets)] # list of (min_len, max_len) within each bucket
    for i, sequence in enumerate(sequences):
        len_sequence = len(sequence)
        bucket_idx = int((len_sequence - min_len) / (max_len - min_len) * num_buckets)
        
        sequences_bucket[bucket_idx].append(sequence)
        if(len_sequence <= bucket_stat[bucket_idx][0]):
            bucket_stat[bucket_idx] = (len_sequence, bucket_stat[bucket_idx][1])
        if(len_sequence >= bucket_stat[bucket_idx][1]):
            bucket_stat[bucket_idx] = (bucket_stat[bucket_idx][0], len_sequence)
    return sequences_bucket, bucket_stat


def compute_terminal_freq(sequences):
    terminal_freq = {}
    max_freq = 0
    for sequence in sequences:
        for terminal in sequence:
            if terminal not in terminal_freq:
                terminal_freq[terminal] = 1
            else:
                terminal_freq[terminal] += 1
            max_freq += 1
    terminal_freq = {k: v / max_freq for k, v in terminal_freq.items()}
    return terminal_freq



"""
Grammar edit
"""

def get_grammar_variables(grammar, grammar_name):
    nonterminal_to_level = {}
    level_to_nonterminals = {}
    terminals = list(grammar._lexical_index.keys())
    nonterminals = list([production.lhs() for production in grammar.productions()])
    
    if(grammar_name in grammar_details_dict):
        level_to_nonterminals = grammar_details_dict[grammar_name]["level_to_nonterminals"]
        nonterminal_to_level = grammar_details_dict[grammar_name]["nonterminal_to_level"]
    else:
        for nonterminal in nonterminals:
            if "_" in nonterminal.symbol():
                level = int(nonterminal.symbol().split("_")[1])
                nonterminal_to_level[nonterminal] = level
                if level not in level_to_nonterminals:
                    level_to_nonterminals[level] = []
                level_to_nonterminals[level].append(nonterminal)
            else:
                nonterminal_to_level[nonterminal] = -100
    # assert 0 not in level_to_nonterminals
    level_to_nonterminals[0] = terminals
    return terminals, nonterminals, nonterminal_to_level, level_to_nonterminals

def get_perturbed_grammar_string(grammar_working, perturbation_result):
    grammar_string_modified = []
    for production in grammar_working.productions():
        if(production.lhs() == perturbation_result["nonterminal"]):
            continue
        else:
            grammar_string_modified.append(" ".join([
                production.lhs().symbol(),
                "->",
                " ".join([elem.symbol() if isinstance(elem, Nonterminal) else f"'{elem}'" for elem in production.rhs()]),
                f"[{production.prob()}]"
            ]))
    for i, production in enumerate(grammar_working.productions(perturbation_result["nonterminal"])):
        if i == perturbation_result["expansion_rule_index"]:
            grammar_string_modified.append(" ".join([
                production.lhs().symbol(),
                "->",
                " ".join([elem.symbol() if isinstance(elem, Nonterminal) else f"'{elem}'" for elem in perturbation_result['expansion_rule']]),
                f"[{production.prob()}]"
            ]))
        else:
            grammar_string_modified.append(" ".join([
                production.lhs().symbol(),
                "->",
                " ".join([elem.symbol() if isinstance(elem, Nonterminal) else f"'{elem}'" for elem in production.rhs()]),
                f"[{production.prob()}]"
            ]))

    return PCFG.fromstring("\n".join(grammar_string_modified))

def get_perturbed_grammar(grammar, 
                          grammar_name, 
                          level=1, 
                          edit=1, 
                          seed=0,
                          forced_action=None, 
                          verbose=False):
    assert forced_action in ["replace", "insert", "delete", None]
    assert level != 0
    terminals, nonterminals, nonterminal_to_level, level_to_nonterminals = get_grammar_variables(grammar, grammar_name)
    if verbose:
        print(f"Terminals: {terminals}")
        print(f"Nonterminals: {nonterminals}")
        print(f"Nonterminal to level: {nonterminal_to_level}")
        print(f"Level to nonterminals: {level_to_nonterminals}")
    assert level in level_to_nonterminals
    assert edit > 0

    # print(seed)
    np.random.seed(seed)
    grammar_working = deepcopy(grammar)
    result = {}
    for _ in range(edit):
        applied_nonterminal = np.random.choice(level_to_nonterminals[level])
        choice_nonterminals = level_to_nonterminals[level-1]
        nonterminals_dict_wo = {}
        for nonterminal in choice_nonterminals:
            # all nonterminals except the current terminal
            nonterminals_dict_wo[nonterminal] = [t for t in choice_nonterminals if t != nonterminal]

        expansion_rule_index = np.random.choice(range(len(grammar_working.productions(applied_nonterminal))))
        expansion_rule = list(grammar_working.productions(applied_nonterminal)[expansion_rule_index].rhs())

        # print(f"Random nonterminal: {applied_nonterminal}")
        # print(f"Choices: {choice_nonterminals}")
        # print(f"Expansion rules: {expansion_rule_index}")   
        # print(nonterminals_dict_wo)

        if verbose:
            print(f"Expansion rule before: {expansion_rule}. Index {expansion_rule_index}")
        if len(expansion_rule) == 1:
            random_position = 0
        else:
            random_position = np.random.randint(0, len(expansion_rule)-1)
        if(forced_action is None):
            if(len(expansion_rule) > 1):
                action = np.random.choice(["replace", 'insert', "delete"])
            else:
                action = np.random.choice(['insert', "replace"])
        else:
            action = forced_action
        old_nonterminal = expansion_rule[random_position] if action != 'insert' else None
        if(action == "insert"):
            expansion_rule.insert(
                random_position, np.random.choice(choice_nonterminals)
            )
        elif(action == "delete"):
            if(len(expansion_rule) > 1):
                expansion_rule.pop(random_position)
            else:
                print("Cannot delete last element")
                continue
        elif(action == "replace"):
            replaced_nonterminal = np.random.choice(nonterminals_dict_wo[expansion_rule[random_position]])
            expansion_rule[random_position] = replaced_nonterminal
        # print(f"Expansion rule after: {expansion_rule}")
        if verbose:
            print(f"action: {action} at position {random_position}")
        

        perturbation_result = {
            "nonterminal": applied_nonterminal,
            "expansion_rule_index": expansion_rule_index,
            "action": action,
            "position": random_position,
            "expansion_rule": expansion_rule
        }


        # update
        grammar_working = get_perturbed_grammar_string(grammar_working, perturbation_result)

        # revise expansion rule index, which may change due to new string import
        at_least_one = False
        for i, production in enumerate(grammar_working.productions(applied_nonterminal)):
            if tuple(expansion_rule) == production.rhs():
                expansion_rule_index = i
                at_least_one = True
                break
        assert at_least_one
        if verbose:
            print("Updated expansion rule index: ", expansion_rule_index)
        perturbation_result["expansion_rule_index"] = expansion_rule_index

        if(applied_nonterminal not in result):
            result[applied_nonterminal] = {}
        if(expansion_rule_index not in result[applied_nonterminal]):
            result[applied_nonterminal][expansion_rule_index] = []

        # update position of previous perturbations in case of insert and delete
        for i in range(len(result[applied_nonterminal][expansion_rule_index])):
            if action == "insert" and result[applied_nonterminal][expansion_rule_index][i][0] >= random_position:
                result[applied_nonterminal][expansion_rule_index][i] = (max(result[applied_nonterminal][expansion_rule_index][i][0]+1, len(expansion_rule)-1),
                                                                    result[applied_nonterminal][expansion_rule_index][i][1],
                                                                    result[applied_nonterminal][expansion_rule_index][i][2])
                if verbose:
                    print(f"update next perturbation position: {result[applied_nonterminal][expansion_rule_index][i][0]} => {result[applied_nonterminal][expansion_rule_index][i][0]+1}")
            elif action == "delete" and result[applied_nonterminal][expansion_rule_index][i][0] >= random_position:
                result[applied_nonterminal][expansion_rule_index][i] = (min(result[applied_nonterminal][expansion_rule_index][i][0]-1, 0),
                                                                    result[applied_nonterminal][expansion_rule_index][i][1],
                                                                    result[applied_nonterminal][expansion_rule_index][i][2])
                if verbose:
                    print(f"update next perturbation position: {result[applied_nonterminal][expansion_rule_index][i][0]} => {result[applied_nonterminal][expansion_rule_index][i][0]-1}")

        if action == "delete" and random_position >= len(expansion_rule):
            result[applied_nonterminal][expansion_rule_index].append((len(expansion_rule)-1, action, old_nonterminal))
        else:      
            result[applied_nonterminal][expansion_rule_index].append((random_position, action, old_nonterminal))

        # print(grammar_working)
        if verbose:
            print(perturbation_result)
            print()

    # print(result)
    # print(grammar_working)
    return grammar_working, result



def to_latex_equation(grammar_string, color_dict=None, script_notation="_"):
    latex_string = []
    latex_string.append("\\begin{align*}")
    for i, line in enumerate(grammar_string.split("\n")):
        line = line.strip()
        if line == "":
            continue
        # preprocess non-terminals
        line_modified = []
        for elem in line.split(" "):
            if "_" in elem:
                num_underscores = elem.count("_")
                split_elem = elem.split("_")
                if num_underscores == 1:
                    line_modified.append(f"{elem.replace('_', '')}")
                elif num_underscores == 2:
                    line_modified.append(f"{split_elem[0]}{split_elem[1]}_{split_elem[2]}")
                else:
                    raise ValueError(f"Too many underscores in {elem}")
            else:
                line_modified.append(elem)

            start_parenthesis = "{"
            end_parenthesis = "}"
            default_color = "\\textcolor{teal}"
            if color_dict is not None:  
                if elem[0] in color_dict:
                    line_modified[-1] = f"{color_dict[elem[0]]}{start_parenthesis}{line_modified[-1]}{end_parenthesis}"
                else:
                    line_modified[-1] = f"{default_color}{start_parenthesis}{line_modified[-1]}{end_parenthesis}"
                    


        # print(line_modified)
        line = " ".join(line_modified)

        rightarrow = "\\rightarrow"
        newline = "\\\\"
        appostrophe = "'"
        # bracket_open = " \; ["
        space = "\;"
        latex_string.append(f"\t& {line.replace(' ', space).replace('->', rightarrow).replace('_', script_notation).replace(appostrophe, '')}{newline}")
    latex_string.append("\\end{align*}")
    
    print("\n".join(latex_string))
