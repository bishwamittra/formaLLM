import os
import pickle
import random
import time

from datasets import Dataset, DatasetDict
from transformers import AutoConfig

from utils_args import separator_dict
from utils_encoding import get_tokenizer, tokenize


def process_for_under_trained_tokens(args, tokenizer, dataset, selected_token_ids):
    """
        Derived from https://github.com/cohere-ai/magikarp
    """
    under_trained_token_id_list = {
        "mistralai/Mistral-7B-v0.3": [32506, 21186, 27404, 27175, 27160, 26851, 19527, 10591, 26601, 8376, 28939, 23907, 15824, 18463, 32131, 12961, 17711, 15524, 21460, 11046],
        "EleutherAI/pythia-6.9b": [26868, 28696, 17030, 37402, 41606, 26362, 15479, 30356, 14798, 39743, 15236],
        "Qwen/Qwen2.5-7B": [78783, 79269, 79270, 83969, 83971, 142386, 97000, 136954, 78323, 88372, 142494, 88371, 138175, 122290, 122474, 127734, 151293, 122223, 122578, 117332],
        "/path/to/Llama-2-7b-hf": [28574, 20609, 3798, 12731, 28354, 28633, 31664, 23313, 11193, 12882, 9831],
        "base_models/Llama-2-7b-hf": [28574, 20609, 3798, 12731, 28354, 28633, 31664, 23313, 11193, 12882, 9831],
        "meta-llama/Meta-Llama-3.1-8B": [85071, 107658, 127896, 103003, 126523, 80369, 79883, 106710, 68896, 118508, 89472, 127117, 126647, 124292, 122549, 122746, 64424, 85069, 80370, 125952]
    }

    if args.model_name not in under_trained_token_id_list:
        raise Exception(f"Model {args.model_name} not supported for under trained tokens")

    # mapping
    under_trained_token_id_map = {}
    idx = 0
    for token_id in selected_token_ids:
        if idx >= len(under_trained_token_id_list[args.model_name]) or tokenizer.encode(separator_dict[args.incontext_separator])[0] == token_id:
            under_trained_token_id_map[token_id] = token_id
        else:
            under_trained_token_id_map[token_id] = under_trained_token_id_list[args.model_name][idx]
            idx += 1
    print(under_trained_token_id_map)

    def apply_token_id_map(dataset, token_id_map):
        for token_id in token_id_map:
            dataset["input_ids"][dataset["input_ids"] == token_id] = token_id_map[token_id]
        return dataset

    dataset = dataset.map(apply_token_id_map, fn_kwargs={"token_id_map": under_trained_token_id_map})

    return dataset, selected_token_ids


def prepare_input_for_incontext(data_dict,
                                num_incontext_examples,
                                num_incontext_repetitions=1,
                                separator="semicolon",
                                is_nlp_dataset=False,
                                seed=5):

    separator = separator_dict[separator]

    # From training examples
    incontext_common_prefix = []
    incontext_dataset = None
    assert "train_sequences" in data_dict.keys()
    incontext_dataset = data_dict["train_sequences"]

    for _ in range(num_incontext_repetitions):
        for sequence in incontext_dataset[:num_incontext_examples]:
            if is_nlp_dataset:
                incontext_common_prefix.extend(sequence)
            else:
                incontext_common_prefix.extend(list(sequence))
            incontext_common_prefix.append(separator)

    if is_nlp_dataset:
        incontext_common_prefix = "".join(incontext_common_prefix)

    result = {}
    for key in data_dict.keys():
        result[key] = data_dict[key]

    if len(incontext_common_prefix) > 0:
        if is_nlp_dataset:
            result['incontext_common_prefix'] = [incontext_common_prefix]
        else:
            result['incontext_common_prefix'] = [tuple(incontext_common_prefix)]

    return result, incontext_common_prefix


def get_data(args, verbose=True):
    data_path = "../data"
    if "data_comment" in vars(args) and args.data_comment is not None:
        filename = f"{data_path}/{args.grammar_name}/sequences_w_edit_distance_{args.grammar_name}_{args.num_samples}_{args.data_seed}_{args.data_comment}.pkl"
    else:
        filename = f"{data_path}/{args.grammar_name}/sequences_w_edit_distance_{args.grammar_name}_{args.num_samples}_{args.data_seed}.pkl"

    if os.path.exists(filename):
        print(f"Loading sequences from {filename}")
        with open(filename, 'rb') as f:
            raw_data_dict = pickle.load(f)

            # in training source is different
            if args.incontext_data_source is not None:
                assert ":" in args.incontext_data_source
                [training_data_grammar_name, training_dataset_name] = args.incontext_data_source.split(":")
                train_sequences = pickle.load(open(f"{data_path}/{training_data_grammar_name}/sequences_w_edit_distance_{training_data_grammar_name}_{args.num_samples}_{args.data_seed}.pkl", "rb"))[training_dataset_name]
                print(f"Applying incontext learning from {training_data_grammar_name} with dataset {training_dataset_name}")
                raw_data_dict['train_sequences'] = train_sequences

            if args.run_seed is None:
                args.run_seed = args.data_seed
                # no need to shuffle
            else:
                random.seed(args.run_seed)
                if args.multilingual:
                    assert "_aNd_" in args.grammar_name
                    assert "train_sequences" not in raw_data_dict
                    assert "train_sequences_g1" in raw_data_dict
                    assert "train_sequences_g2" in raw_data_dict

                    if len(raw_data_dict['train_sequences_g1']) != len(raw_data_dict['train_sequences_g2']):
                        random.shuffle(raw_data_dict['train_sequences_g1'])
                        random.shuffle(raw_data_dict['train_sequences_g2'])
                    else:
                        # Shuffling together so that order is preserved
                        zipped_sequences = list(zip(raw_data_dict['train_sequences_g1'], raw_data_dict['train_sequences_g2']))
                        random.shuffle(zipped_sequences)
                        raw_data_dict['train_sequences_g1'], raw_data_dict['train_sequences_g2'] = zip(*zipped_sequences)
                        raw_data_dict['train_sequences_g1'] = list(raw_data_dict['train_sequences_g1'])
                        raw_data_dict['train_sequences_g2'] = list(raw_data_dict['train_sequences_g2'])

                    # find number of training samples
                    if args.multilingual_ratio == 1:
                        considered_training_samples_g1 = args.considered_training_samples
                        considered_training_samples_g2 = args.considered_training_samples

                    elif args.multilingual_ratio > 1:
                        considered_training_samples_g1 = args.considered_training_samples
                        considered_training_samples_g2 = int(args.considered_training_samples * args.multilingual_ratio)

                    else:
                        considered_training_samples_g1 = int(args.considered_training_samples / args.multilingual_ratio)
                        considered_training_samples_g2 = args.considered_training_samples

                    # merge and shuffle
                    raw_data_dict['train_sequences_g1'] = raw_data_dict['train_sequences_g1'][:considered_training_samples_g1]
                    raw_data_dict['train_sequences_g2'] = raw_data_dict['train_sequences_g2'][:considered_training_samples_g2]
                    train_sequences = raw_data_dict['train_sequences_g1'] + raw_data_dict['train_sequences_g2']
                    random.shuffle(train_sequences)

                    raw_data_dict['train_sequences'] = train_sequences
                else:
                    assert 'train_sequences' in raw_data_dict.keys()
                    random.shuffle(raw_data_dict['train_sequences'])

            raw_data_dict['train_sequences'] = raw_data_dict['train_sequences'][args.skip_training_samples:]

    else:
        raise ValueError(f"File {filename} does not exist")

    if (args.considered_training_samples is not None) and (not args.multilingual):
        assert args.considered_training_samples >= 0
        if args.considered_training_samples == 0 and args.incontext_input:
            args.considered_training_samples = 1
        raw_data_dict['train_sequences'] = raw_data_dict['train_sequences'][:args.considered_training_samples]

    # combine edit distance diff position into 1 dataset
    if 'combine_edit_distance' in vars(args) and args.combine_edit_distance:
        modified_data_dict = {}
        delete_keys = []
        for key in raw_data_dict.keys():
            if "edit_distance" in key:
                split = key.split("_")
                if args.multilingual:
                    new_key = f"{'_'.join(split[:-3])}_{split[-1]}"
                else:
                    new_key = "_".join(split[:-2])
                if new_key not in modified_data_dict:
                    modified_data_dict[new_key] = raw_data_dict[key]
                else:
                    modified_data_dict[new_key] += raw_data_dict[key]
                delete_keys.append(key)
        # random sample len(test_sequences) data
        for key in modified_data_dict.keys():
            # shuffle
            random.seed(args.data_seed)
            random.shuffle(modified_data_dict[key])
            if 'test_sequences' in raw_data_dict.keys():
                modified_data_dict[key] = modified_data_dict[key][:len(raw_data_dict['test_sequences'])]
            else:
                modified_data_dict[key] = modified_data_dict[key][:args.considered_eval_samples]
        raw_data_dict.update(modified_data_dict)
        for key in delete_keys:
            del raw_data_dict[key]

    # counterfactual memorization
    if args.counterfactual_memorization:
        print("Counterfactual memorization")
        assert f"counterfactual_{args.counterfactual_string_index}" in raw_data_dict
        considered_counterfactual_string = max(int((args.considered_training_samples * 2 * len(raw_data_dict[f"counterfactual_{args.counterfactual_string_index}"]))/args.num_samples), 1)
        print(f"Initial considered_counterfactual_string:", len(raw_data_dict[f"counterfactual_{args.counterfactual_string_index}"]))
        print(f"considered_counterfactual_string: {considered_counterfactual_string}")
        for counterfactual_string in raw_data_dict[f"counterfactual_{args.counterfactual_string_index}"][:considered_counterfactual_string]:
            raw_data_dict['train_sequences'].append(counterfactual_string)

        random.shuffle(raw_data_dict['train_sequences'])

    # keep one instance of counterfactual instance
    for key in raw_data_dict:
        if key.startswith("counterfactual"):
            raw_data_dict[key] = raw_data_dict[key][:1]

    if args.incontext_input:
        raw_data_dict, incontext_common_prefix = prepare_input_for_incontext(
            raw_data_dict,
            num_incontext_examples=args.considered_incontext_examples,
            num_incontext_repetitions=args.considered_incontext_repetitions,
            separator=args.incontext_separator,
            is_nlp_dataset=args.nlp_dataset,
            seed=args.run_seed
        )

        model_config = AutoConfig.from_pretrained(args.model_name).to_dict()
        max_position_embeddings = model_config['max_position_embeddings'] if 'max_position_embeddings' in model_config else (
            model_config['n_positions'] if 'n_positions' in model_config else None
        )
        if not args.nlp_dataset:
            if len(incontext_common_prefix) > max_position_embeddings:
                print("Error! Incontext input is too long!")
                quit()
        else:
            tokenizer, _ = get_tokenizer(args)
            print("Length of incontext_common_prefix:", len(tokenizer.encode(incontext_common_prefix)))
            if len(tokenizer.encode(incontext_common_prefix)) > max_position_embeddings:
                print("Error! Incontext input is too long!")
                quit()

    # whether to add instruction to solve the task
    if args.add_instruction:
        filename_instruction = f"{data_path}/{args.grammar_name}/instruction_{args.grammar_name}.pkl"
        if os.path.exists(filename_instruction):
            with open(filename_instruction, 'rb') as f:
                instruction = pickle.load(f)
                for key in raw_data_dict.keys():
                    if args.incontext_input and key != "incontext_common_prefix":
                        continue
                    raw_data_dict[key] = [f"{instruction} {s}" for s in raw_data_dict[key]]
        else:
            # this is the case when we consider a formal grammar
            assert not args.nlp_dataset
            if args.instruction is None:
                # use pre-defined instruction
                instruction_list = [
                    "",
                    f"You will be given sequences from a formal language, separated by '{separator_dict[args.incontext_separator]}'. Your task is to generate a new sequence by learning syntactic patterns from the given sequences. ",
                    f"Generate a new sequence by learning syntactic patterns from the given sequences, separated by '{separator_dict[args.incontext_separator]}'. "
                ]
                assert args.instruction_index < len(instruction_list)
                assert args.incontext_input  # it is not clear how to add instruction in FT
                instruction = instruction_list[args.instruction_index]
                args.instruction = instruction

    # apply memorization-based intervention
    if args.memorization_algo == "deduplication":
        print("\n\nDeduplicating training data")

        deduplicating_datasets = ['train_sequences']
        if args.multilingual:
            deduplicating_datasets += ['train_sequences_g1', 'train_sequences_g2']

        for deduplicating_dataset in deduplicating_datasets:
            print("Dataset:", deduplicating_dataset)
            print("Before:", len(raw_data_dict[deduplicating_dataset]))
            deduplicated_train_sequences = {}
            for seq in raw_data_dict[deduplicating_dataset]:
                if seq not in deduplicated_train_sequences:
                    deduplicated_train_sequences[seq] = True
            raw_data_dict[deduplicating_dataset] = list(deduplicated_train_sequences.keys())
            print("After:", len(raw_data_dict[deduplicating_dataset]))

    if args.nlp_dataset and args.grammar_name == "pcfg_berkeley_500k":
        for key in raw_data_dict.keys():
            non_tuple_sequences = []
            for seq in raw_data_dict[key]:
                non_tuple_sequences.append(" ".join(seq))
            raw_data_dict[key] = non_tuple_sequences

    max_seq_len_list = []
    min_seq_len_list = []
    unique_tokens = {}
    data_dict = {}
    for key in raw_data_dict.keys():
        # filter eval_datasets
        if not args.include_edit_distance_eval and "edit_distance" in key:
            if not args.include_edit_distance_1_eval or not key.startswith("non_grammatical_test_sequences_edit_distance_1"):
                continue

        if not args.include_grammar_edit_eval and "grammar_edit" in key:
            continue
        if not args.include_incorrect_random_eval and key in ["non_grammatical_sequences", "non_grammatical_sequences_g1", "non_grammatical_sequences_g2"]:
            continue
        if args.exclude_test_data and "test" in key:
            continue
        if not args.include_grammar_rule_data and "covered_by_rule" in key:
            continue

        if "edit" in key and "train" in key:
            # edits on the training data are not considered at all
            continue

        if key.startswith("train_sequences"):
            data_dict[key] = raw_data_dict[key]
        else:
            data_dict[key] = raw_data_dict[key][:args.considered_eval_samples]

        if len(data_dict[key]) > 0 and key != "incontext_common_prefix":
            max_seq_len_list.append(max(len(s) for s in data_dict[key]))
            min_seq_len_list.append(min(len(s) for s in data_dict[key]))

        for sentence in data_dict[key]:
            for token in sentence:
                if token not in unique_tokens:
                    unique_tokens[token] = 1
                else:
                    unique_tokens[token] += 1

    if args.discriminative_individual:
        meta_data_lexer_edit_filename = f"../data/{args.grammar_name}/meta_data_lexer_edit_{args.grammar_name}_{args.num_samples}_{args.data_seed}.pkl"
        with open(meta_data_lexer_edit_filename, 'rb') as f:
            meta_data_lexer_edit = pickle.load(f)

        # combine
        for key in ['non_terminal_applied_position_map', 'perturbation_result']:
            if key not in meta_data_lexer_edit:
                continue
            temp = {}
            for inner_key in meta_data_lexer_edit[key]:
                for sequence in meta_data_lexer_edit[key][inner_key]:
                    temp[sequence] = meta_data_lexer_edit[key][inner_key][sequence]
            meta_data_lexer_edit[key] = temp

        # gather 5 random incorrect sequences for each test sequences
        assert "test_sequences" in data_dict.keys()
        assert "sequence_to_perturbed_sequence_map" in meta_data_lexer_edit
        incorrect_sequences_combined = []
        visited_seq_dict = {}
        for seq in data_dict["test_sequences"]:
            if seq not in visited_seq_dict:
                visited_seq_dict[seq] = True
                all_incorrect_sequences = meta_data_lexer_edit["sequence_to_perturbed_sequence_map"][seq].copy()
                random.seed(args.data_seed)
                random.shuffle(all_incorrect_sequences)
                incorrect_sequences_combined.extend(
                    all_incorrect_sequences[:10]
                )
        data_dict["non_grammatical_test_sequences_edit_distance_combined"] = incorrect_sequences_combined
        max_seq_len_list.append(max(len(s) for s in incorrect_sequences_combined))
        min_seq_len_list.append(min(len(s) for s in incorrect_sequences_combined))
        print(len(incorrect_sequences_combined), len(set(incorrect_sequences_combined)))
        print(max(len(s) for s in incorrect_sequences_combined), min(len(s) for s in incorrect_sequences_combined))
        print(incorrect_sequences_combined[:3])
        print()

    max_sequence_length = max(max_seq_len_list)

    # if multilingual, merge test sequences too
    if args.multilingual:
        assert "test_sequences" not in data_dict
        assert "test_sequences_g1" in data_dict
        assert "test_sequences_g2" in data_dict
        data_dict["test_sequences"] = data_dict["test_sequences_g1"] + data_dict["test_sequences_g2"]

    if args.map_to_model_specific_tokens:
        assert not args.nlp_dataset
        tokenizer, _ = get_tokenizer(args)
        special_tokens = []
        for k, v in tokenizer.special_tokens_map.items():
            if isinstance(v, str):
                special_tokens.append(v)
            elif isinstance(v, list):
                special_tokens.extend(v)
            else:
                raise ValueError(v)
        special_tokens = set(special_tokens)
        vocabulary = list(range(len(tokenizer)))

        # is there tokens starting with a space?
        token_starting_with_space = 0
        for token_id in vocabulary:
            decoded_token = tokenizer.decode(token_id)
            if decoded_token.startswith(" "):
                token_starting_with_space += 1
        insufficient_good_tokens = False if token_starting_with_space > 10 * len(unique_tokens) else True

        token_map = {}
        token_map_reverse = {}
        start_time = time.time()
        unique_tokens_sorted = sorted(unique_tokens.keys())
        for unique_token in unique_tokens_sorted:
            while True:
                # get a random token_id from the vocabulary
                token_id = random.choice(vocabulary)
                decoded_token = tokenizer.decode(token_id)
                if (token_id not in special_tokens) and (decoded_token != '') and (decoded_token not in token_map_reverse) and (decoded_token.startswith(" ") or insufficient_good_tokens):
                    token_map[unique_token] = decoded_token
                    token_map_reverse[decoded_token] = unique_token
                    break
                if time.time() - start_time > 10:
                    print("Timeout")
                    quit()

        args.token_map_reverse = token_map_reverse

        # apply mapping to data_dict
        for key in data_dict:
            mapped_data = []
            for sentence in data_dict[key]:
                mapped_sentence = []
                for token in sentence:
                    mapped_sentence.append(token_map[token])
                mapped_data.append(tuple(mapped_sentence))
            data_dict[key] = mapped_data

        unique_tokens = token_map_reverse

    if verbose:
        for key in data_dict.keys():
            print(key)
            print(data_dict[key][:5])
            print()

        print("Unique tokens")
        print(unique_tokens)

    # torch compatible
    dataset_dict = {}
    for key in data_dict.keys():
        dataset_dict[key] = Dataset.from_dict({"text": data_dict[key]})

    datasets = DatasetDict(dataset_dict)
    datasets.set_format(type="torch", columns=["text"])

    return datasets, max_sequence_length, list(unique_tokens.keys())
