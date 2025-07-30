import random
import nltk
from tqdm import tqdm


class PCFG(nltk.grammar.PCFG):

    def generate(self, n, seed=None):
        """
        Generates a sequence of sentences using the current probability context-free grammar.

        Parameters:
            n (int): The number of sentences to generate.
            seed (int, optional): The random seed to use. Defaults to None.

        Yields:
            str: The generated sentence.

        """
        random.seed(seed)
        for _ in range(n):
            self.non_terminal_applied_position = {}
            yield self._generate_derivation(self.start(), 0), self.non_terminal_applied_position

    def prefix_conditioned_generate(self, prefix, n, cache=False,  n_max=10000, seed=None, use_tqdm=False):
        random.seed(seed)
        success = 0

        if(cache):
            # check if self._cache is initialized
            if not hasattr(self, '_prefix_cache'):
                self._prefix_cache = {}
            if prefix in self._prefix_cache:
                # print("Yielding from cache")
                for sentence in self._prefix_cache[prefix]:
                    yield (sentence, self._prefix_cache[prefix][sentence])
                    success += 1
                if(success < n):
                    # print("Searching with no seed")
                    random.seed(None)

        
        
        # a naive implementation that tries to generate n sentences with the prefix by checking post generation
        for _ in tqdm(range(n_max), disable=not use_tqdm):
            if success >= n:
                break
            sentence, probability = self._generate_derivation(self.start())
            if(cache):
                for i in range(1, len(sentence) + 1):
                    if sentence[:i] not in self._prefix_cache:
                        self._prefix_cache[sentence[:i]] = {}
                    if(sentence not in self._prefix_cache[sentence[:i]]):    
                        self._prefix_cache[sentence[:i]][sentence] = probability

            if(sentence.startswith(prefix)):
                yield (sentence, probability)
                success += 1
    
    def _reduce_once(self, nonterminal):
        production, probability, index = self._choose_production_reducing(nonterminal)
        return production.rhs(), probability, index

    def _choose_production_reducing(self, nonterminal):
        productions = self._lhs_index[nonterminal]
        probabilities = [production.prob() for production in productions]
        production_indices = list(range(len(productions)))
        random_production_index = random.choices(production_indices, weights=probabilities)[0]
        return productions[random_production_index], probabilities[random_production_index], random_production_index
    

    def _generate_derivation(self, nonterminal, prefix_length):
        sentence = []
        sentence_len = 0
        probability_rhs = 1
        rhs_symbols, probability_lhs, index = self._reduce_once(nonterminal)
        # print(nonterminal, rhs_symbols, index)
        for current_position, symbol in enumerate(rhs_symbols):
            if isinstance(symbol, str):
                derivation = symbol
                sentence.append(derivation)
                sentence_len += 1
            else:
                derivation, probability_non_terminal = self._generate_derivation(symbol, prefix_length + sentence_len)
                probability_rhs *= probability_non_terminal
                sentence.extend(derivation)
                derivation_len = len(derivation)
                sentence_len += derivation_len
                start_position = prefix_length + sentence_len - derivation_len
                end_position = prefix_length + sentence_len - 1
                if(symbol not in self.non_terminal_applied_position):
                    self.non_terminal_applied_position[symbol] = []
                self.non_terminal_applied_position[symbol].append((start_position, end_position, (current_position, nonterminal, index)))                
                # print(symbol, start_position, end_position, sentence)
        probability_sentence = probability_lhs * probability_rhs
        # return "".join(sentence), probability_sentence
        return sentence, probability_sentence