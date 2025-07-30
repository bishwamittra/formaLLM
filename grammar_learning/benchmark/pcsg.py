import random
import nltk
from nltk.grammar import ProbabilisticProduction, Nonterminal


def from_pcsg_string(grammar_string, start_symbol="S"):
    """
        Convert a string representation of a PCFG into a PCFG object.
    """

    productions = []
    nonterminals = set()
    for line in grammar_string.split('\n'):
        line = line.strip()
        if "->" not in line:
            continue



        lhs_string, rhs_prob_string = line.split("->")
        
        # parse lhs
        lhs = []
        lhs_string = lhs_string.strip()
        for elem in lhs_string.split():
            # terminal must start with '
            if elem.startswith("'"):
                lhs.append(elem.replace("'", ""))
            else:
                assert elem[0].isupper(), f"Nonterminal {elem} must start with a capital letter"
                lhs.append(Nonterminal(elem))
                if Nonterminal(elem) not in nonterminals:
                    nonterminals.add(Nonterminal(elem))
        # print(lhs)
        # print([type(elem) for elem in lhs])




        assert "[" in rhs_prob_string, f"Expected '[' in {rhs_prob_string} to denote probability"
        rhs_string, prob_string = rhs_prob_string.strip().split("[")
        
        # parse rhs
        rhs = []
        rhs_string = rhs_string.strip()
        for elem in rhs_string.split():
            # terminal must start with '
            if elem.startswith("'"):
                rhs.append(elem.replace("'", ""))
            else:
                assert elem[0].isupper(), f"Nonterminal {elem} must start with a capital letter"
                rhs.append(Nonterminal(elem))
                if Nonterminal(elem) not in nonterminals:
                    nonterminals.add(Nonterminal(elem))
        # print(rhs)
        # print([type(elem) for elem in rhs])
        

        # parse probability
        assert "]" in prob_string
        prob_string = prob_string.strip()[:-1]
        prob = float(prob_string)
        
        
        # print(rhs_string)
        # print(prob)


        productions.append(
            ProbabilisticProduction(tuple(lhs), tuple(rhs), prob=prob)
        )

    # print(nonterminals)
    assert Nonterminal(start_symbol) in nonterminals, f"Start symbol {start_symbol} not in nonterminals"    
    return PCSG(productions, Nonterminal(start_symbol))



class PCSG:
    def __init__(self, productions, start_symbol):
        """
        Initializes a probabilistic context-sensitive grammar (PCSG).
        
        Args:
            productions (list): List of ProbabilisticProduction rules.
            start_symbol (Nonterminal): The start symbol of the grammar.
        """
        self.productions = productions
        self.start_symbol = start_symbol
        self._lhs_index = self._index_productions()
        self._lexical_index = self._get_terminals()


    def __str__(self):
        
        s = f"PCSG with {len(self.productions)} productions:\n"
        for production in self.productions:
            s += f"{production}" + "\n"
        return s
    

    def _get_terminals(self):
        terminals = {}
        for production in self.productions:
            for elem in production.rhs():
                if isinstance(elem, str):
                    terminals[elem] = True
            for elem in production.lhs():
                if isinstance(elem, str):
                    terminals[elem] = True

        return terminals


    def _index_productions(self):
        """
        Index productions based on their left-hand side (LHS).
        Returns a dictionary mapping LHS symbols to production rules.
        """
        index = {}
        probs = {}
        self.nonterminals = {}
        for production in self.productions:
            assert len(production.lhs()) >= 1
            assert len(production.rhs()) >= len(production.lhs())
            lhs = tuple(production.lhs())  # LHS can be a tuple in CSG
            if lhs not in index:
                index[lhs] = []
            if lhs not in probs:
                probs[lhs] = []
            index[lhs].append(production)
            probs[lhs].append(production.prob())
            for nt in production.lhs():
                if nt not in self.nonterminals:
                    self.nonterminals[nt] = True

        self.nonterminals = set(self.nonterminals.keys())
        for key in probs:
            assert abs(sum(probs[key]) - 1) <= 0.03, f"Probabilities for {key} do not sum to 1: {probs[key]} | {sum(probs[key])}"
        return index

    def generate(self, n, seed=None):
        """
        Generates `n` strings from the PCSG.
        
        Args:
            n (int): Number of sentences to generate.
            seed (int, optional): Random seed.
        
        Yields:
            tuple: Generated sentence and its probability.
        """
        random.seed(seed)
        for _ in range(n):
            self.lhs_applied_position = {}
            yield self._generate_derivation([self.start_symbol]), self.lhs_applied_position
            # yield self._generate_derivation([self.start_symbol])

    def _choose_production_reducing(self, context):
        """
        Selects a production rule that can be applied given the context.
        
        Args:
            context (list): Current sentence state.
        
        Returns:
            tuple: (Production rule, probability, index in context).
        """
        applicable_rules = []
        for i in range(len(context)):
            for length in range(1, len(context) - i + 1):
                segment = tuple(context[i:i+length])
                if segment in self._lhs_index:
                    productions = self._lhs_index[segment]
                    probabilities = [prod.prob() for prod in productions]
                    applicable_rules.append((productions, probabilities, i, length)) # store all information. Sample later

        if not applicable_rules:
            return None  # No valid rule to apply


        # select a random lhs
        applicable_rule = random.choice(applicable_rules)
        productions = applicable_rule[0]
        production_indices = list(range(len(productions)))
        probabilities = applicable_rule[1]
        i = applicable_rule[2]
        length = applicable_rule[3]

        # random select production index
        random_production_index = random.choices(production_indices, weights=probabilities)[0]
        return productions[random_production_index], probabilities[random_production_index], i, length, random_production_index
        # return random.choice(applicable_rules)

    def _generate_derivation(self, sentence):
        """
        Expands the sentence until only terminal symbols remain.
        
        Args:
            sentence (list): Current sentence state.
        
        Returns:
            tuple: (Generated sentence, total probability).
        """
        probability = 1.0
        while any(isinstance(sym, Nonterminal) for sym in sentence):
            # print()
            # print(sentence)
            choice = self._choose_production_reducing(sentence)
            if choice is None:
                # print("No valid expansion")
                break  # No valid expansion, terminate
            

            # print(choice)
            production, prob, index, length, production_rule_index = choice
            probability *= prob
            sentence[index:index+length] = production.rhs()  # Apply production
            if production.lhs() not in self.lhs_applied_position:
                self.lhs_applied_position[production.lhs()] = []
            self.lhs_applied_position[production.lhs()].append((index, 
                                                                index+len(production.rhs()),
                                                                (0, production.lhs(), production_rule_index)))
        
        # return " ".join(sentence), probability  # Convert list to string
        # return sentence, probability
        return tuple(sentence), probability