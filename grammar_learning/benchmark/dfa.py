import time
import numpy as np
from pcfg import PCFG
from pcsg import PCSG

class DFA:
    def __init__(self, states, alphabet, transitions, start_state, final_states):
        """
        Initializes a DFA.
        
        :param states: Set of states
        :param alphabet: Set of input symbols
        :param transitions: Dictionary {state: {symbol: next_state}}
        :param start_state: Initial state
        :param final_states: Set of accepting states
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states

    def __str__(self):
        return f"DFA(states={self.states}, alphabet={self.alphabet}, transitions={self.transitions}, start_state={self.start_state}, final_states={self.final_states})"

    def accepts(self, string):
        """
        Checks if the given string is accepted by the DFA.
        
        :param string: Input string
        :return: True if the string is accepted, False otherwise
        """
        current_state = self.start_state
        cost = 0
        for symbol in string:
            cost += 1
            if symbol not in self.alphabet or current_state not in self.transitions:
                return False, cost  # Reject if symbol not in alphabet or invalid transition
            current_state = self.transitions[current_state].get(symbol, None)
            if current_state is None:
                return False, cost  # Invalid transition
        
        return current_state in self.final_states, cost


def get_dfa(grammar_name, grammar_string, start_state="S"):
# def get_dfa(grammar_row, grammar_string, start_state="S"):
    # grammar_name = grammar_row['grammar_name']
    # grammar_string = grammar_row['grammar_string']

    if grammar_name.startswith("pcfg") or grammar_name.startswith("preg") or grammar_name.startswith("pfsa"):
        grammar = PCFG.fromstring(grammar_string)
    elif grammar_name.startswith("pcsg"):
        grammar = from_pcsg_string(grammar_string)
    else:
        grammar = None
        raise NotImplementedError()

    # print(grammar)
    alphabet = set(grammar._lexical_index.keys())
    final_states = []

    non_terminals_to_productions = {} # excluding empty transitions (final states)
    for production in grammar.productions():
        if production.lhs().symbol() not in non_terminals_to_productions:
            non_terminals_to_productions[production.lhs().symbol()] = {}
        assert len(production.rhs()) <= 2
        if len(production.rhs()) == 0:
            final_states.append(production.lhs().symbol())
        else:
            assert len(production.rhs()) == 2
            if production.rhs()[0] not in non_terminals_to_productions[production.lhs().symbol()]:
                non_terminals_to_productions[production.lhs().symbol()][production.rhs()[0]] = production.rhs()[1].symbol()
            else:
                assert non_terminals_to_productions[production.lhs().symbol()][production.rhs()[0]] == production.rhs()[1].symbol()

    # print(non_terminals_to_productions)
    # print(final_states)

    return DFA(
        states=set(non_terminals_to_productions.keys()),
        alphabet=alphabet,
        transitions=non_terminals_to_productions,
        start_state=start_state,
        final_states=set(final_states)
    )


def membership_dfa_stats(dfa, test_strings, should_accept=True, stat="mean"):
    assert stat in ["median", "mean"]
    
    all_verdict = []
    all_cost = []
    all_time = []
    for test_string in test_strings:
        start_time = time.perf_counter()
        verdict, cost = dfa.accepts(test_string)
        assert verdict == should_accept
        all_verdict.append(verdict)
        all_cost.append(cost)
        end_time = time.perf_counter()
        all_time.append(end_time - start_time)

    
    all_cost = np.array(all_cost)
    all_time = np.array(all_time)

    if stat == "median":
        return np.median(all_cost), np.median(all_time)
    elif stat == "mean":
        return np.mean(all_cost), np.mean(all_time)

