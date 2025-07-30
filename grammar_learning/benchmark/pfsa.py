import numpy as np
import math

def low_rank_approximation(A, rank):
    """
    Perform SVD and return a low-rank approximation of matrix A.
    
    Parameters:
        A (numpy.ndarray): The original matrix.
        rank (int): The desired rank for the approximation.
    
    Returns:
        numpy.ndarray: The low-rank approximation of A.
    """
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep only the top 'rank' singular values
    S_reduced = np.diag(S[:rank])
    U_reduced = U[:, :rank]
    Vt_reduced = Vt[:rank, :]
    
    # Reconstruct the low-rank approximation
    A_approx = U_reduced @ S_reduced @ Vt_reduced
    
    return A_approx  


def generate_deterministic_transition_matrix(num_states: int, num_symbols: int, seed: int = 5):
    """
    Generates a deterministic transition matrix for a probabilistic finite-state automaton.
    Each state-symbol pair deterministically transitions to exactly one next state.
    The transition matrix is represented as a 2D NumPy array where each row corresponds
    to a state, each column corresponds to an input symbol, and each entry contains
    a probability distribution over the next state (which is always 1 for a single state).
    
    Next compute the rank of the weight matrix and perform SVD on the weight matrix.
    For each k in range(1, rank + 1):
        Compressed Representation (Feature Compression)
        Perform softmax on the compressed representation
        Add the compressed representation to the list_weight_matrix
    
    return list_weight_matrix 
    """
    np.random.seed(seed)
    transition_matrix = np.random.randint(0, num_states, size=(num_states, num_symbols))
    
    # a random weight matrix of the same dimension as the transition matrix. Each entry is a Gaussian random variable with mean 0 and standard deviation 4.
    mu = 0
    sigma = 4
    weight_matrix_original = np.random.normal(mu, sigma, transition_matrix.shape)
    
    
    # compute the rank of weight_matrix
    rank = np.linalg.matrix_rank(weight_matrix_original)

    
    list_weight_matrix = []
    list_rank = []
    for k in range(1, rank + 1):

        weight_matrix = low_rank_approximation(weight_matrix_original, k) # k rank approximation
        assert np.isclose(np.linalg.matrix_rank(weight_matrix), k)
        list_rank.append(np.linalg.matrix_rank(weight_matrix))

        # each row is applied a softmax function to obtain a probability distribution over the next state
        weight_matrix = np.exp(weight_matrix) / np.sum(np.exp(weight_matrix), axis=1, keepdims=True)


        # if probability is less than threshold then set it to 0. Except for the last column
        prob_threshold = 0.1
        weight_matrix[:, :-1][weight_matrix[:, :-1] < prob_threshold] = 0
        # renormalize
        weight_matrix = weight_matrix / np.sum(weight_matrix, axis=1, keepdims=True)

        list_weight_matrix.append(weight_matrix)
    
    return transition_matrix, list_weight_matrix, list_rank


def convert_to_probabilistic_regular_grammar(transition_matrix, 
                                             weight_matrix=None, 
                                             alphabet=None,
                                             last_symbol_epsilon=True):
    """
    Converts a deterministic transition matrix into a probabilistic regular grammar description.

    :return: A string representing the probabilistic regular grammar.
    """
    num_states = transition_matrix.shape[0]
    num_symbols = transition_matrix.shape[1]
    if alphabet is None:
        alphabet = [chr(97 + i) for i in range(num_symbols)]
        if last_symbol_epsilon:
            alphabet = alphabet[:-1]
    else:
        isinstance(alphabet, list)
    if last_symbol_epsilon:
        assert len(alphabet) == num_symbols - 1
    else:
        assert len(alphabet) == num_symbols
    if weight_matrix is None:
        weight_matrix = np.ones((num_states, num_symbols))
    grammar_rules = []

    
    for state in range(num_states):
        for symbol in range(num_symbols):
            rule = ""
            if last_symbol_epsilon and symbol == num_symbols - 1:
                rule = f"S{state-1} -> [{round(weight_matrix[state, symbol], 4)}]"
            else:
                if round(weight_matrix[state, symbol], 4) == 0:
                    continue
                next_state = transition_matrix[state, symbol]
                rule = f"S{state-1} -> '{alphabet[symbol]}' S{next_state-1} [{round(weight_matrix[state, symbol], 4)}]"
        
        
            # special treatment for start symbol
            rule = rule.replace("S-1", "S")
            grammar_rules.append(rule)
    
    return "\n".join(grammar_rules)

def compute_entropy_analytic(transition_matrix, weight_matrix, verbose=False):
    # initial state probability
    initial_state_prob = np.zeros(transition_matrix.shape[0])
    initial_state_prob[0] = 1 # deterministic initial state


    # state to state probability
    state_to_state_prob = np.zeros((transition_matrix.shape[0], transition_matrix.shape[0]))

    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1] - 1):
            state_to_state_prob[i, transition_matrix[i, j]] += weight_matrix[i, j]



    # entropy of transition at each state
    vector_zeta = np.zeros(transition_matrix.shape[0])
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            if weight_matrix[i, j] == 0:
                continue
            vector_zeta[i] += -1 * weight_matrix[i, j] * math.log(weight_matrix[i, j], 2)


    identity_matrix = np.eye(transition_matrix.shape[0])

    entropy = initial_state_prob.T @ np.linalg.pinv(identity_matrix - state_to_state_prob) @ vector_zeta


    if verbose:
        print("Initial State Probability:")
        print(initial_state_prob)
        print()
        print("State to State Probability:")
        print(state_to_state_prob)
        print()
        print("Entropy of Transition at Each State:")
        print(vector_zeta)
        print()
    
    return entropy




def get_random_pfsa(num_states,
                    num_symbols,
                    index_weight_matrix,
                    alphabet=None,
                    last_symbol_epsilon=True,
                    seed=5,
                    verbose=False):


    transition_matrix, list_weight_matrix, list_rank = generate_deterministic_transition_matrix(num_states=num_states, 
                                                                                                num_symbols=num_symbols, 
                                                                                                seed=seed)
    

    if index_weight_matrix >= len(list_weight_matrix):
        return None
    
    grammar = convert_to_probabilistic_regular_grammar(transition_matrix, 
                                                       weight_matrix=list_weight_matrix[index_weight_matrix],
                                                       alphabet=alphabet,
                                                       last_symbol_epsilon=last_symbol_epsilon,
                                                       )


    if verbose:
        print(f"Rank: {list_rank[index_weight_matrix]}")
        print(list_weight_matrix[index_weight_matrix])
        print()
        print(grammar)
        print("\n")
        print(transition_matrix)
        print()

    entropy = compute_entropy_analytic(transition_matrix, 
                                       list_weight_matrix[index_weight_matrix],
                                       verbose=verbose)
    if verbose:
        print(f"Entropy: {entropy}")

    return {
        "grammar_string": grammar,
        "rank": list_rank[index_weight_matrix],
        "entropy_analytic": entropy
    }

# get_random_pfsa(
#     num_states=3,
#     num_symbols=3,
#     index_weight_matrix=2,
#     verbose=True
# )



    
