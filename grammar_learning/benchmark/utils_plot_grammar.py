import plotly.graph_objects as go
from nltk.grammar import Nonterminal
    
def color(color, 
          text, 
          underline=False, 
          comment_dict=None
    ):
    if(underline):
        s = f"<span style='text-decoration: underline; color:{color}'>{text}</span>"
    else:      
        s = f"<span style='color:{color}'>{text}</span>"

    if(comment_dict is not None):
        if "font_color" not in comment_dict:
            comment_dict["font_color"] = "#808080"
        if "font_weight" not in comment_dict:
            comment_dict["font_weight"] = 1
        assert "comment" in comment_dict
        if "depth" not in comment_dict:
            comment_dict["depth"] = None

        if comment_dict["depth"] is None:
            # print(comment_dict['comment'])
            s += f"<span style='color:{comment_dict['font_color']}; font-size:{comment_dict['font_weight']}em'>{comment_dict['comment']}</span>"
        else:
            comment_split = comment_dict['comment'].split('<br>')[1:]
            for i in range(len(comment_split)):
                if i < comment_dict['depth']:
                    # use font color
                    s += f"<span style='color:{comment_dict['font_color']}; font-size:{comment_dict['font_weight']}em'><br>{comment_split[i]}</span>"
                else:
                    # use default gray
                    s += f"<span style='color:#808080; font-size:{comment_dict['font_weight']}em'><br>{comment_split[i]}</span>"
                    
    
    return s

def xticks_processor(token_sequence, nonterminal_applied_position_map, is_hierarchy=False):
    print(nonterminal_applied_position_map.keys())
    print(nonterminal_applied_position_map)
    token_sequence_comment = {}
    for non_terminal in nonterminal_applied_position_map.keys():
        position_covered = [False for _ in range(len(token_sequence))]
        for (start_position, end_position, _) in nonterminal_applied_position_map[non_terminal]:
            for position in range(start_position, end_position+1):
                if position < len(token_sequence):
                    position_covered[position] = True
                if(position not in token_sequence_comment):
                    token_sequence_comment[position] = []
                if(position == start_position):
                    if isinstance(non_terminal, tuple):
                        non_terminal_symbol = ''.join([sym.symbol() if isinstance(sym, Nonterminal) else sym for sym in non_terminal])
                    else:
                        non_terminal_symbol = non_terminal.symbol()
                    if "_" in non_terminal_symbol:
                        num_underscores = non_terminal_symbol.count("_")
                        split_elem = non_terminal_symbol.split("_")
                        if num_underscores == 1:
                            non_terminal_symbol = non_terminal_symbol.replace('_', '')
                        elif num_underscores == 2:
                            non_terminal_symbol = f"{split_elem[0]}{split_elem[1]}<sub>{split_elem[2]}</sub>"
                        else:
                            raise ValueError(f"Too many underscores in {non_terminal_symbol}")
                    # print(non_terminal_symbol)
                    token_sequence_comment[position].append(non_terminal_symbol)
                elif position == end_position:
                    token_sequence_comment[position].append(';')
                else:
                    token_sequence_comment[position].append('.')

        if not is_hierarchy:
            for position in range(len(token_sequence)):
                if position_covered[position]:
                    continue
                if(position not in token_sequence_comment):
                    token_sequence_comment[position] = []
                token_sequence_comment[position].append("")

    for position in range(len(token_sequence)):
        if position in token_sequence_comment:
            token_sequence[position] = color('#000000', 
                                            token_sequence[position], 
                                            comment_dict = {
                                                "comment": f"<br>{'<br>'.join(token_sequence_comment[position])}",
                                                "font_weight": 0.7,
                                            }
            )        
                    
        else:
            token_sequence[position] = color('#000000', 
                                            token_sequence[position],
            )

    return token_sequence

def plot_nonterminal_map(token_sequence, nonterminal_applied_position_map, is_hierarchy):
    token_sequence_xticks = xticks_processor(token_sequence, nonterminal_applied_position_map, is_hierarchy=is_hierarchy)
    # scatter = go.Scatter(x=list(range(len(sequences[index]))), y=sequences[index])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(token_sequence))), y=[0]*len(token_sequence), mode='markers'))

    fig.update_layout(
                    width=1000,
                    # height=200 if is_hierarchy else 500,
                    height=200,
                    xaxis = dict(
                        tickvals = list(range(len(token_sequence))),
                        ticktext = token_sequence_xticks,
                        tickangle=0,
                        title=None
                    ),
                    title_font_size=16,
                    yaxis = dict(
                        title=None,
                        showticklabels=False),
            )
    # fig.show()
    return fig


def xticks_processor_with_edit(token_sequence,
        nonterminal_applied_position_map,
        grammar_perturbed,
        perturbation_result,
        edit_level,
        verbose=False
):
    token_sequence_comment = {}
    for non_terminal in nonterminal_applied_position_map.keys():
        for (start_position, end_position, _) in nonterminal_applied_position_map[non_terminal]:
            for position in range(start_position, end_position+1):
                if position not in token_sequence_comment:
                    token_sequence_comment[position] = []
                if position == start_position:
                    token_sequence_comment[position].append(non_terminal.symbol())
                elif position == end_position:
                    token_sequence_comment[position].append(';')
                else:
                    token_sequence_comment[position].append('.')

             
    position_color_red = [False for _ in range(len(token_sequence))]
    # map perturbation to position in the token sequence
    for nonterminal_parent in perturbation_result.keys():
        for rule_index_parent in perturbation_result[nonterminal_parent].keys():
            expansion_rule = grammar_perturbed.productions(nonterminal_parent)[rule_index_parent].rhs()
            for perturb_position, action, old_nonterminal in perturbation_result[nonterminal_parent][rule_index_parent]:
                nonterminal_target = grammar_perturbed.productions(nonterminal_parent)[rule_index_parent].rhs()[perturb_position]


                if isinstance(nonterminal_target, Nonterminal) and nonterminal_target in nonterminal_applied_position_map:
                    for (start_position, end_position, (current_position,nonterminal, rule_index)) in nonterminal_applied_position_map[nonterminal_target]:
                        if nonterminal == nonterminal_parent and rule_index == rule_index_parent and current_position == perturb_position:
                            if verbose:
                                print(f"Perturbed nonterminal: {nonterminal_target} at positions {(start_position, end_position)} | action: {action} | old nonterminal: {old_nonterminal}")
                            for position in range(start_position, end_position+1):
                                position_color_red[position] = True
                                if position == start_position:
                                    if action == "replace":
                                        token_sequence_comment[position].extend(["R", old_nonterminal.symbol()])
                                    elif action == "insert":
                                        token_sequence_comment[position].extend(["I"])
                                    elif action == "delete":
                                        token_sequence_comment[position].extend(["D", old_nonterminal.symbol()])
                                    else:
                                        raise ValueError(f"Invalid action: {action}")
                elif nonterminal_parent in nonterminal_applied_position_map :
                    # track from parent nonterminal in case of perturbation of terminals
                    for (start_position, end_position, (current_position,nonterminal, rule_index)) in nonterminal_applied_position_map[nonterminal_parent]:
                        if tuple(list(token_sequence)[start_position:end_position+1]) == expansion_rule:
                            if verbose:
                                print(f"Perturbed terminal: {nonterminal_target} at position {start_position + perturb_position} | action: {action} | old terminal: {old_nonterminal}")
                            position_color_red[start_position + perturb_position] = True
                            if action == "replace":
                                token_sequence_comment[start_position + perturb_position].extend(["R", old_nonterminal])
                            elif action == "insert":
                                token_sequence_comment[start_position + perturb_position].extend(["I"])
                            elif action == "delete":
                                token_sequence_comment[start_position + perturb_position].extend(["D", old_nonterminal])
                            else:
                                raise ValueError(f"Invalid action: {action}")
                else:
                    pass
                if verbose:
                    print()

    
    for position in range(len(token_sequence)):
        if position in token_sequence_comment:
            token_sequence[position] = color('#000000' if not position_color_red[position] else '#FF0000', 
                                            token_sequence[position], 
                                            comment_dict = {
                                                "comment": f"<br>{'<br>'.join(token_sequence_comment[position])}",
                                                "font_weight": 0.7,
                                                "font_color": '#808080' if not position_color_red[position] else '#FF0000',
                                                "depth": edit_level if position_color_red[position] else None
                                            }
            )
        else:
            token_sequence[position] = color('#000000', 
                                            token_sequence[position],
            )

    return token_sequence



def plot_nonterminal_map_with_edit(
        token_sequence,
        nonterminal_applied_position_map,
        grammar_perturbed,
        perturbation_result,
        edit_level=None,
        verbose=False,
):
    token_sequence_xticks = xticks_processor_with_edit(token_sequence, nonterminal_applied_position_map, grammar_perturbed, perturbation_result, edit_level, verbose)
    # scatter = go.Scatter(x=list(range(len(sequences[index]))), y=sequences[index])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(token_sequence))), y=[0]*len(token_sequence), mode='markers'))

    fig.update_layout(
                    width=1000,
                    height=300,
                    xaxis = dict(
                        tickvals = list(range(len(token_sequence))),
                        ticktext = token_sequence_xticks,
                        tickangle=0,
                        title=None
                    ),
                    title_font_size=16,
                    yaxis = dict(
                        title=None,
                        showticklabels=False),
            )
    # fig.show()
    return fig