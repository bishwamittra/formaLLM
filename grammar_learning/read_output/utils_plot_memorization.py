import numpy as np
from plotly import graph_objects as go
import pandas as pd
import plotly.express as px

def get_start_of_memorization(df, 
                              training_dataset, 
                              eval_dataset, 
                              metric,
                              eval_column='eval_dataset',
                              columns_preserve_variance=[],
                              approach="contextual_memorization",
                              memorization_threshold=None,
    ):
    """
        Compute the start of memorization of a given sample in the training dataset
    """
    assert training_dataset in df[eval_column].unique()
    assert eval_column in df.columns
    assert eval_dataset in df[eval_column].unique()
    assert metric in df.columns
    assert approach in ["contextual_memorization", "counterfactual_memorization", "recollection_memorization"]
    if len(columns_preserve_variance) > 0:
        assert all(column in df.columns for column in columns_preserve_variance)
    assert df[df[eval_column] == training_dataset]['token_sequence'].nunique() == 1

    
    df = df[df[eval_column].isin([training_dataset, eval_dataset])].copy()
    df_original = df[[eval_column, 'epoch', metric] + columns_preserve_variance].copy()
    df = df.groupby([eval_column, 'epoch']).aggregate({metric: 'mean'}).reset_index()


    
    epochs = df['epoch'].unique()
    epochs = np.sort(epochs)

    
    
    list_training_recollection = []
    threshold_computed = []
    computed_memorization = []
    epoch_start_of_memorization = None
    training_recollection_at_start_of_memorization = None

    epochs_higher_than_threshold = []
    best_contextual_recollection = None
    if approach == "contextual_memorization":
        if metric == "correct":
            best_contextual_recollection = df[df[eval_column] == eval_dataset].groupby(
                ['epoch']
            ).aggregate({metric: 'max'}).reset_index()[metric].max()
        else:
            best_contextual_recollection = df[df[eval_column] == eval_dataset].groupby(
                ['epoch']
            ).aggregate({metric: 'min'}).reset_index()[metric].min()
    elif approach == "recollection_memorization":
        if memorization_threshold is None:
            if metric == "correct":
                best_contextual_recollection = 0.95
            else:
                best_contextual_recollection = 0.2
        else:
            best_contextual_recollection = memorization_threshold        
    threshold = 0
    # print("Best contextual recollection:", best_contextual_recollection)

    
    
    # compare with threshold
    for epoch in epochs:
        df_item = df[df['epoch'] == epoch]
        if metric == "correct":
            if approach in ["contextual_memorization", "recollection_memorization"]:
                memorization = df_item[df_item[eval_column] == training_dataset][metric].item() - best_contextual_recollection
            else:
                memorization = df_item[df_item[eval_column] == training_dataset][metric].item() - df_item[df_item[eval_column] == eval_dataset][metric].item()
        elif metric == "target_token_negative_log_prob":
            if approach in ["contextual_memorization", "recollection_memorization"]:
                memorization = best_contextual_recollection - df_item[df_item[eval_column] == training_dataset][metric].item()
            else:
                memorization = df_item[df_item[eval_column] == eval_dataset][metric].item() - df_item[df_item[eval_column] == training_dataset][metric].item()

        if memorization > threshold:
            epochs_higher_than_threshold.append(True)
        else:
            epochs_higher_than_threshold.append(False)

        threshold_computed.append(memorization)
        list_training_recollection.append(df_item[df_item[eval_column] == training_dataset][metric].item())


        
    memorization_has_started=True
    epochs_higher_than_threshold = np.array(epochs_higher_than_threshold)
    if not np.all(epochs_higher_than_threshold): # in at least one epoch, memorization is below threshold
        # find the last epoch after which memorization is above threshold
        last_epoch_below_threshold = np.where(epochs_higher_than_threshold == False)[0][-1]
        # print("Last epoch below threshold:", last_epoch_below_threshold)
        if last_epoch_below_threshold < len(epochs) - 1:
            epoch_start_of_memorization = epochs[last_epoch_below_threshold+1]
            training_recollection_at_start_of_memorization = list_training_recollection[last_epoch_below_threshold+1]
        else:
            epoch_start_of_memorization = epochs[-1]
            training_recollection_at_start_of_memorization = list_training_recollection[-1]
            memorization_has_started = False
            
    else:
        epoch_start_of_memorization = epochs[0]
        training_recollection_at_start_of_memorization = list_training_recollection[0]
        
    

    # compute memorization score
    for key, df_item in df_original.groupby(columns_preserve_variance + ['epoch']):
        if df_item[eval_column].nunique() != 2:
            continue

        baseline = None
        if approach in ["contextual_memorization", "recollection_memorization"]:
            baseline = best_contextual_recollection
        else:
            # counterfactual memorization
            baseline = df_item[df_item[eval_column] == eval_dataset][metric].item()

        if metric == "correct":
            if baseline == 1:
                memorization = 0
            elif df_item[df_item[eval_column] == training_dataset][metric].item() < baseline:
                memorization = 0
            else:
                memorization = (df_item[df_item[eval_column] == training_dataset][metric].item() - baseline) / (1 - baseline)

            
        
        elif metric == "target_token_negative_log_prob":
            if baseline == 0:
                memorization = 0
            elif df_item[df_item[eval_column] == training_dataset][metric].item() > baseline:
                memorization = 0
            else:
                memorization = 1 - df_item[df_item[eval_column] == training_dataset][metric].item() / baseline

            

        if approach == "recollection_memorization":
            memorization = 1 if memorization > 0 else 0 # binary

        memorization_results = {
            "memorization": memorization
        }
        for i, column in enumerate(columns_preserve_variance + ['epoch']):
            memorization_results[column] = key[i]

        computed_memorization.append(memorization_results) 

    df_computed_memorization = pd.DataFrame(computed_memorization)
    

    # before the start of memorization, degree of memorization is 0 or nan
    df_computed_memorization['memorization'] = df_computed_memorization.apply(
        lambda x: x['memorization'] if x['epoch'] >= epoch_start_of_memorization else np.nan, axis=1
        # lambda x: x['memorization'] if x['epoch'] >= epoch_start_of_memorization else 0, axis=1
    )
    

    
    
    return df_computed_memorization, epoch_start_of_memorization, best_contextual_recollection, memorization_has_started








def get_arrow_line(fig, x0, y0, x1, y1, color, annotation_text):
    arrowhead = 2
    arrowsize = 1
    arrowwidth = 2

    # Arrow from point A to B
    fig.add_annotation(
        x=x1, y=y1,
        ax=x0, ay=y0,
        xref="x", yref="y",
        axref="x", ayref="y",
        text="",  # No label here
        showarrow=True,
        arrowhead=arrowhead,
        arrowsize=arrowsize,
        arrowwidth=arrowwidth,
        arrowcolor=color
    )

    # Arrow from point B to A (reverse direction)
    fig.add_annotation(
        x=x0, y=y0,
        ax=x1, ay=y1,
        xref="x", yref="y",
        axref="x", ayref="y",
        text="",  # No label here
        showarrow=True,
        arrowhead=arrowhead,
        arrowsize=arrowsize,
        arrowwidth=arrowwidth,
        arrowcolor=color
    )

    # Add center label
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=(y0 + y1) / 2,
        text=annotation_text,
        showarrow=False,
        font=dict(size=10),
        font_color="black",
        bgcolor="white",
        opacity=0.85
    )
    
    return fig


def plot_vline_with_conflict(fig, 
                             v_line_dict, 
                             width=0.5, 
                             line_dash='dot', 
                             font_size=10,
                             annotation_position='top',
    ):
    """
        If multiple vertical lines exist, this code is useful to plot them
    """
    # print(v_line_dict)
    for epoch in v_line_dict:
        memorization_has_started = sum([flag for _, flag in v_line_dict[epoch]])
        if len(v_line_dict[epoch]) == 1:
            
            fig.add_vline(x=epoch, 
                    line_dash=line_dash, 
                    line_color=v_line_dict[epoch][0][0],
                    annotation_text=f"{'> ' if not memorization_has_started else ''}{int(epoch)}",
                    annotation_font=dict(
                        color='gray',
                        size=font_size,
                    ),
                    annotation_position=annotation_position,
                    annotation_yshift=-20 if annotation_position == 'top' else 20,
                    annotation_bgcolor="white",
                    annotation_opacity=0.85,
            )
        else:
            
            fig.add_vline(x=epoch, 
                    line_dash=line_dash, 
                    line_color='rgba(255, 0, 0, 0)',
                    annotation_text=f"{'> ' if not v_line_dict[epoch][0][1] else ''}{int(epoch)}",
                    annotation_font=dict(
                        color='gray',
                        size=font_size,
                    ),
                    annotation_position=annotation_position,
                    annotation_yshift=-20 if annotation_position == 'top' else 20,
                    annotation_bgcolor="white",
                    annotation_opacity=0.85,
            )

            for i, epoch_split in enumerate(np.linspace(epoch - width/2, epoch + width/2, len(v_line_dict[epoch]))):
                fig.add_vline(x=epoch_split, 
                    line_dash=line_dash, 
                    line_color=v_line_dict[epoch][i][0],
                )
    
    return fig




# Descriminative test at the level of individual strings
from nltk.metrics.distance import edit_distance
def get_edit_distance(df, eval_dataset_1, eval_dataset_2):
    assert eval_dataset_1 in df['eval_dataset'].unique()
    assert eval_dataset_2 in df['eval_dataset'].unique()

    token_pair_distance = {}
    eval_1_to_eval_2_distance = {}
    eval_2_seq_to_sample_id = {}
    eval_1_min_distant_eval_2_sample_ids = {}
    for eval_1_token_sequence in df[(df['eval_dataset'] == eval_dataset_1)]['token_sequence'].unique():
        for eval_2_token_sequence in df[(df['eval_dataset'] == eval_dataset_2)]['token_sequence'].unique():
            if eval_2_token_sequence not in eval_2_seq_to_sample_id:
                test_sample_ids = tuple(set(df[(df['eval_dataset'] == eval_dataset_2) & (df['token_sequence'] == eval_2_token_sequence)]['sample_id']))
                eval_2_seq_to_sample_id[eval_2_token_sequence] = test_sample_ids

            # print(eval_1_token_sequence[:10], eval_2_token_sequence[:10])
            if (eval_1_token_sequence, eval_2_token_sequence) not in token_pair_distance:
                # print(edit_distance(eval_1_token_sequence, eval_2_token_sequence))
                token_pair_distance[(eval_1_token_sequence, eval_2_token_sequence)] = edit_distance(eval_1_token_sequence, eval_2_token_sequence)
                # token_pair_distance[(eval_1_token_sequence, eval_2_token_sequence)] = 0

            if eval_1_token_sequence not in eval_1_to_eval_2_distance:
                eval_1_to_eval_2_distance[eval_1_token_sequence] = []
            
            eval_1_to_eval_2_distance[eval_1_token_sequence].append({
                "token_sequence": eval_2_token_sequence,
                "sample_ids": eval_2_seq_to_sample_id[eval_2_token_sequence],
                "distance":  token_pair_distance[(eval_1_token_sequence, eval_2_token_sequence)]
                }
            )

        df_eval_1_sequence_distance = pd.DataFrame(eval_1_to_eval_2_distance[eval_1_token_sequence])
        df_eval_1_sequence_distance = df_eval_1_sequence_distance[df_eval_1_sequence_distance['distance'] == df_eval_1_sequence_distance['distance'].min()] # all rows with minimum distance

        
        optimal_sample_ids = []
        for sample_ids in df_eval_1_sequence_distance['sample_ids'].unique():
            assert len(sample_ids) != 0
            for sample_id in sample_ids:
                optimal_sample_ids.append(sample_id)

        assert df_eval_1_sequence_distance['distance'].nunique() == 1

        optimal_sample_ids = tuple(set(optimal_sample_ids))
        eval_1_min_distant_eval_2_sample_ids[eval_1_token_sequence] = (df_eval_1_sequence_distance['distance'].unique()[0], optimal_sample_ids)
        
    return eval_1_min_distant_eval_2_sample_ids

def compare_with_nearest_test_sequence(df_target, training_dataset="train_sequences", test_dataset="test_sequences", distance_column="language_gen_prob"):
    assert training_dataset in df_target['eval_dataset'].unique()
    assert test_dataset in df_target['eval_dataset'].unique()
    df_target = df_target[df_target['eval_dataset'].isin([training_dataset, test_dataset])].copy()

    sequence_to_gen_prob = {}
    for _, row in df_target.iterrows():
        if row['eval_dataset'] not in sequence_to_gen_prob:
            sequence_to_gen_prob[row['eval_dataset']] = {}

        if row['token_sequence'] not in sequence_to_gen_prob[row['eval_dataset']]:
            sequence_to_gen_prob[row['eval_dataset']][row['token_sequence']] = row[distance_column]
        else:
            assert row[distance_column] == sequence_to_gen_prob[row['eval_dataset']][row['token_sequence']]

    # # distance computation
    token_pair_distance = {}
    train_to_test_distance = {}
    test_token_sequence_to_sample_id = {}
    train_sequence_min_distant_test_sample_ids = {}
    train_sequence_min_distant_test_prob = {}

    for train_token_sequence in df_target[(df_target['eval_dataset'] == training_dataset)]['token_sequence'].unique():
        for test_token_sequence in df_target[(df_target['eval_dataset'] == test_dataset)]['token_sequence'].unique():
            
            # test sequence may be repeated and hence multiple sample ids
            if test_token_sequence not in test_token_sequence_to_sample_id:
                test_sample_ids = tuple(set(df_target[(df_target['eval_dataset'] == test_dataset) & (df_target['token_sequence'] == test_token_sequence)]['sample_id']))
                test_token_sequence_to_sample_id[test_token_sequence] = test_sample_ids

            if (train_token_sequence, test_token_sequence) not in token_pair_distance:
                # token_pair_distance[(train_token_sequence, test_token_sequence)] = edit_distance(train_token_sequence, test_token_sequence)
                token_pair_distance[(train_token_sequence, test_token_sequence)] = abs(
                    sequence_to_gen_prob[training_dataset][train_token_sequence] - sequence_to_gen_prob[test_dataset][test_token_sequence]
                )

            if train_token_sequence not in train_to_test_distance:
                train_to_test_distance[train_token_sequence] = []
            
            train_to_test_distance[train_token_sequence].append({
                "token_sequence": test_token_sequence,
                "sample_ids": test_token_sequence_to_sample_id[test_token_sequence],
                "distance":  token_pair_distance[(train_token_sequence, test_token_sequence)]
                }
            )

        df_train_sequence_distance = pd.DataFrame(train_to_test_distance[train_token_sequence])
        df_train_sequence_distance = df_train_sequence_distance[df_train_sequence_distance['distance'] == df_train_sequence_distance['distance'].min()] # all rows with minimum distance

        # optimal sample ids of test sequences that are minimally distant
        optimal_sample_ids = []
        for sample_ids in df_train_sequence_distance['sample_ids'].unique():
            assert len(sample_ids) != 0
            for sample_id in sample_ids:
                optimal_sample_ids.append(sample_id)

        # unique
        optimal_sample_ids = list(set(optimal_sample_ids))

        optimal_sample_ids = tuple(set(optimal_sample_ids))
        train_sequence_min_distant_test_sample_ids[train_token_sequence] = optimal_sample_ids
        train_sequence_min_distant_test_prob[train_token_sequence] = df_train_sequence_distance['distance'].unique()

    # separate
    df_train_split = df_target[df_target['eval_dataset'] == training_dataset].copy()
    df_test_split = df_target[df_target['eval_dataset'] == test_dataset].copy()
    # df_test_split = df_test_split.set_index(['epoch', 'sample_id'])

    df_train_split['min_distant_test_sample_ids'] = df_train_split['token_sequence'].apply(lambda x: train_sequence_min_distant_test_sample_ids[x])
    df_train_split['min_distant_test_prob'] = df_train_split['token_sequence'].apply(lambda x: train_sequence_min_distant_test_prob[x])
    df_train_split['nearest_test_samples'] = df_train_split['min_distant_test_sample_ids'].apply(lambda x: len(x))


    list_df_test_split = []
    for _, df_item in df_train_split.groupby(['epoch']):
        for _, row in df_item.iterrows():
            df_test_split_subset = df_test_split[df_test_split['sample_id'].isin(row['min_distant_test_sample_ids'])].copy()
            df_test_split_subset['sample_id_training_equiv'] = row['sample_id']
            list_df_test_split.append(df_test_split_subset)
        break

    # print("Loop time", time() - start_time)

    df_test_split_revised = pd.concat(list_df_test_split).groupby(['epoch', 'eval_dataset', 'sample_id_training_equiv']).agg(
            correct=('correct', 'median'), # taking the median value
            correct_mean=('correct', 'mean'),
            correct_std=('correct', 'std'),
            target_token_negative_log_prob=('target_token_negative_log_prob', 'median'), # 
            target_token_negative_log_prob_mean=('target_token_negative_log_prob', 'mean'),
            target_token_negative_log_prob_std=('target_token_negative_log_prob', 'std'),
    ).reset_index()
    # rename back
    df_test_split_revised = df_test_split_revised.rename(columns={'sample_id_training_equiv': 'sample_id'})
    df_test_split_revised['compared_to'] = training_dataset


    df_combined = pd.concat([df_train_split, df_test_split_revised])
    for key, df_item in df_combined.groupby(['epoch', 'sample_id']):
        # print(df_item['eval_dataset'].unique())
        assert df_item['eval_dataset'].nunique() == 2

    return df_combined
