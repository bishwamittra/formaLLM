import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import random
from copy import deepcopy
from itertools import cycle

symbol_sequence = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right', 'triangle-ne', 'triangle-se', 'triangle-sw', 'triangle-nw', 'pentagon', 'hexagon', 'hexagon2', 'octagon', 'star', 'hexagram', 'star-triangle-up', 'star-triangle-down', 'star-square', 'star-diamond', 'diamond-tall', 'diamond-wide', 'hourglass', 'bowtie', 'circle-cross', 'circle-x', 'square-cross', 'square-x', 'diamond-cross', 'diamond-x', 'cross-thin', 'x-thin', 'asterisk', 'hash', 'y-up', 'y-down', 'y-left', 'y-right', 'line-ew', 'line-ns', 'line-ne', 'line-nw', 'arrow-up', 'arrow-down', 'arrow-left', 'arrow-right', 'arrow-bar-up', 'arrow-bar-down', 'arrow-bar-left', 'arrow-bar-right']

def print_summary(df, list_group, list_aggregate, show_std=True, latex_format=False, round_to_digits=True):
    df_summary = mean_std_df(df, list_group, list_aggregate)
    for column in list_aggregate:
        if round_to_digits:
            df_summary[column+'_mean'] = df_summary[column+'_mean'].apply(lambda x: f"{x:.2f}")
            df_summary[column+'_std'] = df_summary[column+'_std'].apply(lambda x: f"{x:.2f}")


        # check nan
        if(show_std):
            if(latex_format):
                df_summary[column] = df_summary.apply(lambda x: f"\\textemdash" if not pd.notna(x[column+'_mean'])
                                    else f"$ {x[column+'_mean']} \pm {x[column+'_std']} $", axis=1)
            else:
                df_summary[column] = df_summary.apply(lambda x: f"-" if not pd.notna(x[column+'_mean'])
                                    else f"{x[column+'_mean']} ({x[column+'_std']})", axis=1)
        else:
            if(latex_format):
                df_summary[column] = df_summary.apply(lambda x: f"\\textemdash" if not pd.notna(x[column+'_mean'])
                                    else f"$ {x[column+'_mean']} $", axis=1)
            else:
                df_summary[column] = df_summary.apply(lambda x: f"-" if not pd.notna(x[column+'_mean'])
                                    else f"{x[column+'_mean']}", axis=1)

    return df_summary[list_group + list_aggregate]


def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands', None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode is None:
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
    elif error_y_mode in {'bar','bars'}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + np.nan_to_num(data['error_y']['array']))
            y_lower = list(data['y'] - np.nan_to_num(data['error_y']['array']) if data['error_y']['arrayminus'] is None else data['y'] - np.nan_to_num(data['error_y']['arrayminus']))
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.2)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig

import scipy.stats
def mean_std_df(df, group_columns, columns_to_agg):
    xdf = df.groupby(group_columns).agg({column : ['mean', 'std'] for column in columns_to_agg})
    xdf.columns = xdf.columns.map("_".join)
    return xdf.reset_index()

    
def nice_plot_multi_columns(
        df,
        x_axis,
        y_axis_list,
        plot_type='line',
        x_axis_title=None,
        y_axis_title=None,
        legend_title=None,
        legend_names=None,
        group_by=None,
        group_names=None,
        group_title=None,
        separator=" | ",
        error_y_mode='band',
        enforce_legend_order=False,
        render_mode="auto",
        add_marker=False,
    ):
    df = df.copy()
    _orig_legend_title = legend_title
    assert x_axis in df.columns
    if(x_axis_title is None):
        x_axis_title = x_axis
       
    if(legend_names is None):
        legend_names = dict(zip(y_axis_list, map(str, y_axis_list)))
    else:
        assert isinstance(legend_names, dict)
        for y_axis in y_axis_list:
            if y_axis not in legend_names.keys():
                legend_names[y_axis] = y_axis
    if(legend_title is None):
        legend_title = f"dummy_column_{random.randint(0, 1000000)}"

    for y_axis in y_axis_list:
        assert y_axis in df.columns
        assert y_axis in legend_names.keys()
    assert len(y_axis_list) <= len(legend_names)

    if group_by is not None:
        # preprocessing
        assert group_by in df.columns
        if(group_names is not None):
            assert isinstance(group_names, dict)
            for group in df[group_by].unique():
                if group not in group_names.keys():
                    group_names[group] = group
        else:
            group_names = dict(zip(df[group_by].unique(), map(str, df[group_by].unique())))


        if(group_title is not None):
            if(_orig_legend_title is None):
                _orig_legend_title = group_title
            else:
                _orig_legend_title = group_title + separator + _orig_legend_title
        

    
        legend_title = f"{group_by}{separator}{legend_title}"

        modifed_y_axis_list = []
        for group in df[group_by].unique():
            for column in y_axis_list:
                df[f"{group_names[group]}{separator}{legend_names[column]}"] = df[column].where(df[group_by] == group, np.nan)
                modifed_y_axis_list.append(f"{group_names[group]}{separator}{legend_names[column]}")    

        df_melt = df.melt(id_vars=[x_axis], 
                        value_vars=modifed_y_axis_list, 
                        value_name='value',
                        var_name=legend_title)

    else:
        df_melt = df.melt(id_vars=[x_axis], 
                        value_vars=y_axis_list, 
                        value_name='value',
                        var_name=legend_title)
        
        df_melt[legend_title] = df_melt[legend_title].map(legend_names)

    df_mean = mean_std_df(df_melt, [x_axis, legend_title], ['value'])

    category_orders={
                    legend_title: [legend_names[y_axis] for y_axis in y_axis_list]
    } if enforce_legend_order else None
    if(plot_type == 'line'):
        fig = line(data_frame = df_mean,
                x=x_axis,
                y='value_mean',
                error_y='value_std',
                color=legend_title,
                symbol=legend_title if add_marker else None,
                symbol_sequence=symbol_sequence if add_marker else None,
                category_orders=category_orders,
                error_y_mode=error_y_mode,
                render_mode=render_mode
        )
    elif(plot_type == 'bar'):
        fig = px.bar(df_mean, 
                        x=x_axis, 
                        y='value_mean', 
                        color=legend_title,
                        category_orders=category_orders, 
                        error_y='value_std' if error_y_mode is not None else None)
        fig.update_layout(barmode='group')


    fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            legend_title=_orig_legend_title,
            font=dict(
                # family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
            ),
            width=800,
            height=400,
            # hovermode="x"
        )
    return fig


def nice_plot(
        df,
        x_axis,
        y_axis,
        y_error=None,
        plot_type='line',
        x_axis_title=None,
        y_axis_title=None,
        group_by=None,
        group_names=None,
        legend_title=None,
        error_y_mode='band',
        enforce_legend_order=None,
        render_mode="auto",
        add_marker=False,
        color_group=None,
        color_starting_index=0,
        group_hide_legend=None,
        group_dashed_line=None,
        dash_type="dot"
    ):
    df = df.copy()
    group_names = deepcopy(group_names)
    assert y_axis in df.columns
    assert error_y_mode in ['band', 'bands', 'bar', 'bars', None]
    if(x_axis_title is None):
        x_axis_title = x_axis
    if(y_axis_title is None):
        y_axis_title = y_axis
    if(group_by is not None):
        assert group_by in df.columns
        if(group_names is not None):
            assert isinstance(group_names, dict)
            for group in df[group_by].unique():
                if group not in group_names.keys():
                    group_names[group] = group
        else:
            group_names = dict(zip(df[group_by].unique(), map(str, df[group_by].unique())))
        if(legend_title is None):
            legend_title = group_by

    if y_error is None:
        if(group_by is not None):
            df_mean = mean_std_df(df, [x_axis, group_by], [y_axis])
        else:
            df_mean = mean_std_df(df, [x_axis], [y_axis])
    else:
        assert y_error in df.columns
        if group_by is not None:
            df_mean = df[[x_axis, group_by, y_axis, y_error]].copy()
        else:
            df_mean = df[[x_axis, y_axis, y_error]].copy()
        df_mean.rename(columns={y_axis: f'{y_axis}_mean', y_error: f'{y_axis}_std'}, inplace=True)
        if df_mean[x_axis].dtype != 'object':
            df_mean.sort_values(x_axis, inplace=True)
    
    if(group_by is not None):
        df_mean[group_by] = df_mean[group_by].map(group_names)
    
    if isinstance(enforce_legend_order, list):
        enforce_legend_order = [
            group_names[group] for group in enforce_legend_order if group in group_names and group_names[group] in df_mean[group_by].unique()
        ]
        assert len(enforce_legend_order) == len(set(df_mean[group_by].unique()).intersection(set(enforce_legend_order)))

    category_orders={
            group_by: enforce_legend_order
    } if isinstance(enforce_legend_order, list) else None

    # print(category_orders)
    # fig = px.bar(df, color_discrete_map= {'GOOG': 'black',
    #                                   'AAPL': 'grey',
    #                                   'AMZN': 'blue',
    #                                   'FB': 'green',
    #                                   'NFLX': 'red',
    #                                   'MSFT':'firebrick'}


    color_discrete_map = {}
    color_pool = cycle(px.colors.qualitative.Plotly)
    for _ in range(color_starting_index):
        next(color_pool)
    
    if color_group is not None:
        # print(color_group)
        assert isinstance(color_group, list)
        for group in color_group:
            assert isinstance(group, list)
            color = next(color_pool)
            for elem in group:
                if elem not in group_names:
                    continue
                elem = group_names[elem] # renamed
                if elem not in df_mean[group_by].unique():
                    continue
                color_discrete_map[elem] = color

        for elem in df_mean[group_by].unique():
            if elem not in color_discrete_map:
                color_discrete_map[elem] = next(color_pool)


    # print(color_discrete_map)
    
    if(plot_type == 'line'):
        fig = line(data_frame = df_mean,
                x=x_axis,
                y=f'{y_axis}_mean',
                error_y=f'{y_axis}_std',
                color=group_by,
                symbol=group_by if add_marker else None,
                symbol_sequence=symbol_sequence if add_marker else None,
                category_orders=category_orders,
                error_y_mode=error_y_mode,
                color_discrete_map=color_discrete_map,
                render_mode=render_mode
        )
    elif(plot_type == 'bar'):
        fig = px.bar(df_mean, 
                        x=x_axis, 
                        y=f'{y_axis}_mean', 
                        color=group_by, 
                        category_orders=category_orders,
                        error_y=f'{y_axis}_std' if error_y_mode is not None else None,
                        color_discrete_map=color_discrete_map
        )
        fig.update_layout(barmode='group')
    else:
        raise NotImplementedError
    
    
    fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            font=dict(
                # family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
            ),
            legend_title=legend_title,
            width=800,
            height=400,
            # hovermode="x"
    )
    

    if group_dashed_line is not None:
        assert isinstance(group_dashed_line, list)
        for group in group_dashed_line:
            if group in group_names and group_names[group] in df_mean[group_by].unique() and plot_type == 'line':
                fig.update_traces(patch={"line": {"dash": dash_type}}, selector={"legendgroup": group_names[group]})

    if group_hide_legend is not None:
        assert isinstance(group_hide_legend, list)
        group_hide_legend = [group_names[group] for group in group_hide_legend if group in group_names and group_names[group] in df_mean[group_by].unique()]
        for trace in fig['data']: 
            if (trace['name'] in group_hide_legend):
                trace['showlegend'] = False
       
    
    return fig



def bar_plot_for_multi_columns(
        df,
        x_axis='epoch',
        y_axiss = ['train_loss', 'val_loss'],
        x_axis_title = 'Epoch',
        y_axis_title = 'Loss',
        legend_title = 'Loss',
        legend_names={
            'train_loss': 'Train',
            'val_loss': 'Validation'
        }
    ):
    
    for y_axis in y_axiss:
        assert y_axis in df.columns
        assert y_axis in legend_names.keys()
    assert len(y_axiss) == len(legend_names)


    df_melt = df.melt(id_vars=[x_axis], 
                    value_vars=y_axiss, 
                    value_name='value',
                    var_name=legend_title)
    
    df_melt[legend_title] = df_melt[legend_title].map(legend_names)
    df_mean = mean_std_df(df_melt, [x_axis, legend_title], ['value'])

    fig = px.bar(df_mean, x=x_axis, y='value_mean', color=legend_title, error_y='value_std')
    
    fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            font=dict(
                # family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
            ),
            width=800,
            height=400,
            # hovermode="x"
        )
    return fig


def save_image_template(fig, 
                        width=400, 
                        height=300, 
                        legend_x=0, 
                        legend_y=1,
                        legend_orientation="h",
                        legend_font_size=16,
                        hide_grid_x=False,
                        hide_grid_y=False,
                        legend_yanchor="bottom",
                        legend_xanchor="left",
                        legend_entrywidth=0,
                        title=None,
                        title_font_size=16,
                        ):
    fig.update_layout(width=width, 
                     height=height,
                     title=title,
                     title_font_size=title_font_size,
                     margin=dict(l=5, r=5, b=5, t=5),
                     legend=dict(
                        entrywidth=legend_entrywidth,
                        orientation=legend_orientation,
                        yanchor=legend_yanchor,
                        y=legend_y,
                        xanchor=legend_xanchor,
                        x=legend_x,
                        font=dict(
                            size=legend_font_size
                        ),
                        bgcolor="rgba(0,0,0,0)"
                        # bgcolor="LightSteelBlue",
                        # bordercolor="Black",
                        # borderwidth=2,
                     ),
                     plot_bgcolor="white",
    )  
    # fig.update_traces(marker_size=1)
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='lightgrey',
        gridcolor='lightgrey' if not hide_grid_x else None
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='lightgrey',
        gridcolor='lightgrey' if not hide_grid_y else None
    )


    fig.update_layout(width=width, 
                     height=height)

    return fig

def nice_scatter_plot(
        df,
        x_axis,
        y_axis,
        # y_error=None,
        # plot_type='line',
        x_axis_title=None,
        y_axis_title=None,
        group_by=None,
        group_names=None,
        size_column=None,
        legend_title=None,
        # error_y_mode='band',
        enforce_legend_order=None,
        render_mode="auto",
        add_marker=False,
        color_group=None,
        group_hide_legend=None,
        # group_dashed_line=None,
        # dash_type="dot"
    ):
    df = df.copy()
    assert y_axis in df.columns
    # assert error_y_mode in ['band', 'bands', 'bar', 'bars', None]
    if(x_axis_title is None):
        x_axis_title = x_axis
    if(y_axis_title is None):
        y_axis_title = y_axis
    if(group_by is not None):
        assert group_by in df.columns
        if(group_names is not None):
            assert isinstance(group_names, dict)
            for group in df[group_by].unique():
                if group not in group_names.keys():
                    group_names[group] = group
        else:
            group_names = dict(zip(df[group_by].unique(), map(str, df[group_by].unique())))
        if(legend_title is None):
            legend_title = group_by

    if size_column is not None:
        assert size_column in df.columns

    
    
    if(group_by is not None):
        df[group_by] = df[group_by].map(group_names)
    
    if isinstance(enforce_legend_order, list):
        enforce_legend_order = [
            group_names[group] for group in enforce_legend_order if group in group_names and group_names[group] in df[group_by].unique()
        ]
        assert len(enforce_legend_order) == len(set(df[group_by].unique()).intersection(set(enforce_legend_order)))

    category_orders={
            group_by: enforce_legend_order
    } if isinstance(enforce_legend_order, list) else None

    

    color_discrete_map = {}
    color_pool = cycle(px.colors.qualitative.Plotly)
    if color_group is not None:
        # print(color_group)
        assert isinstance(color_group, list)
        for group in color_group:
            assert isinstance(group, list)
            color = next(color_pool)
            for elem in group:
                if elem not in group_names:
                    continue
                elem = group_names[elem] # renamed
                if elem not in df_mean[group_by].unique():
                    continue
                color_discrete_map[elem] = color

        for elem in df_mean[group_by].unique():
            if elem not in color_discrete_map:
                color_discrete_map[elem] = next(color_pool)


    # scatter plot
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color=group_by,
        size=size_column,
        category_orders=category_orders,
        symbol=group_by if add_marker else None,
        symbol_sequence=symbol_sequence if add_marker else None,
        color_discrete_map=color_discrete_map,
        render_mode=render_mode
    )
   
    
    fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            font=dict(
                # family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
            ),
            legend_title=legend_title,
            width=800,
            height=400,
            # hovermode="x"
    )
    

    if group_hide_legend is not None:
        assert isinstance(group_hide_legend, list)
        group_hide_legend = [group_names[group] for group in group_hide_legend if group in group_names and group_names[group] in df[group_by].unique()]
        for trace in fig['data']: 
            if (trace['name'] in group_hide_legend):
                trace['showlegend'] = False
       
    
    return fig




class Report():
    def __init__(self, filename, delete_existing=False):
        self.filename = filename
        if delete_existing:
            os.system(f"rm -f {self.filename}")
    
    def write(self, obj, append=True, add_new_line=True, replace_with_line_break=True, print_obj=True):
        with open(self.filename, 'a' if append else 'w') as f:
            if isinstance(obj, plotly.graph_objs._figure.Figure):
                f.write(obj.to_html(full_html=False, include_plotlyjs='cdn'))
            else:
                if print_obj:
                    print(obj)
                if replace_with_line_break:
                    obj = str(obj).replace("\n", "<br>")
                f.write(obj)
            if add_new_line:
                f.write("<br>")

            f.write("\n\n")