#import sys
import os
#import sqlite3
import numpy as np
import matplotlib.pyplot as pl
#import holoviews as hv
#hv.extension('bokeh')
import bokeh.palettes as pa
import bokeh.plotting as bp
#import bokeh.charts as bc
#import bkcharts as bk
from bokeh.models import HoverTool, ColumnDataSource
from Database import DBQuery, ratingDB, query_player_ID_fast, \
    query_function_fast, fast_colnames
#import datetime
import pandas as pd

playerdb_fname = 'sqlitedb_complete.sqlite'
paired_query = DBQuery(query_player_ID_fast, query_function_fast,
                       playerdb_fname, fast_colnames)

player_plot_IDs = list(range(1,20))
#player_plot_IDs = [9, 13, 18, 21, 24, 31]
name_dict = paired_query.get_dict_names(player_plot_IDs)
paired_query.query_for_elo()


#current Elo!
ratingdb_fname = 'ratingdb.sqlite'
rating_DB = ratingDB(matches=paired_query.matches, database_name=ratingdb_fname)
#rating_DB.PlotElo(player_plot_IDs, player_names=name_dict)
#pl.show()
#print('Elo done')

#TODO: customize the hover tool!
hover = HoverTool()
#hover = HoverTool(tooltips=[
#    ("index", "$index"),
#    ("(x,y)", "($x, $y)"),
#    ("desc", "@desc"),
#])

inferno_palette = pa.inferno(len(player_plot_IDs))
ratings_all = []
p = bp.figure(plot_width=1000, plot_height=600, x_axis_type="datetime")#, tools=[hover])
p.title.text = "Player Ratings"
for ID in enumerate(player_plot_IDs):
    #datetime.datetime.strptime()
    ratings = rating_DB.GetEloRating(ID[1])
    labelstr = ' '.join(name_dict[ID[1]].split()) if name_dict is not None else str(ID[1])
    ratings_all += [list(elem)+[labelstr,]+[inferno_palette[ID[0]],] for elem in ratings]
    #ratingsdf = pd.DataFrame(ratings)

    #dates = pd.to_datetime(np.array(ratings)[:, 0])
    #ratings = np.array(ratings)[:, 1]
    #labelstr = name_dict[ID] if name_dict is not None else str(ID)
    #p.line(dates, ratings)
    #p.scatter(dates, ratings, legend=labelstr)
    #print(dates, ratings, labelstr)
df = pd.DataFrame(ratings_all, columns=['Date', 'Rating', 'Name', 'Color'])
df['Date'] = pd.to_datetime(df['Date'])
source = ColumnDataSource(df)


##bc.Scatter(df, x='Dates', y='Rating', legend='Name')

p = bp.figure(plot_width=1000, plot_height=600, x_axis_type="datetime", tools=[hover])#, tools=[hover])
#p.title.text = "Player Ratings"
#p.scatter(dates, ratings, legend=labelstr)
# hmm, multi_line wants a 2D array. Why can't this
#p.multi_line('Date', 'Rating', source=source, legend='Name')
for player in pd.unique(df['Name']):
    df_player = df.loc[df['Name'] == player, ['Date', 'Rating', 'Color']]
    color = pd.unique(df.loc[df['Name'] == player, ['Color']]['Color'])[0]
    source_player = ColumnDataSource(df.loc[df['Name'] == player, ['Date', 'Rating', 'Color']])
    p.line(df_player['Date'].values, df_player['Rating'].values, line_color=color) #, source=source_player
p.circle('Date', 'Rating', source=source, legend='Name', fill_color='Color')
p.legend.location = "bottom_right"
p.legend.click_policy = "hide"
bp.show(p)
#bp.save(p, '/Users/epeterson/Documents/RaspberryPi/squash/ratingplot_Py.html')

rating_DB.Close()