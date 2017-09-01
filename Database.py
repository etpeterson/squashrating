# from bs4 import BeautifulSoup
from datetime import datetime, date
# import re
import sys
# import os
import sqlite3
# import sklearn as skl
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
import scipy as sp
from itertools import combinations, chain
#import skills.elo as elo  # figure this out: player ratings! Is this glicko2?
# import trueskill  # also player ratings (or use TrueSkill in skills?) NOT FREE!
from pandas import DataFrame, Series
#from functools import partial
import matplotlib.pyplot as pl

__author__ = 'epeterson'


# Helper functions
all_player_ID_query_str = "SELECT Player_ID FROM Player " \
                               "WHERE Male IS NOT NULL " \
                               "AND Height IS NOT NULL " \
                               "AND Weight IS NOT NULL"

#returns: Male, Height, Weight for selected player
basic_player_query_str = "SELECT Male, Height, Weight " \
                              "FROM Player WHERE  Player_ID=?"
basic_player_query_names = ('Male', 'Height', 'Weight')

#returns: Number_of_Games, Duration, Game1_total_points, Game1_point_differential,
# Game2_total_points, Game2_point_differential, Game3_total_points, Game3_point_differential,
# Game4_total_points, Game4_point_differential, Game5_total_points, Game5_point_differential,
# PSA, Player_age_in_days
individual_match_query_date_str = "SELECT " \
         "Match.Number_of_Games, " \
         "Match.Duration, " \
         "coalesce(Match.Game1_Winner_Score,0) + coalesce(Match.Game1_Loser_Score,0), " \
         "coalesce(Match.Game1_Winner_Score,0) - coalesce(Match.Game1_Loser_Score,0), " \
         "coalesce(Match.Game2_Winner_Score,0) + coalesce(Match.Game2_Loser_Score,0), " \
         "coalesce(Match.Game2_Winner_Score,0) - coalesce(Match.Game2_Loser_Score,0), " \
         "coalesce(Match.Game3_Winner_Score,0) + coalesce(Match.Game3_Loser_Score,0), " \
         "coalesce(Match.Game3_Winner_Score,0) - coalesce(Match.Game3_Loser_Score,0), " \
         "coalesce(Match.Game4_Winner_Score,0) + coalesce(Match.Game4_Loser_Score,0), " \
         "coalesce(Match.Game4_Winner_Score,0) - coalesce(Match.Game4_Loser_Score,0), " \
         "coalesce(Match.Game5_Winner_Score,0) + coalesce(Match.Game5_Loser_Score,0), " \
         "coalesce(Match.Game5_Winner_Score,0) - coalesce(Match.Game5_Loser_Score,0), " \
         "Match.PSA, " \
         "CAST(julianday(Match.Date) - julianday(Player.DOB) AS INTEGER) " \
         "FROM Match JOIN Player " \
         "ON Player.Player_ID=? " \
         "WHERE Match.Number_of_Games>0 " \
         "AND (Match.Winner_ID=? " \
         "OR Match.Loser_ID=?) " \
         "AND Match.Duration IS NOT NULL " \
         "AND Match.Retired=0"
individual_match_query_date_names = ('Number_of_Games','Duration','Game1_Points',
                                     'Game1_Differential','Game2_Points','Game2_Differential',
                                     'Game3_Points','Game3_Differential','Game4_Points',
                                     'Game4_Differential','Game5_Points','Game5_Differential',
                                     'PSA','Age_at_Match')

#returns: Winner_ID, Number_of_Games, Duration, Game1_total_points, Game1_point_differential,
# Game2_total_points, Game2_point_differential, Game3_total_points, Game3_point_differential,
# Game4_total_points, Game4_point_differential, Game5_total_points, Game5_point_differential,
# PSA, Loser_age_minus_winner_age_in_days
shared_match_query_date_str = "SELECT " \
         "Match.Winner_ID, " \
         "Match.Number_of_Games, " \
         "Match.Duration, " \
         "coalesce(Match.Game1_Winner_Score,0) + coalesce(Match.Game1_Loser_Score,0), " \
         "coalesce(Match.Game1_Winner_Score,0) - coalesce(Match.Game1_Loser_Score,0), " \
         "coalesce(Match.Game2_Winner_Score,0) + coalesce(Match.Game2_Loser_Score,0), " \
         "coalesce(Match.Game2_Winner_Score,0) - coalesce(Match.Game2_Loser_Score,0), " \
         "coalesce(Match.Game3_Winner_Score,0) + coalesce(Match.Game3_Loser_Score,0), " \
         "coalesce(Match.Game3_Winner_Score,0) - coalesce(Match.Game3_Loser_Score,0), " \
         "coalesce(Match.Game4_Winner_Score,0) + coalesce(Match.Game4_Loser_Score,0), " \
         "coalesce(Match.Game4_Winner_Score,0) - coalesce(Match.Game4_Loser_Score,0), " \
         "coalesce(Match.Game5_Winner_Score,0) + coalesce(Match.Game5_Loser_Score,0), " \
         "coalesce(Match.Game5_Winner_Score,0) - coalesce(Match.Game5_Loser_Score,0), " \
         "Match.PSA, " \
         "CAST(julianday(Loser.DOB) - julianday(Winner.DOB) AS INTEGER) " \
         "FROM Match JOIN Player AS Winner " \
         "ON Winner.Player_ID=Match.Winner_ID " \
         "JOIN Player AS Loser " \
         "ON Loser.Player_ID=Match.Loser_ID " \
         "WHERE Match.Number_of_Games>0 " \
         "AND ((Match.Winner_ID=? " \
         "AND Match.Loser_ID=?) " \
         "OR (Match.Loser_ID=? " \
         "AND Match.Winner_ID=?)) " \
         "AND Match.Duration IS NOT NULL " \
         "AND Match.Retired=0"
shared_match_query_date_names = ('Winner_ID','Number_of_Games','Duration','Game1_Points',
                                     'Game1_Differential','Game2_Points','Game2_Differential',
                                     'Game3_Points','Game3_Differential','Game4_Points',
                                     'Game4_Differential','Game5_Points','Game5_Differential',
                                     'PSA','Age_Difference')

ordered_match_query_date_str = "SELECT " \
                               "Winner_ID, " \
                               "Loser_ID " \
                               "Date, " \
                               "FROM MATCH " \
                               "ORDER BY Date ASC"

ordered_match_query_date_round_str = "SELECT " \
                                     "Winner_ID, " \
                                     "Loser_ID, " \
                                     "Date, " \
                                     "lower(Round), " \
                                     "(6 - Number_of_Games)" \
                                     "FROM Match " \
                                     "ORDER BY " \
                                     "Date, " \
                                     "CASE lower(Round) " \
                                     "WHEN 'f' THEN 100 " \
                                     "WHEN '3/4' THEN 95 " \
                                     "WHEN 'sf' THEN 90 " \
                                     "WHEN '5/6' THEN 87 " \
                                     "WHEN '7/8' THEN 83 " \
                                     "WHEN '5/8' THEN 81 " \
                                     "WHEN 'qf' THEN 80 " \
                                     "WHEN 'r4' THEN 70 " \
                                     "WHEN 'r3' THEN 60 " \
                                     "WHEN 'r2' THEN 50 " \
                                     "WHEN 'r1' THEN 40 " \
                                     "WHEN 'q' THEN 30 " \
                                     "WHEN 'p' THEN 20 " \
                                     "WHEN '' THEN 10 " \
                                     "END;"


def query_player_ID(c, querystr):
    #  all_player_ID_query_str
    c.execute(querystr)
    return list(chain(*c.fetchall()))

def query_player(c, querystr, pairval, player_pair):
    # basic_player_query_str
    c.execute(querystr, (player_pair[pairval],))
    return np.asarray(c.fetchone())
    # if arr:
    #    return DataFrame(np.atleast_2d(np.asarray(arr, dtype=np.int)),
    #                     columns=columns)
    # else:
    #    return DataFrame([], columns=columns)

def query_match(c, querystr, columns, pairval, player_pair):
    # individual_match_query_date_str
    c.execute(querystr, (player_pair[pairval],)*3)
    return np.asarray(c.fetchall())
    # if arr:
    #     return DataFrame(np.atleast_2d(np.asarray(arr, dtype=np.int)),
    #                      columns=columns)
    # else:
    #     return DataFrame([], columns=columns)

def query_match_shared(c, querystr, columns, player_pair):
    # shared_match_query_date_str
    c.execute(querystr, player_pair*2)
    return np.asarray(c.fetchall())
    # if arr:
    #     return DataFrame(np.atleast_2d(np.asarray(arr, dtype=np.int)),
    #                      columns=columns)
    # else:
    #     return DataFrame([], columns=columns)
    # dtypes=[(x,np.int) for x in columns]
    # starr=np.asarray(arr, dtype=dtypes)


fast_colnames = ['P1_{}'.format(i) for i in basic_player_query_names] +\
                ['M1_{}'.format(i) for i in individual_match_query_date_names] +\
                ['M1_Number_of_Matches'] +\
                ['P2_{}'.format(i) for i in basic_player_query_names] +\
                ['M2_{}'.format(i) for i in individual_match_query_date_names] +\
                ['M2_Number_of_Matches'] +\
                ['M12_{}'.format(i) for i in shared_match_query_date_names[1:]] +\
                ['M12_Number_of_Matches']

def query_function_fast(c, player_pair):
    c.execute(shared_match_query_date_str, player_pair*2)
    m12 = np.asarray(c.fetchall(), dtype=np.int)
    if m12.size > 0:
        c.execute(basic_player_query_str, (player_pair[0],))
        p1 = np.asarray(c.fetchone(), dtype=np.int)
        c.execute(basic_player_query_str, (player_pair[1],))
        p2 = np.asarray(c.fetchone(), dtype=np.int)
        c.execute(individual_match_query_date_str, (player_pair[0],)*3)
        m1 = np.asarray(c.fetchall(), dtype=np.int)
        c.execute(individual_match_query_date_str, (player_pair[1],)*3)
        m2 = np.asarray(c.fetchall(), dtype=np.int)
        # TODO: query wins and losses separately

        #match_dates = m12[:, -1]
        Winner_ID = m12[:, 0]
        # a winner ID method, works well!
        # y = (np.sum(Winner_ID == player_pair[0]) -\
        #         np.sum(Winner_ID == player_pair[1])) /\
        #         Winner_ID.size
        # a game based method, not as well, but interesting results
        y = 6 - m12[:, 1]
        y[Winner_ID == player_pair[1]] *= -1
        y = y.mean()

        m12 = m12[:, 1:]
        #if m1.any() and m2.any() and m12.any():
        X = np.atleast_2d(np.hstack(
            (p1, m1.mean(axis=0), np.atleast_1d(m1.shape[0]), p2,
             m2.mean(axis=0), np.atleast_1d(m2.shape[0]), m12.mean(axis=0),
             np.atleast_1d(m12.shape[0]))))
        # if X.any():
        #     X = np.concatenate((X, X_new), axis=0)
        #     y = np.concatenate((y, np.atleast_1d(y_new)), axis=0)
        #     #id_key = np.concatenate((id_key, np.atleast_2d(player_pair)),
        #     #                        axis=0)
        # else:
        #     X = X_new
        #     y = np.atleast_1d(y_new)
        #     #id_key = np.atleast_2d(player_pair)
        return (X, y)
    else:
        return (None, None)


# add an elo processing module here to generate past and current elo ratings



class SquashPredict(object):

    def __init__(self, X, y=None, method=svm.SVR()):

        if y is None:  # this means X is a DataFrame and contains X and y
            self.y = X['y'].squeeze()
            X.drop('y')
            self.X = np.atleast_2d(X.values)
        else:
            self.X = np.atleast_2d(X)
            self.y = y.squeeze()
            if isinstance(X, DataFrame):
                self.X = np.atleast_2d(self.X.values)
            if isinstance(y, DataFrame):
                self.y = self.y.values.squeeze()
        self.method = method
        self.is_fitted = False

    def __str__(self):
        return 'Is {} fitted?'.format(self.method, self.is_fitted)

    def fit(self):
        self.method.fit(self.X, self.y)
        self.is_fitted = True

    def score(self):
        return self.method.score(self.X, self.y)

    def predict(self, return_probability = False):
        pred = np.atleast_2d(self.method.predict(self.X)).T
        if return_probability:
            try:
                pred = np.hstack((pred, self.method.predict_proba(self.X)))
            except AttributeError:
                pass  # not a great idea
        return pred

    def comparison(self, return_probability = False):
        return np.hstack((np.atleast_2d(self.y).T,
                          self.predict(return_probability)))

    def PCA(self):
        print('principal component analysis')

    def SVD(self):
        return np.linalg.svd(self.X)

    def visualize(self):
        print('some visualization with matplotlib?')





colnames = ['P1_{}'.format(i) for i in basic_player_query_names] +\
           ['P2_{}'.format(i) for i in basic_player_query_names] +\
           ['M1_{}'.format(i) for i in individual_match_query_date_names] +\
           ['M2_{}'.format(i) for i in individual_match_query_date_names] +\
           ['M12_{}'.format(i) for i in shared_match_query_date_names]

def query_player_ID_fast(c):
    #  all_player_ID_query_str
    c.execute(all_player_ID_query_str)
    return list(chain(*c.fetchall()))

def query_player_ID_DB(c):
    return query_player_ID(c, all_player_ID_query_str)

def query_player_DB1(c, player_pair):
    return query_player(c, basic_player_query_str, colnames[0:3], 0, player_pair)

def query_player_DB2(c, player_pair):
    return query_player(c, basic_player_query_str, colnames[3:6], 1, player_pair)

def query_match_DB1(c, player_pair):
    df = query_match(c, individual_match_query_date_str, colnames[6:20], 0, player_pair)
    return df.mean().to_frame().transpose()
    #return DataFrame(np.atleast_2d(df.mean().values), columns=colnames[6:20])

def query_match_DB2(c, player_pair):
    df = query_match(c, individual_match_query_date_str, colnames[20:34], 1, player_pair)
    return df.mean().to_frame().transpose()
    #return DataFrame(np.atleast_2d(df.mean().values), columns=colnames[20:34])

def query_match_shared_DB(c, player_pair):
    ret = query_match_shared(c, shared_match_query_date_str, colnames[34:49], player_pair)
    Winner_ID = ret['M12_Winner_ID']
    # a winner ID method, works well!
    # y_new = (np.sum(Winner_ID == player_pair[0]) -\
    #         np.sum(Winner_ID == player_pair[1])) /\
    #         Winner_ID.size
    # a game based method, not as well, but interesting results
    y_new = 6 - ret['M12_Number_of_Games']
    y_new[Winner_ID == player_pair[1]] *= -1
    ret = ret.join(DataFrame({'y':y_new}))
    # ugh, have to redefine the dataframe because the mean collapses it to a Series
    return ret.mean().to_frame().transpose()
    #return DataFrame(np.atleast_2d(ret.mean().values),columns=colnames[34:49] + ['y'])
#end the new helper functions

ordered_match_query_date_round_str = "SELECT " \
                                     "Winner_ID, " \
                                     "Loser_ID, " \
                                     "Date, " \
                                     "lower(Round), " \
                                     "(6 - Number_of_Games)" \
                                     "FROM Match " \
                                     "ORDER BY " \
                                     "Date, " \
                                     "CASE lower(Round) " \
                                     "WHEN 'f' THEN 100 " \
                                     "WHEN '3/4' THEN 95 " \
                                     "WHEN 'sf' THEN 90 " \
                                     "WHEN '5/6' THEN 87 " \
                                     "WHEN '7/8' THEN 83 " \
                                     "WHEN '5/8' THEN 81 " \
                                     "WHEN 'qf' THEN 80 " \
                                     "WHEN 'r4' THEN 70 " \
                                     "WHEN 'r3' THEN 60 " \
                                     "WHEN 'r2' THEN 50 " \
                                     "WHEN 'r1' THEN 40 " \
                                     "WHEN 'q' THEN 30 " \
                                     "WHEN 'p' THEN 20 " \
                                     "WHEN '' THEN 10 " \
                                     "END;"

ordered_match_query_date_round_num_str = "SELECT " \
                                     "Winner_ID, " \
                                     "Loser_ID, " \
                                     "Date, " \
                                     "CASE lower(Round) " \
                                     "WHEN 'f' THEN 100 " \
                                     "WHEN '3/4' THEN 95 " \
                                     "WHEN 'sf' THEN 90 " \
                                     "WHEN '5/6' THEN 87 " \
                                     "WHEN '7/8' THEN 83 " \
                                     "WHEN '5/8' THEN 81 " \
                                     "WHEN 'qf' THEN 80 " \
                                     "WHEN 'r4' THEN 70 " \
                                     "WHEN 'r3' THEN 60 " \
                                     "WHEN 'r2' THEN 50 " \
                                     "WHEN 'r1' THEN 40 " \
                                     "WHEN 'q' THEN 30 " \
                                     "WHEN 'p' THEN 20 " \
                                     "WHEN '' THEN 10 " \
                                     "END, " \
                                     "(6 - Number_of_Games)" \
                                     "FROM Match " \
                                     "ORDER BY " \
                                     "Date, " \
                                     "CASE lower(Round) " \
                                     "WHEN 'f' THEN 100 " \
                                     "WHEN '3/4' THEN 95 " \
                                     "WHEN 'sf' THEN 90 " \
                                     "WHEN '5/6' THEN 87 " \
                                     "WHEN '7/8' THEN 83 " \
                                     "WHEN '5/8' THEN 81 " \
                                     "WHEN 'qf' THEN 80 " \
                                     "WHEN 'r4' THEN 70 " \
                                     "WHEN 'r3' THEN 60 " \
                                     "WHEN 'r2' THEN 50 " \
                                     "WHEN 'r1' THEN 40 " \
                                     "WHEN 'q' THEN 30 " \
                                     "WHEN 'p' THEN 20 " \
                                     "WHEN '' THEN 10 " \
                                     "END;"


class DBQuery(object):

    def __init__(self, initial_query, query_function, database_name,
                 column_names, elo_query=ordered_match_query_date_round_num_str,
                 player_combination=2):
        self.column_names = column_names
        self.player_combination = player_combination
        self.query_function = query_function
        self.database_name = database_name
        self.elo_query = elo_query
        self.X = np.array([]).reshape(0, len(self.column_names))
        self.y = np.array([]).reshape(0, 1)
        self.initial_query = initial_query
        self.matches = []

    def query_data(self):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()

        all_players = self.initial_query(c)
        #qlen = len(self.looping_query_list)
        #dfi = 0
        for player_pair in combinations(all_players, self.player_combination):
            X_tmp, y_tmp = self.query_function(c, player_pair)
            if X_tmp is not None and y_tmp is not None:
                self.X = np.vstack((self.X, X_tmp))
                self.y = np.vstack((self.y, y_tmp))
            #TODO: run self.query_function here!
            # #print(player_pair)
            # i = 0
            # l = True
            # while i < qlen and l:
            #     Xy_tmp = self.looping_query_list[i](player_pair)
            #     if not Xy_tmp.empty:
            #         #if isinstance(Xy_tmp, Series):  # so annoying!
            #         #    Xy_tmp = Xy_tmp.to_frame()
            #         self.Xy.set_value(dfi, list(Xy_tmp.columns), Xy_tmp.values)
            #         #df.set_value(0,('two','three'),(2,3))
            #         #df.ix[1,'two']=12
            #     else:
            #         l = False
            #     i += 1
            # if l:
            #     dfi += 1
        #self.X = self.Xy.ix[:, self.column_names]
        #self.y = self.Xy.ix[:, 'y']
        c.close()
        conn.close()

    def query_for_elo(self):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        c.execute(self.elo_query)
        self.matches = c.fetchall()
        c.close()
        conn.close()


    def get_data(self):
        return (self.X, self.y)

    def get_data_frame(self):
        return DataFrame(np.hstack((self.X, self.y)), columns=self.column_names + 'y')

    def get_dict_names(self, player_ID_list):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        c.execute("SELECT First_Name, Middle_Name, Last_Name "
              "FROM Player WHERE Player_ID IN {}"
              "".format(tuple(player_ID_list)))
        name_list = c.fetchall()
        name_list = [' '.join(name) for name in name_list]
        c.close()
        conn.close()
        return dict(zip(player_ID_list, name_list))


class ratingDB(object):
    def __init__(self, curdate=date.today(), database_name=':memory:',
                 matches = None,
                 EloRatings = None, BayesRatings = None,
                 EloStartingRating = 1500, BayesStartingRating = 5,
                 BayesStartingStd = 1):
        # this database should have 4 tables:
        # Elo_Ratings (autoincrement, Player_ID, Date, Rating)
        # Bayes_Ratings (autoincrement, Player_ID, Date, Rating)
        # Matches (autoincrement, Winner_ID, Loser_ID, Date, Round, MOV)
        # Settings (???)
        self.curdate = curdate
        self.database_name = database_name
        self.conn = sqlite3.connect(self.database_name)
        self.c = self.conn.cursor()
        self.EloStartingRating = EloStartingRating
        self.EloStartingDate = '0001-01-1'
        self.BayesStartingRating = BayesStartingRating
        self.BayesStartingStd = BayesStartingStd

        self.c.execute(
            "CREATE TABLE IF NOT EXISTS Elo_Ratings("
            "ID INTEGER PRIMARY KEY AUTOINCREMENT,"
            "Player_ID INTEGER,"
            "Date TEXT,"
            "Round INTEGER,"
            "Rating REAL);")

        self.c.execute(
            "CREATE TABLE IF NOT EXISTS Bayes_Ratings("
            "ID INTEGER PRIMARY KEY AUTOINCREMENT,"
            "Player_ID INTEGER,"
            "Date TEXT,"
            "Round INTEGER,"
            "Rating REAL,"
            "STD REAL);")

        self.c.execute(
            "CREATE TABLE IF NOT EXISTS Matches("
            "ID INTEGER PRIMARY KEY AUTOINCREMENT,"
            "Winner_ID INTEGER,"
            "Loser_ID INTEGER,"
            "Date TEXT,"
            "Round INTEGER,"
            "MOV REAL);")

        self.matches = matches
        self.EloRatings = EloRatings
        self.BayesRatings = BayesRatings

        #add to the databases
        self.EloRatings2DB()
        self.BayesRatings2DB()
        self.Matches2DB()


    #def BeginTransaction(self):
    #    self.c.execute("BEGIN TRANSACTION")

    #def EndTransaction(self):
    #    self.c.execute("END TRANSACTION")

    def Matches2DB(self, matches=None):
        #matches is nx4
        #each row is a match
        #the columns are: winner ID, loser ID, match date, margin of victory
        if matches is None:
            matches = self.matches

        for match in matches:
            self.c.execute(
                    "INSERT INTO Matches("
                    "Winner_ID,"
                    "Loser_ID,"
                    "Date,"
                    "Round,"
                    "MOV)"
                    "VALUES(?, ?, ?, ?, ?)",
                    match)

    def EloRatings2DB(self, EloRatings = None):
        if EloRatings is None:
            EloRatings = self.EloRatings

        if EloRatings is not None:
            for rating in EloRatings:
                self.c.execute(
                    "INSERT INTO Elo_Ratings("
                    "Player_ID,"
                    "Date,"
                    "Round,"
                    "Rating) "
                    "VALUES(?, ?, ?, ?)",
                    rating)

    def SetEloRating(self, ID, Date, Round, Rating):
        self.EloRatings2DB(((ID, Date, Round, Rating),))

    def BayesRatings2DB(self, BayesRatings=None):
        if BayesRatings is None:
            BayesRatings = self.BayesRatings

        if BayesRatings is not None:
            for rating in BayesRatings:
                self.c.execute(
                    "INSERT INTO Bayes_Ratings("
                    "Player_ID,"
                    "Date,"
                    "Rating,"
                    "STD)"
                    "VALUES(?, ?, ?)",
                    rating)

    def GetMatches(self):
        self.c.execute("SELECT "
                       "Winner_ID,"
                       "Loser_ID,"
                       "Date,"
                       "Round,"
                       "MOV "
                       "FROM Matches")
        return self.c.fetchall()

    def GetUniqueMatchDates(self):
        print("unimplemented")

    def GetEloRating(self, Player_ID, Date = None):
        if Date is None:
            try:
                self.c.execute("SELECT "
                               "Date,"
                               "Rating "
                               "FROM Elo_Ratings "
                               "WHERE Player_ID IN {} "
                               "ORDER BY Date ASC, Round ASC, ID ASC "
                               "".format(tuple(Player_ID)))
            except TypeError:
                self.c.execute("SELECT "
                               "Date,"
                               "Rating "
                               "FROM Elo_Ratings "
                               "WHERE Player_ID={} "
                               "ORDER BY Date ASC, Round ASC, ID ASC "
                               "".format(Player_ID))
        else:
            try:
                self.c.execute("SELECT "
                               "Date,"
                               "Rating "
                               "FROM Elo_Ratings "
                               "WHERE Player_ID IN {0} "
                               "AND Date IN {1} "
                               "ORDER BY Date ASC, Round ASC, ID ASC "
                               "".format(tuple(Player_ID)), tuple(Date))
            except TypeError:
                self.c.execute("SELECT "
                               "Date,"
                               "Rating "
                               "FROM Elo_Ratings "
                               "WHERE Player_ID={0} "
                               "AND Date={1} "
                               "ORDER BY Date ASC, Round ASC, ID ASC "
                               "".format(Player_ID, Date))
        rating = self.c.fetchall()
        return self.EloStartingRating if rating is None else rating

    def GetLatestEloRating(self, Player_ID):
        self.c.execute("SELECT "
                       "Rating "
                       "FROM Elo_Ratings "
                       "WHERE Player_ID={} "
                       "ORDER BY Date DESC, Round DESC, ID DESC "
                       "LIMIT 1"
                       "".format(Player_ID))
        rating = self.c.fetchone()
        return self.EloStartingRating if rating is None else rating[0]

    def GetLatestEloRatings(self, Player_ID):
        self.c.execute("SELECT "
                       "Rating "
                       "FROM Elo_Ratings "
                       "WHERE Player_ID IN {} "
                       "ORDER BY Date DESC, Round DESC, ID DESC "
                       "LIMIT 1"
                       "".format(tuple(Player_ID)))
        rating = self.c.fetchone()
        return self.EloStartingRating if rating is None else rating[0]

    def GetLatestEloDate(self, Player_ID):
        self.c.execute("SELECT "
                       "Date "
                       "FROM Elo_Ratings "
                       "WHERE Player_ID={} "
                       "ORDER BY Date DESC, Round DESC, ID DESC "
                       "LIMIT 1"
                       "".format(Player_ID))
        date = self.c.fetchone()
        return self.EloStartingDate if date is None else date[0]


    def GetBayesRating(self, Player_ID, Date=None):
        if Date is None:
            self.c.execute("SELECT"
                           "Date,"
                           "Rating,"
                           "STD "
                           "FROM Bayes_Ratings "
                           "ORDER BY Date "
                           "WHERE Player_ID IN {}"
                           "".format(tuple(Player_ID)))
        else:
            self.c.execute("SELECT"
                           "Date,"
                           "Rating,"
                           "STD "
                           "FROM Bayes_Ratings "
                           "ORDER BY Date "
                           "WHERE Player_ID IN {0}"
                           "AND Date IN {1}"
                           "".format(tuple(Player_ID)), tuple(Date))
        rating = self.c.fetchone()
        return (self.BayesStartingRating, self.BayesStartingStd) \
            if rating is None else rating

    def GetLatestBayesRating(self, Player_ID):
        self.c.execute("SELECT"
                       "Rating "
                       "FROM Bayes_Ratings "
                       "ORDER BY Date "
                       "WHERE Player_ID IN {}"
                       "LIMIT 1"
                       "".format(tuple(Player_ID)))
        rating = self.c.fetchone()
        return self.BayesStartingRating if rating is None else rating

    def PlotElo(self, ID_list, prevent_same_day_matches=True,
                player_names=None):
        fig = pl.figure()
        ax = pl.subplot()
        for ID in ID_list:
            ratings = self.GetEloRating(ID)
            dates = np.array(ratings)[:, 0]
            ratings = np.array(ratings)[:, 1]
            dayssince2000 = [(datetime.strptime(d,'%Y-%m-%d') -
                              datetime(2000, 1, 1)).days for d in dates]
            #kind of random x-axis, days since Jan 1, 2000
            if prevent_same_day_matches:
                for idx in list(range(1, len(dayssince2000))):
                    if dayssince2000[idx - 1] >= dayssince2000[idx]:
                        dayssince2000[idx] = dayssince2000[idx - 1] + 1
            labelstr = player_names[ID] if player_names is not None else str(ID)
            # if player_names is not None:
            #    labelstr = player_names[ID]
            # else:
            #    labelstr = str(ID)
            ax.plot(dayssince2000, ratings, '.--', label=labelstr)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def Commit(self):
        self.conn.commit()

    def Close(self):
        self.c.close()


        #TODO: make sure all the players who played matches are added with default values!