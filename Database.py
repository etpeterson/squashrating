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
                               "ORDER BY date(Date) ASC, Round ASC, ID ASC "
                               "".format(tuple(Player_ID)))
            except TypeError:
                self.c.execute("SELECT "
                               "Date,"
                               "Rating "
                               "FROM Elo_Ratings "
                               "WHERE Player_ID={} "
                               "ORDER BY date(Date) ASC, Round ASC, ID ASC "
                               "".format(Player_ID))
        else:
            try:
                self.c.execute("SELECT "
                               "Date,"
                               "Rating "
                               "FROM Elo_Ratings "
                               "WHERE Player_ID IN {0} "
                               "AND Date IN {1} "
                               "ORDER BY date(Date) ASC, Round ASC, ID ASC "
                               "".format(tuple(Player_ID)), tuple(Date))
            except TypeError:
                self.c.execute("SELECT "
                               "Date,"
                               "Rating "
                               "FROM Elo_Ratings "
                               "WHERE Player_ID={0} "
                               "AND Date={1} "
                               "ORDER BY date(Date) ASC, Round ASC, ID ASC "
                               "".format(Player_ID, Date))
        rating = self.c.fetchall()
        return self.EloStartingRating if rating is None else rating

    def GetLatestEloRating(self, Player_ID):
        self.c.execute("SELECT "
                       "Rating "
                       "FROM Elo_Ratings "
                       "WHERE Player_ID={} "
                       "ORDER BY date(Date) DESC, Round DESC, ID DESC "
                       "LIMIT 1"
                       "".format(Player_ID))
        rating = self.c.fetchone()
        return self.EloStartingRating if rating is None else rating[0]

    def GetLatestEloRatings(self, Player_ID):
        self.c.execute("SELECT "
                       "Rating "
                       "FROM Elo_Ratings "
                       "WHERE Player_ID IN {} "
                       "ORDER BY date(Date) DESC, Round DESC, ID DESC "
                       "LIMIT 1"
                       "".format(tuple(Player_ID)))
        rating = self.c.fetchone()
        return self.EloStartingRating if rating is None else rating[0]

    def GetLatestEloDate(self, Player_ID):
        self.c.execute("SELECT "
                       "Date "
                       "FROM Elo_Ratings "
                       "WHERE Player_ID={} "
                       "ORDER BY date(Date) DESC, Round DESC, ID DESC "
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


        #TODO: make sure all the players who played matches are added with default values!