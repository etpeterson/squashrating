# from bs4 import BeautifulSoup
from datetime import datetime
# import re
import sys
import os
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
from Rate import EloCalc, BayesCalc
from Database import DBQuery, ratingDB, query_player_ID_fast, \
    query_function_fast, fast_colnames


#possible rounds:
# F
# sf
# qf
# r2
# r1
# 3/4
# r3
# P
#
# Q
# r4
# 7/8
# 5/6
# 5/8
#sqlite> SELECT lower(*) FROM Match ORDER BY Date, Round ASC LIMIT 50;
#SELECT Winner_ID, Loser_ID, Date, lower(Round) FROM Match ORDER BY Date, Round ASC LIMIT 50;
#https://stackoverflow.com/questions/3303851/sqlite-and-custom-order-by
#SELECT Winner_ID, Loser_ID, Date, lower(Round) FROM Match ORDER BY Date, CASE lower(Round) WHEN 'f' THEN 100 WHEN '3/4' THEN 95 WHEN 'sf' THEN 90 WHEN '5/6' THEN 87 WHEN '7/8' THEN 83 WHEN '5/8' THEN 81 WHEN 'qf' THEN 80 WHEN 'r4' THEN 70 WHEN 'r3' THEN 60 WHEN 'r2' THEN 50 WHEN 'r1' THEN 40 WHEN 'q' THEN 30 WHEN 'p' THEN 20 WHEN '' THEN 10 END LIMIT 10;

"""
THIS IS THE OLD VERSION OF THIS STUFF. IT'S NOW BEEN MODIFIED AND MOVED TO RATE
class EloCalc(object):

    def __init__(self, mov_thresh=5, klog_center=2200,
                 klog_width=200, klog_min_k=10, klog_max_k=40, ratings=None,
                 rating_dates=None, rating_round=None, starting_rating=1500):
        self.mov_thresh = mov_thresh
        self.klog_center = klog_center
        self.klog_width = klog_width
        self.klog_min_k = klog_min_k
        self.klog_max_k = klog_max_k
        if ratings is None:
            self.ratings = {}
        else:
            self.ratings = ratings
        if rating_dates is None:
            self.rating_dates = {}
        else:
            self.rating_dates = rating_dates
        if rating_round is None:
            self.rating_round = {}
        else:
            self.rating_round = rating_round
        self.starting_rating = starting_rating

    # thanks to andr3w321 for this function
    # http://andr3w321.com/elo-ratings-part-1/
    # p1 is the winner, p2 the loser
    # here if k is none, it gets automatically scaled, otherwise the input
    # value is used
    def rate_1vs1(self, p1, p2, mov=0, k=None, drawn=False):
        k_multiplier = 1.0
        corr_m = 1.0
        if mov >= self.mov_thresh and not drawn:
            k_multiplier = np.log(abs(mov) + 1)
            corr_m = 2.2 / ((p1 - p2)*.001 + 2.2)
        if k is None:
            k = self.k_logistic((p1 + p2) / 2)
        rp1 = np.power(10, p1/400)
        rp2 = np.power(10, p2/400)
        exp_p1 = rp1 / (rp1 + rp2)
        exp_p2 = rp2 / (rp1 + rp2)
        if drawn:
            s1 = 0.5
            s2 = 0.5
        else:
            s1 = 1
            s2 = 0
        new_p1 = p1 + k_multiplier * corr_m * k * (s1 - exp_p1)
        new_p2 = p2 + k_multiplier * corr_m * k * (s2 - exp_p2)
        return new_p1, new_p2

    def calc_matches(self, matches):
        #Winner_ID, Loser_ID, Date (optional), Round (optional), margin (optional)

        for match in matches:
            #winner_rating = self.ratings.setdefault(match[0], self.starting_rating)
            winner_rating = self.ratings.get(match[0], [None])[-1]
            # get the last ranking entry
            if winner_rating is None:
                winner_rating = self.starting_rating
                self.ratings.update({match[0]:[winner_rating]})
                if len(match) > 2:
                    self.rating_dates.update(
                        {match[0]:[datetime.strptime(match[2], '%Y-%m-%d')]})
                    if len(match) > 3:
                        self.rating_round.update(
                            {match[0]:[match[3]]})
            loser_rating = self.ratings.get(match[1], [None])[-1]
            if loser_rating is None:
                loser_rating = self.starting_rating
                self.ratings.update({match[1]:[loser_rating]})
                if len(match) > 2:
                    self.rating_dates.update(
                        {match[1]:[datetime.strptime(match[2], '%Y-%m-%d')]})
                    if len(match) > 3:
                        self.rating_round.update(
                            {match[1]:[match[3]]})
            if len(match) > 4:
                mov = match[4]
            else:
                mov = 0
            winner_rating, loser_rating = self.rate_1vs1(winner_rating,
                                                         loser_rating,
                                                         mov)

            self.ratings[match[0]].append(winner_rating)
            self.ratings[match[1]].append(loser_rating)
            if len(match) > 2:
                self.rating_dates[match[0]].append(
                    datetime.strptime(match[2],'%Y-%m-%d'))
                self.rating_dates[match[1]].append(
                    datetime.strptime(match[2],'%Y-%m-%d'))
                if len(match) > 3:
                    self.rating_round[match[0]].append(match[3])
                    self.rating_round[match[1]].append(match[3])
        #c.close()
        #conn.close()

    def win_probability(self, p1, p2):
        diff = p1 - p2
        p = 1 - 1 / (1 + np.power(10, diff / 400.0))
        return p

    def win_probability_alt(self, p1, p2):
        diff = p1 - p2
        p = 1 - 1 / (1 + np.exp(0.00583 * diff - 0.0505))
        return p

    def k_logistic(self, mean_rank):
        return self.klog_min_k + \
               (self.klog_max_k - self.klog_min_k) * \
               sp.special.expit((self.klog_center - mean_rank) *
                                5 / self.klog_width)

    def plot(self, ID_list, prevent_same_day_matches=True, player_names=None):
        fig = pl.figure()
        ax = pl.subplot()
        for ID in ID_list:
            dayssince2000 = [(dt - datetime(2000, 1, 1)).days for dt in elo.rating_dates[ID]]
            if prevent_same_day_matches:
                for idx in list(range(1,len(dayssince2000))):
                    if dayssince2000[idx - 1] >= dayssince2000[idx]:
                        dayssince2000[idx] = dayssince2000[idx - 1] + 1
            labelstr = player_names[ID] if player_names is not None else str(ID)
            #if player_names is not None:
            #    labelstr = player_names[ID]
            #else:
            #    labelstr = str(ID)
            ax.plot(dayssince2000,elo.ratings[ID], '.--', label=labelstr)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))


#an implementation of the AGA (Bayesian) rating system adopted for squash
#TODO: try MCMC
#TODO: try a player vs player rating change
class AGACalc(object):

    def __init__(self, ratings=None, rating_dates=None, rating_round=None,
                 std=None, starting_rating=5, starting_std=1, group_dates=True):
        if ratings is None:
            self.ratings = {}
        else:
            self.ratings = ratings
        if rating_dates is None:
            self.rating_dates = {}
        else:
            self.rating_dates = rating_dates
        if rating_round is None:
            self.rating_round = {}
        else:
            self.rating_round = rating_round
        if std is None:
            self.std = {}
        else:
            self.std = std
        self.starting_rating = starting_rating
        self.starting_std = starting_std
        self.group_dates = group_dates
        self.sqrt2 = np.sqrt(2)
        self.log2 = np.log(2)

    def calc_matches(self, matches):
        # Winner_ID, Loser_ID, Date, Round (optional), margin (optional)
        #TODO: fancy data structure here to keep track of the players and games
        #maybe a 2d dictionary of [winner][loser]=(date, round, margin)
        #and a 1d dictionary of [player]=(rating, date)


        players = {}  # player is the key, rating is the payload
        games = {}  # winner player is the key, list of loser players are the payload
        game_dates = np.sort(np.unique(matches[:, 2]))
        initial_ratings = np.zeros((len(game_dates), 2))
        initial_std = np.zeros((len(game_dates), 2))
        FIM = np.zeros((len(game_dates), len(game_dates)))
        mov = np.zeros(len(game_dates))
        for date in game_dates:
            gms = matches[date == matches[:, 2], :]
            unique_players = np.sort(np.unique(np.vstack((gms[:, 0], gms[:, 1]))))
            for gmidx in list(range(gms.shape[0])):
                winner_rating = self.ratings.setdefault(gms[gmidx, 0], [self.starting_rating])[-1]
                winner_std = self.std.setdefault(gms[gmidx, 0], [self.starting_std])[-1]
                loser_rating = self.ratings.setdefault(gms[gmidx, 1], [self.starting_rating])[-1]
                loser_std = self.std.setdefault(gms[gmidx, 1], [self.starting_std])[-1]
                initial_ratings[gmidx, :] = [winner_rating, loser_rating]
                initial_std[gmidx, :] = [winner_std, loser_std]
                if matches.shape[1]>3:
                    mov[gmidx] = gms[gmidx, 4]
            unique_ratings = [self.ratings[pl] for pl in unique_players]
            unique_std = [self.std[pl] for pl in unique_players]
            CRLB = self.log_like_match_hess(unique_ratings, unique_ratings, unique_std)
            for idx in list(range(len(CRLB))):
                self.std[unique_players[idx]].append(CRLB[idx])
            gms_std = np.hstack(([self.std[pl] for pl in gms[:, 0]], [self.std[pl] for pl in gms[:, 0]]))
            keys = np.hstack((np.searchsorted(unique_players, gms[:, 0]), np.searchsorted(unique_players, gms[:, 1])))
            ratings = self.rate_group(gms[:, 0:2], gms_std, keys, mov)
            for idx in list(range(len(ratings))):
                self.ratings[unique_players[idx]].append(ratings[idx])


    def rate_group(self, player_ratings, player_std, player_match_idx, mov=0):
        # let's assume the first player in each is the winner
        bnds = ((0, 10),) * player_ratings.size[0]
        ratings = sp.optimize.minimize(self.log_like_match, player_ratings,
                                       args=(player_ratings, player_std,
                                             player_match_idx, mov),
                                       jac=self.log_like_match_jac, bounds=bnds)
        return ratings

    def log_like_match_hess(self, pr, pu, ps, mov=0):
        #no! I need some fancier data structures here!
        pr1 = self.vec2squaremat(pr)
        pr2 = pr1.T
        pu1 = self.vec2squaremat(pu)
        pu2 = pu1.T
        ps1 = self.vec2squaremat(ps)
        ps2 = ps1.T
        mov1 = self.vec2squaremat(mov)
        mov2 = mov1.T
        F = self.dd_log_Pp(ps1 + ps2) + self.dd_log_Pg(pr1, pr2, mov1)
        return np.diagonal(np.pinv(-F))


    def log_like_match_jac(self, pr, pu, ps, key, mov=0):
        return np.vstack((self.d_neg_log_Pp(pr[key, 0], pu[key, 0], ps[key, 0]),
               self.d_neg_log_Pp(pr[key, 1], pu[key, 1], ps[key, 1]),
               self.d_neg_log_Pg(pr[key, 0], pr[key, 1], mov)))

    def log_like_match(self, pr, pu, ps, key, mov=0):
        # let's assume player 1 wins
        return self.neg_log_Pp(pr[key, 0], pu[key, 0], ps[key, 0]) +\
               self.neg_log_Pp(pr[key, 1], pu[key, 1], ps[key, 1]) +\
               self.neg_log_Pg(pr[key, 0], pr[key, 1], mov)

    def neg_log_Pp(self, r, u, s):
        return np.square((r - u) / s) / 2 + np.log(np.sqrt(2 * np.pi))

    def d_neg_log_Pp(self, r, u, s):
        return (u - r) / np.square(s)  # -(r - u) / np.square(s)

    def dd_log_Pp(self, s):
        return -s

    def neg_log_Pg(self, r1, r2, mov=0):
        RD = self.calc_RD(r1, r2, mov)
        spx = self.calc_spx()
        return -np.log(sp.special.erfc( RD / (spx * np.sqrt(2)))) + self.log2

    def d_neg_log_Pg(self, r1, r2, dr1=True, mov=0):
        RD = self.calc_RD(r1, r2, mov)
        spx = self.calc_spx()
        retval = 1 / spx * \
                np.sqrt(2 / np.pi) * \
                1 / sp.special.erfc( -RD / (spx * self.sqrt2)) * \
                np.exp(-np.square(RD) / (2 * np.square(spx)))
        if dr1:
            return retval
        else:
            return -retval

    def dd_log_Pg(self, r1, r2, mov=0):
        RD = self.calc_RD(r1, r2, mov)
        RDsq = np.square(RD)
        spx = self.calc_spx()
        spxsq = np.square(spx)
        expfn = np.exp(-RDsq / (2 * spxsq))
        erffn = sp.special.erfc(-RD / (self.sqrt2 * spx))
        return -np.sqrt(2 / np.pi) * \
               RD / np.power(spx, 3) * \
               expfn / erffn -\
               2 / np.pi *\
               1 / spxsq *\
               np.square(expfn) / np.square(erffn)

    def calc_RD(self, r1, r2, mov):
        return r1 - r2 - self.calc_d(mov)

    def calc_d(self, mov):
        return mov/3  # first guess at a margin of victory

    def calc_spx(self):
        return 1  # first guess at std to balance the ratings

    def vec2squaremat(self, v):
        v = np.atleast_2d(v)
        return np.repeat(v, max(v.shape), np.argmin(v.shape))


    #the following is specific for Go
    def neg_log_Pg_go(self, r1, r2, handicap=0, komi=0, winner=True):
        # I don't think I actually need to use their handicap and komi system...
        RD = self.calc_RD_go(r1, r2, handicap, komi)
        if not winner:  # player 2 wins
            RD *= -1
        spx = self.calc_spx_go(handicap=0, komi=0)
        return -np.log(sp.special.erfc( RD / (spx * np.sqrt(2)))) + np.log(2)

    def calc_RD_go(self, r1, r2, handicap=0, komi=0):
        return r1 - r2 - self.calc_d_go(handicap, komi)

    def calc_d_go(self, handicap=0, komi=0):
        if handicap == 0 or handicap == 1:
            return 0.580 - 0.0757 * komi
        else:
            return handicap - 0.0757 * komi

    def calc_spx_go(self, handicap=0, komi=0):
        if handicap == 0 or handicap == 1:
            return 1.0649 - 0.0021976 * komi + 0.00014984 * np.square(komi)
        else:
            if handicap == 2:
                b = 1.13672
            elif handicap == 3:
                b = 1.18795
            elif handicap == 4:
                b = 1.22841
            elif handicap == 5:
                b = 1.27457
            elif handicap == 6:
                b = 1.31978
            elif handicap == 7:
                b = 1.35881
            elif handicap == 8:
                b = 1.39782
            else:  # handicap == 9
                b = 1.43614
            return -0.0035169 * komi + b
"""

#rate_1vs1(1600, 1400)
#rate_1vs1(1400, 1600)
#win_probability(1600, 1400)
#win_probability(1400, 1600)
elo = EloCalc(mov_thresh=1)
bayes = BayesCalc()

"""
#returns: all player IDs that are not NULL
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

"""
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



conn = sqlite3.connect(str(sys.argv[1]))
c = conn.cursor()

"""
colnames = ['P1_{}'.format(i) for i in basic_player_query_names] +\
           ['P2_{}'.format(i) for i in basic_player_query_names] +\
           ['M1_{}'.format(i) for i in individual_match_query_date_names] +\
           ['M2_{}'.format(i) for i in individual_match_query_date_names] +\
           ['M12_{}'.format(i) for i in shared_match_query_date_names]

def query_player_ID_fast(c):
    #  all_player_ID_query_str
    c.execute(all_player_ID_query_str)
    return list(chain(*c.fetchall()))

def query_player_ID_DB():
    return query_player_ID(c, all_player_ID_query_str)

def query_player_DB1(player_pair):
    return query_player(c, basic_player_query_str, colnames[0:3], 0, player_pair)

def query_player_DB2(player_pair):
    return query_player(c, basic_player_query_str, colnames[3:6], 1, player_pair)

def query_match_DB1(player_pair):
    df = query_match(c, individual_match_query_date_str, colnames[6:20], 0, player_pair)
    return df.mean().to_frame().transpose()
    #return DataFrame(np.atleast_2d(df.mean().values), columns=colnames[6:20])

def query_match_DB2(player_pair):
    df = query_match(c, individual_match_query_date_str, colnames[20:34], 1, player_pair)
    return df.mean().to_frame().transpose()
    #return DataFrame(np.atleast_2d(df.mean().values), columns=colnames[20:34])

def query_match_shared_DB(player_pair):
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

"""

#paired_query = DBQuery(query_player_ID_DB, (query_match_shared_DB,
#                                            query_player_DB1, query_player_DB2,
#                                            query_match_DB1, query_match_DB2),
#                       column_names=colnames)
#paired_query.query_paired_all()
#I need to query individual matches AND keep track if they win or lose!
#print('done!')

paired_query = DBQuery(query_player_ID_fast, query_function_fast,
                       str(sys.argv[1]), fast_colnames)

#player_plot_IDs = list(range(2,10))
player_plot_IDs = [9, 13, 18, 21, 24, 31]
name_dict = paired_query.get_dict_names(player_plot_IDs)
paired_query.query_for_elo()

#ELO!
#elo.calc_matches(paired_query.matches)
#elo.plot(player_plot_IDs, player_names=name_dict)
#pl.show()

#new Elo!
ratingdb_fname = 'ratingdb.sqlite'
try:
    os.remove(ratingdb_fname)
except OSError:
    pass
rating_DB = ratingDB(matches=paired_query.matches, database_name=ratingdb_fname)
elo.calc_ratings(rating_DB)
rating_DB.PlotElo(player_plot_IDs, player_names=name_dict)
pl.show()
rating_DB.Close()
print('Elo done')

#game permutations and indicies
#tuple(map('{0:02b}'.format,range(4)))
#tuple(map(lambda x:tuple(map(int,x)),map('{0:02b}'.format,range(4))))
#the 02b means there are 2 games, and the 4 is the possible permutations or 2^ngames
#'{{0:0{}b}}'.format(2)
#tuple(map(lambda x:tuple(map(int,x)),map('{{0:0{}b}}'.format(ngames).format,range(2**ngames))))
def bracket_permutations(ngames):
    if ngames>=1:
        b = (tuple(map(lambda x:tuple(map(int,x)),map('{{0:0{}b}}'.format(ngames).format,range(2**ngames)))),)
        for p in map(lambda x:tuple(map(int,x)),map('{{0:0{}b}}'.format(ngames).format,range(2**ngames))):
            bp = bracket_permutations(ngames//2)
            b = b+p+bp
        return (b,)
    else:
        return (None,)

tournament={}
base=(('A','B'),('C','D'),('E','F'),('G','H'))
tournament[base]=base
def bracket_gen(base):
    #sublen = len(base[0])
    ngames = len(base)
    if ngames>1:
        b = tuple(map(lambda x:tuple(map(int,x)),map('{{0:0{}b}}'.format(ngames).format,range(2**ngames))))
        t = tuple(map(lambda y: tuple(map(lambda x: x[1][x[0]], zip(y, base))), b))
        #if len(t[0])<2:
        #    return (base,)+t[0]+(base,)+t[1]
        #else:
        j = tuple(map(lambda x: tuple(zip(x[0::2], x[1::2])), t))
        o = (tuple(map(bracket_gen,j)),)
        tournament[base] = j
        return o
        #return (base,) + o
    else:
        o = ((base[0][0],),)+((base[0][1],),)
        tournament[base] = o
        return o
        #return (base,) + o
        #return b+tuple(map(bracket_gen,map(lambda y: tuple(map(lambda x: x[1][x[0]], zip(y, base)))), b))
        #for i in range(len(b)):  #can this be mapped too?
        #    tuple(map(lambda x: base[x][i], b[i]))  #not quite, should be [0,2]
        #    tuple(map(lambda x: tuple(map(lambda y: base[y], x)), b))
        #    tuple(map(lambda x: x[1][0], zip(b[0], base)))
        #    tuple(map(lambda y: tuple(map(lambda x: x[1][x[0]], zip(y, base))),b))
        #tuple(map(lambda y: (tuple(map(lambda x: x[1][x[0]], zip(y, base))),), b))
        #    bp = bracket_gen(base[i][b[i]])
        #    return bp
        #for p in map(lambda x:tuple(map(int,x)),map('{{0:0{}b}}'.format(ngames).format,range(2**ngames))):
        #    bp = bracket_permutations(ngames//2)
        #    b = b+p+bp
        #return b
    #else:
    #    return base[0]

bracket=bracket_permutations(2)
bracket_named=bracket_gen(base)


#bayes!
bayes.calc_matches(paired_query.matches)

paired_query.query_data()
print('done!')
idx = 50


predictor_PCA = PCA()
predictor_PCA.fit(paired_query.X)
print(predictor_PCA.explained_variance_ratio_)
predictor_PCA.set_params(n_components=2)
pred_X = predictor_PCA.transform(paired_query.X)

predictor_SVC = SquashPredict(paired_query.X, np.round(paired_query.y),
                              method=svm.SVC(probability=True))
predictor_SVC.fit()
print(predictor_SVC.score())
results_SVC = predictor_SVC.comparison(return_probability=True)
print(results_SVC[idx, :])

predictor_SVR = SquashPredict(paired_query.X, paired_query.y, method=svm.SVR())
predictor_SVR.fit()
print(predictor_SVR.score())
results_SVR = predictor_SVR.comparison()
print(results_SVR[idx, :])
print('done!')

exit()


full_query_str = "SELECT " \
                 "Match.Winner_ID," \
                 "Match.Loser_ID," \
                 "Match.Event," \
                 "Match.Date," \
                 "Match.Country," \
                 "Match.Round," \
                 "Match.Number_of_Games," \
                 "Match.Duration," \
                 "Match.Game1_Winner_Score," \
                 "Match.Game1_Loser_Score," \
                 "Match.Game2_Winner_Score," \
                 "Match.Game2_Loser_Score," \
                 "Match.Game3_Winner_Score," \
                 "Match.Game3_Loser_Score," \
                 "Match.Game4_Winner_Score," \
                 "Match.Game4_Loser_Score," \
                 "Match.Game5_Winner_Score," \
                 "Match.Game5_Loser_Score," \
                 "Match.Retired," \
                 "Match.PSA," \
                 "Winner.Country," \
                 "Winner.Current_Rank," \
                 "Winner.Highest_Rank," \
                 "Winner.DOB," \
                 "Winner.Male," \
                 "Winner.Height," \
                 "Winner.Weight," \
                 "Winner.Right_Handed," \
                 "Winner.Refresh_Date, " \
                 "Loser.Country," \
                 "Loser.Current_Rank," \
                 "Loser.Highest_Rank," \
                 "Loser.DOB," \
                 "Loser.Male," \
                 "Loser.Height," \
                 "Loser.Weight," \
                 "Loser.Right_Handed," \
                 "Loser.Refresh_Date " \
                 "FROM Match JOIN Player AS Winner " \
                 "ON Match.Winner_ID=Winner.Player_ID " \
                 "JOIN Player AS Loser " \
                 "ON Match.Loser_ID=Loser.Player_ID"

basic_query_str = "SELECT " \
                 "Match.Winner_ID," \
                 "Match.Loser_ID," \
                 "Match.Number_of_Games," \
                 "Match.Duration," \
                 "coalesce(Match.Game1_Winner_Score,0)," \
                 "coalesce(Match.Game1_Loser_Score,0)," \
                 "coalesce(Match.Game2_Winner_Score,0)," \
                 "coalesce(Match.Game2_Loser_Score,0)," \
                 "coalesce(Match.Game3_Winner_Score,0)," \
                 "coalesce(Match.Game3_Loser_Score,0)," \
                 "coalesce(Match.Game4_Winner_Score,0)," \
                 "coalesce(Match.Game4_Loser_Score,0)," \
                 "coalesce(Match.Game5_Winner_Score,0)," \
                 "coalesce(Match.Game5_Loser_Score,0)," \
                 "Match.Retired," \
                 "Match.PSA," \
                 "Winner.Male," \
                 "Winner.Height," \
                 "Winner.Weight," \
                 "Loser.Male," \
                 "Loser.Height," \
                 "Loser.Weight " \
                 "FROM Match JOIN Player AS Winner " \
                 "ON Match.Winner_ID=Winner.Player_ID " \
                 "JOIN Player AS Loser " \
                 "ON Match.Loser_ID=Loser.Player_ID " \
                 "WHERE Match.Number_of_Games>0 " \
                 "AND Winner.Male IS NOT NULL " \
                 "AND Loser.Male IS NOT NULL " \
                 "AND Match.Duration IS NOT NULL " \
                 "AND Winner.Height IS NOT NULL " \
                 "AND Winner.Weight IS NOT NULL " \
                 "AND Loser.Height IS NOT NULL " \
                 "AND Loser.Weight IS NOT NULL "


calculation_query_str = "SELECT " \
                 "Match.Winner_ID," \
                 "Match.Loser_ID," \
                 "Match.Number_of_Games," \
                 "Match.Duration," \
                 "coalesce(Match.Game1_Winner_Score,0) + coalesce(Match.Game1_Loser_Score,0)," \
                 "coalesce(Match.Game1_Winner_Score,0) - coalesce(Match.Game1_Loser_Score,0)," \
                 "coalesce(Match.Game2_Winner_Score,0) + coalesce(Match.Game2_Loser_Score,0)," \
                 "coalesce(Match.Game2_Winner_Score,0) - coalesce(Match.Game2_Loser_Score,0)," \
                 "coalesce(Match.Game3_Winner_Score,0) + coalesce(Match.Game3_Loser_Score,0)," \
                 "coalesce(Match.Game3_Winner_Score,0) - coalesce(Match.Game3_Loser_Score,0)," \
                 "coalesce(Match.Game4_Winner_Score,0) + coalesce(Match.Game4_Loser_Score,0)," \
                 "coalesce(Match.Game4_Winner_Score,0) - coalesce(Match.Game4_Loser_Score,0)," \
                 "coalesce(Match.Game5_Winner_Score,0) + coalesce(Match.Game5_Loser_Score,0)," \
                 "coalesce(Match.Game5_Winner_Score,0) - coalesce(Match.Game5_Loser_Score,0)," \
                 "Match.Retired," \
                 "Match.PSA," \
                 "Winner.Male," \
                 "Winner.Height," \
                 "Winner.Weight," \
                 "Loser.Male," \
                 "Loser.Height," \
                 "Loser.Weight " \
                 "FROM Match JOIN Player AS Winner " \
                 "ON Match.Winner_ID=Winner.Player_ID " \
                 "JOIN Player AS Loser " \
                 "ON Match.Loser_ID=Loser.Player_ID " \
                 "WHERE Match.Number_of_Games>0 " \
                 "AND Winner.Male IS NOT NULL " \
                 "AND Loser.Male IS NOT NULL " \
                 "AND Match.Duration IS NOT NULL " \
                 "AND Winner.Height IS NOT NULL " \
                 "AND Winner.Weight IS NOT NULL " \
                 "AND Loser.Height IS NOT NULL " \
                 "AND Loser.Weight IS NOT NULL "

calculation_ID_query_str = "SELECT " \
                 "Match.Winner_ID," \
                 "Match.Loser_ID," \
                 "Match.Number_of_Games," \
                 "Match.Duration," \
                 "coalesce(Match.Game1_Winner_Score,0) + coalesce(Match.Game1_Loser_Score,0)," \
                 "coalesce(Match.Game1_Winner_Score,0) - coalesce(Match.Game1_Loser_Score,0)," \
                 "coalesce(Match.Game2_Winner_Score,0) + coalesce(Match.Game2_Loser_Score,0)," \
                 "coalesce(Match.Game2_Winner_Score,0) - coalesce(Match.Game2_Loser_Score,0)," \
                 "coalesce(Match.Game3_Winner_Score,0) + coalesce(Match.Game3_Loser_Score,0)," \
                 "coalesce(Match.Game3_Winner_Score,0) - coalesce(Match.Game3_Loser_Score,0)," \
                 "coalesce(Match.Game4_Winner_Score,0) + coalesce(Match.Game4_Loser_Score,0)," \
                 "coalesce(Match.Game4_Winner_Score,0) - coalesce(Match.Game4_Loser_Score,0)," \
                 "coalesce(Match.Game5_Winner_Score,0) + coalesce(Match.Game5_Loser_Score,0)," \
                 "coalesce(Match.Game5_Winner_Score,0) - coalesce(Match.Game5_Loser_Score,0)," \
                 "Match.Retired," \
                 "Match.PSA," \
                 "Winner.Male," \
                 "Winner.Height," \
                 "Winner.Weight," \
                 "Loser.Male," \
                 "Loser.Height," \
                 "Loser.Weight " \
                 "FROM Match JOIN Player AS Winner " \
                 "ON Match.Winner_ID=Winner.Player_ID " \
                 "JOIN Player AS Loser " \
                 "ON Match.Loser_ID=Loser.Player_ID " \
                 "WHERE Match.Number_of_Games>0 " \
                 "AND ((Match.Winner_ID=? " \
                 "AND Match.Loser_ID=?) " \
                 "OR (Match.Loser_ID=? " \
                 "AND Match.Winner_ID=?)) " \
                 "AND Winner.Male IS NOT NULL " \
                 "AND Loser.Male IS NOT NULL " \
                 "AND Match.Duration IS NOT NULL " \
                 "AND Winner.Height IS NOT NULL " \
                 "AND Winner.Weight IS NOT NULL " \
                 "AND Loser.Height IS NOT NULL " \
                 "AND Loser.Weight IS NOT NULL "

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

individual_match_query_str = "SELECT " \
                 "Number_of_Games, " \
                 "Duration, " \
                 "coalesce(Game1_Winner_Score,0) + coalesce(Game1_Loser_Score,0), " \
                 "coalesce(Game1_Winner_Score,0) - coalesce(Game1_Loser_Score,0), " \
                 "coalesce(Game2_Winner_Score,0) + coalesce(Game2_Loser_Score,0), " \
                 "coalesce(Game2_Winner_Score,0) - coalesce(Game2_Loser_Score,0), " \
                 "coalesce(Game3_Winner_Score,0) + coalesce(Game3_Loser_Score,0), " \
                 "coalesce(Game3_Winner_Score,0) - coalesce(Game3_Loser_Score,0), " \
                 "coalesce(Game4_Winner_Score,0) + coalesce(Game4_Loser_Score,0), " \
                 "coalesce(Game4_Winner_Score,0) - coalesce(Game4_Loser_Score,0), " \
                 "coalesce(Game5_Winner_Score,0) + coalesce(Game5_Loser_Score,0), " \
                 "coalesce(Game5_Winner_Score,0) - coalesce(Game5_Loser_Score,0), " \
                 "PSA " \
                 "FROM Match " \
                 "WHERE Number_of_Games>0 " \
                 "AND (Winner_ID=? " \
                 "OR Loser_ID=?) " \
                 "AND Duration IS NOT NULL " \
                 "AND Retired=0"

shared_match_query_str = "SELECT " \
                 "Winner_ID, " \
                 "Number_of_Games, " \
                 "Duration, " \
                 "coalesce(Game1_Winner_Score,0) + coalesce(Game1_Loser_Score,0), " \
                 "coalesce(Game1_Winner_Score,0) - coalesce(Game1_Loser_Score,0), " \
                 "coalesce(Game2_Winner_Score,0) + coalesce(Game2_Loser_Score,0), " \
                 "coalesce(Game2_Winner_Score,0) - coalesce(Game2_Loser_Score,0), " \
                 "coalesce(Game3_Winner_Score,0) + coalesce(Game3_Loser_Score,0), " \
                 "coalesce(Game3_Winner_Score,0) - coalesce(Game3_Loser_Score,0), " \
                 "coalesce(Game4_Winner_Score,0) + coalesce(Game4_Loser_Score,0), " \
                 "coalesce(Game4_Winner_Score,0) - coalesce(Game4_Loser_Score,0), " \
                 "coalesce(Game5_Winner_Score,0) + coalesce(Game5_Loser_Score,0), " \
                 "coalesce(Game5_Winner_Score,0) - coalesce(Game5_Loser_Score,0), " \
                 "PSA, " \
                 "Date " \
                 "FROM Match " \
                 "WHERE Number_of_Games>0 " \
                 "AND ((Winner_ID=? " \
                 "AND Loser_ID=?) " \
                 "OR (Loser_ID=? " \
                 "AND Winner_ID=?)) " \
                 "AND Duration IS NOT NULL " \
                 "AND Retired=0"

# TODO: 1. tests with a small, fake DB
#  2. is from host country
# 3. player rankings! i think i may do this with elo
# 4. we don't need to check so many queries, actually just m12!

#set up the database
conn = sqlite3.connect(str(sys.argv[1]))
c = conn.cursor()

# [Player 1 info, Player 1 general match history, Player 2 info, Player 2 general match history, Shared match history]
# I need a ranking number that shows how good the opponents are, e.g. do they
#   generally beat higher ranked players?
c.execute("SELECT Player_ID FROM Player " \
          "WHERE Male IS NOT NULL " \
          "AND Height IS NOT NULL " \
          "AND Weight IS NOT NULL")
all_players = list(chain(*c.fetchall()))
X = np.array([], dtype=np.int)
y = np.array([], dtype=np.float)
id_key = np.array([], dtype=np.int)
for player_pair in combinations(all_players, 2):
    c.execute("SELECT Male, Height, Weight FROM Player WHERE  Player_ID=?",
              (player_pair[0],))
    p1 = np.asarray(c.fetchone(), dtype=np.int)
    c.execute("SELECT Male, Height, Weight FROM Player WHERE  Player_ID=?",
              (player_pair[1],))
    p2 = np.asarray(c.fetchone(), dtype=np.int)
    c.execute(individual_match_query_date_str, (player_pair[0],)*3)
    m1 = np.asarray(c.fetchall(), dtype=np.int)
    c.execute(individual_match_query_date_str, (player_pair[1],)*3)
    m2 = np.asarray(c.fetchall(), dtype=np.int)
    c.execute(shared_match_query_date_str, player_pair*2)
    m12 = np.asarray(c.fetchall(), dtype=np.int)
    if m1.size > 0 and m2.size > 0 and m12.size > 0:
        #match_dates = m12[:, -1]
        Winner_ID = m12[:, 0]
        # a winner ID method, works well!
        # y_new = (np.sum(Winner_ID == player_pair[0]) -\
        #         np.sum(Winner_ID == player_pair[1])) /\
        #         Winner_ID.size
        # a game based method, not as well, but interesting results
        y_new = 6 - m12[:, 1]
        y_new[Winner_ID == player_pair[1]] *= -1
        y_new = y_new.mean()

        m12 = m12[:, 1:-1]
        #if m1.any() and m2.any() and m12.any():
        X_new = np.atleast_2d(np.concatenate(
            (p1, m1.mean(axis=0), np.atleast_1d(m1.shape[0]), p2,
             m2.mean(axis=0), np.atleast_1d(m2.shape[0]), m12.mean(axis=0),
             np.atleast_1d(m12.shape[0])), axis=0))
        if X.any():
            X = np.concatenate((X, X_new), axis=0)
            y = np.concatenate((y, np.atleast_1d(y_new)), axis=0)
            id_key = np.concatenate((id_key, np.atleast_2d(player_pair)),
                                    axis=0)
        else:
            X = X_new
            y = np.atleast_1d(y_new)
            id_key = np.atleast_2d(player_pair)
        #print(m12)

    #c.execute(calculation_ID_query_str, player_pair*2)
    #matches = np.asarray(c.fetchall(), dtype=np.int)
    #if matches.any():
        #need to calculate the head-to-head of these players!
    #    matches = matches
    #    if all_matches.any():
    #        all_matches = np.concatenate((all_matches, matches), axis=0)
    #    else:
    #        all_matches = matches


#this doesnt work!
#c.execute(calculation_query_str)
#matches = c.fetchall()
#matches = np.asarray(matches, dtype=np.int)

#NEED TO PREPROCESS THE DATA HERE!


SVR = svm.SVR()
SVR.fit(X, y)
y_rounded = np.round(y)
y_rounded[y_rounded == 0] = 1  #which winner to pick?
SVC = svm.SVC(probability=True)
SVC.fit(X, y_rounded)

#SVR: let's see how it is!
idx = 50
print(SVR.score(X, y))
results_SVR = np.array([SVR.predict(X), y]).T
print(results_SVR[idx, :])

#SVC: let's see how it is!
print(SVC.score(X, y_rounded))
results_SVC = np.array([SVC.predict(X), y_rounded]).T
results_SVC = np.concatenate((results_SVC, SVC.predict_proba(X)), axis=1)
print(results_SVC[idx, :])

c.close()
conn.close()