# from bs4 import BeautifulSoup
from datetime import datetime
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




class EloCalc(object):

    def __init__(self, mov_thresh=5,
                 klog_center=2200, klog_width=200, klog_min_k=10, klog_max_k=40,
                 ratings=None, rating_dates=None, rating_round=None,
                 matches=None, starting_rating=1500):
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
        if matches is None:
            self.matches = {}
        else:
            self.matches = matches
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

    def calc_ratings(self, rating_DB):
        #this is a little slow, could maybe speed up with database tricks
        matches = rating_DB.GetMatches() #need to avoid previously rated matches!
        #rating_DB.BeginTransaction()
        for match in matches:
            dtmatch=datetime.strptime(match[2], '%Y-%m-%d')
            # enforce causality: we don't want to rate games from the past
            if datetime.strptime(rating_DB.GetLatestEloDate(match[0]),
                                 '%Y-%m-%d') <= dtmatch and \
                            datetime.strptime(
                                rating_DB.GetLatestEloDate(match[1]),
                                  '%Y-%m-%d') <= dtmatch:
                #match = (Winner_ID, Loser_ID, Date, Round (fake), MOV (margin of victory)
                winner_rating = rating_DB.GetLatestEloRating(match[0])
                loser_rating = rating_DB.GetLatestEloRating(match[1])
                #mov = match[4]
                winner_rating, loser_rating = self.rate_1vs1(winner_rating,
                                                             loser_rating,
                                                             match[4])
                rating_DB.SetEloRating(match[0], match[2], match[3], winner_rating)
                rating_DB.SetEloRating(match[1], match[2], match[3], loser_rating)
        rating_DB.Commit()
        #rating_DB.EndTransaction()


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
            dayssince2000 = [(dt - datetime(2000, 1, 1)).days for dt in self.rating_dates[ID]]
            if prevent_same_day_matches:
                for idx in list(range(1,len(dayssince2000))):
                    if dayssince2000[idx - 1] >= dayssince2000[idx]:
                        dayssince2000[idx] = dayssince2000[idx - 1] + 1
            labelstr = player_names[ID] if player_names is not None else str(ID)
            #if player_names is not None:
            #    labelstr = player_names[ID]
            #else:
            #    labelstr = str(ID)
            ax.plot(dayssince2000, self.ratings[ID], '.--', label=labelstr)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))


#an implementation of Bayesian (like AGA) rating system adopted for squash
#TODO: try MCMC
#TODO: try a player vs player rating change
class BayesCalc(object):

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

    def calc_ratings(self, rating_DB):
        #here's a rough outline of how I think it should work
        match_dates = rating_DB.GetUniqueMatchDates()
        for date in match_dates:
            matches = rating_DB.GetMatchesOnDate(date)
            ratings = rating_DB.GetRatingsOnDate(date)
            CRLB = self.log_like_match_hess(unique_ratings, unique_ratings,
                                            unique_std)
            ratings = self.rate_group(gms[:, 0:2], gms_std, keys, mov)
            rating_DB.SetBayesRatings(ratings)


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