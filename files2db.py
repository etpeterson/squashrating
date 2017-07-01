from bs4 import BeautifulSoup
from datetime import datetime
import dateparser
# from pycountry import pycountry
import re
import sys
import os
import sqlite3


# cd /Users/epeterson/Documents/RaspberryPi/squash
# python3 files2db.py squashfolder/players testdb.sqlite

def parse_match(match):
    # [date, opponent, win/loss, event, location, round, result, psa?]
    # print(len(match))
    if len(match) != 8:
        return None
    try:  # original wget
        Date = datetime.strptime(match[0], '%b %Y').strftime('%Y-%m') + '-1'
    except ValueError:  # archive from base folder
        Date = datetime.strptime(match[0], '%b %y').strftime('%Y-%m') + '-1'
    # because we don't know the day let's just say it's day 1
    # print(match[1])
    namelist = match[1].split()
    Opponent_FName = namelist[0]
    Opponent_LName = namelist[-1]
    Opponent_MName = " ".join(namelist[1:len(namelist) - 1])
    Result = match[2]
    Event = match[3]
    # work on countries
    Country = match[4]
    Round = None if match[5] == '-' else match[5]
    # if match[5] == '-':
    #	Round = None
    # else:
    #	Round = match[5]
    Scores = [list(map(int, game.split('-')))
              for game in re.findall('\d*-\d*', match[6])]
    if len(Scores) == 0:
        Scores = None
    Duration = re.findall('\(\d*m\)', match[6])
    if len(Duration) == 1:
        Duration = int(Duration[0][1:-2])
    else:
        Duration = None
    Retired = match[6].find('ret.') > -1
    # print(Scores)
    # print(Duration)
    # print(Retired)
    PSA = True if match[7] == 'Y' else False
    return {'Date': Date, 'Opponent_First_Name': Opponent_FName,
            'Opponent_Middle_Name': Opponent_MName,
            'Opponent_Last_Name': Opponent_LName, 'Duration': Duration,
            'Retired': Retired,
            'Result': Result, 'Event': Event, 'Country': Country,
            'Round': Round, 'Scores': Scores, 'PSA': PSA}


def parse_matches(fname):
    matches = []
    try:  # utf-8
        soup = BeautifulSoup(open(fname, encoding='UTF-8'), "html.parser")
    except UnicodeDecodeError:
        soup = BeautifulSoup(open(fname, encoding='ISO-8859-1'), "html.parser")
    try:  # original
        match_table = soup.find_all(id="match_summary_table")[0]
        match_table_body = match_table.find('tbody')
        rows = match_table_body.find_all('tr')
    except IndexError:  # archive from base
        try:
            match_table_body = soup.find_all(id="matchsummary_table")[0]
            rows = match_table_body.find_all('tr')
        except IndexError:  # empty matches
            rows = []
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        match = parse_match(cols)
        if match:
            matches.append(parse_match(cols))
        # cols = [ele for ele in cols if ele] # Get rid of empty values
        # data.append([ele for ele in cols if ele]) # Get rid of empty values
    # print(soup.title)
    return matches


def check_keys(key_string, value_string):
    if key_string == 'Born':
        return ['DOB',
                dateparser.parse(value_string).strftime('%Y-%m-%d')]
                #datetime.strptime(value_string,
                #                  '%d %b %Y ').strftime('%Y-%m-%d')]
    elif key_string == 'Gender':
        return ['Male', True if value_string == 'Male' else False]
    elif key_string == 'Country':
        return [key_string, value_string]
    elif key_string == 'Height':
        try:
            return [key_string, int(value_string.split('cm')[0])]
        except ValueError:
            return None
    elif key_string == 'Weight':
        try:
            return [key_string, int(value_string.split('kg')[0])]
        except ValueError:
            return None
    elif key_string == 'Plays':
        return ['Right_Handed',
                True if value_string == 'Right-handed' else False]
    elif key_string == 'Highest WR' or key_string == 'Highest World Ranking':
        try:
            return ['Highest_Rank', int(value_string)]
        except ValueError:
            return None
    elif key_string == 'World Ranking' or key_string == 'Current World Ranking':
        try:
            return ['Current_Rank', int(value_string)]
        except ValueError:
            return None
    elif key_string == 'First Name':
        return ['First_Name', value_string]
    elif key_string == 'Middle Name':
        return ['Middle_Name', value_string]
    elif key_string == 'Last Name':
        return ['Last_Name', value_string]
    elif key_string == 'Refresh Date':
        return ['Refresh_Date', value_string]
    else:
        return None


def parse_player(fname):
    player_dict = {}
    try:  # utf-8
        soup = BeautifulSoup(open(fname, encoding='UTF-8'), "html.parser")
    except UnicodeDecodeError:
        soup = BeautifulSoup(open(fname, encoding='ISO-8859-1'), "html.parser")

    name = soup.find('h1')
    if name is None:  # if there's no name what can we do?
        return None
    name = soup.find('h1').text.strip().split("(")
    # print(name[0])
    namelist = name[0].split()
    fname = namelist[0]
    lname = namelist[-1]
    mname = " ".join(namelist[1:len(namelist) - 1])
    check_first_name = check_keys('First Name', fname)
    if check_first_name:
        player_dict.update({check_first_name[0]: check_first_name[1]})
    check_middle_name = check_keys('Middle Name', mname)
    if check_middle_name:
        player_dict.update({check_middle_name[0]: check_middle_name[1]})
    check_last_name = check_keys('Last Name', lname)
    if check_last_name:
        player_dict.update({check_last_name[0]: check_last_name[1]})
        # fname, lname, _ = name[0].split(" ")

    #try dateparser
    try:
        refresh_date = dateparser.parse(
            soup.find(id='current_date_time').text.strip()).strftime('%Y-%m-%d')
    except AttributeError:
        try:
            refresh_date = dateparser.parse(
                soup.find(id='current_date').text.strip()).strftime('%Y-%m-%d')
        except AttributeError:
            refresh_date = dateparser.parse(
                soup.find_all(class_='navinfo')[1].text.strip())\
                .strftime('%Y-%m-%d')
    player_clean = check_keys('Refresh Date', refresh_date)
    #TODO: get rid of these trys an see if soup returns None
    # try:  # original direct download
    #     try:  # original
    #         refresh_date = datetime.strptime(
    #             soup.find(id='current_date_time').text.strip(),
    #             '%I:%M %p %d %b %Y').strftime('%Y-%m-%d')
    #     except ValueError:
    #         try:  # no space between am/pm and day
    #             refresh_date = datetime.strptime(
    #                 soup.find(id='current_date_time').text.strip(),
    #                 '%I:%M %p%d %b %Y').strftime('%Y-%m-%d')
    #         except ValueError:  # with a space but also has time zone
    #             refresh_date = datetime.strptime(
    #                 soup.find(id='current_date_time').text.strip(),
    #                 '%I:%M %p (%Z)%d %b %Y').strftime('%Y-%m-%d')
    # except AttributeError:  # archive named file in players folder
    #     try:
    #         refresh_date = datetime.strptime(
    #             soup.find(id='current_date').text.strip(),
    #             '%d %b %Y').strftime('%Y-%m-%d')
    #     except AttributeError:  # archive file in base folder
    #         refresh_date = datetime.strptime(
    #             soup.find_all(class_='navinfo')[1].text.strip(),
    #             '%d %b %Y').strftime('%Y-%m-%d')
    # print(soup.find_all(id="world_ranking"))
    world_ranking = soup.find_all(id="world_ranking")
    if len(world_ranking) == 0:
        world_ranking = None
    else:
        world_ranking = world_ranking[0].text.strip()
        world_ranking = check_keys('World Ranking', world_ranking)
        if world_ranking:
            player_dict.update({world_ranking[0]: world_ranking[1]})
    # world_ranking = int(soup.find_all(id="world_ranking")[0].text.strip())
    try:  # original direct download
        country = soup.find('h1').find(style='color:#444444;').text.strip("()")
    except AttributeError:  # archive file in base folder
        try:
            country = soup.find(style='width:100%').find_all('td')[3].text
        except AttributeError:
            country = None
    if country:
        player_dict.update({'Country': country})

    player_table = soup.find_all(class_="row")
    if len(player_table) == 0:
        player_table = soup.find(style='width:100%')
        if player_table is not None:
            player_table = player_table.find_all('tr')
    if player_table is not None:
        for row in player_table:
            delem = row.find_all("span")
            if len(delem) == 0:
                delem2 = row.text.split(':')
                if len(delem2) == 2:
                    player_key, player_value = [e.strip() for e in delem2]
                else:
                    player_key = None
                    player_value = None
            else:
                player_key = delem[0].text.strip(":")
                player_value = delem[1].text.strip().split("(")[0]
            player_clean = check_keys(player_key, player_value)
            if player_clean:
                player_dict.update({player_clean[0]: player_clean[1]})
            # player_dict[delem[0].text.strip(":") = delem[1].text.strip().split("(")[0]
            # datetime.strptime(datestring,'%d %b %Y ') datetime parsing

        if player_clean:
            player_dict.update({player_clean[0]: player_clean[1]})
    return player_dict


player_folder = str(sys.argv[1])
print(player_folder)

# set up the database
conn = sqlite3.connect(str(sys.argv[2]))
c = conn.cursor()
c.execute(
    "CREATE TABLE IF NOT EXISTS Player("
    "Player_ID INTEGER PRIMARY KEY AUTOINCREMENT,"
    "First_Name TEXT,"
    "Middle_Name TEXT,"
    "Last_Name TEXT,"
    "Country TEXT,"
    "Current_Rank INTEGER,"
    "Highest_Rank INTEGER,"
    "DOB TEXT,"
    "Male INTEGER,"
    "Height INTEGER,"
    "Weight INTEGER,"
    "Right_Handed INTEGER,"
    "Refresh_Date TEXT);")
# note that weight and current ranking are subject to the refresh date
c.execute(
    "CREATE TABLE IF NOT EXISTS Match("
    "Match_ID INTEGER PRIMARY KEY AUTOINCREMENT,"
    "Winner_ID INTEGER,"
    "Loser_ID INTEGER,"
    "Event TEXT,"
    "Date TEXT,"
    "Country TEXT,"
    "Round TEXT,"
    "Number_of_Games INTEGER,"
    "Duration INTEGER,"
    "Game1_Winner_Score INTEGER,"
    "Game1_Loser_Score INTEGER,"
    "Game2_Winner_Score INTEGER,"
    "Game2_Loser_Score INTEGER,"
    "Game3_Winner_Score INTEGER,"
    "Game3_Loser_Score INTEGER,"
    "Game4_Winner_Score INTEGER,"
    "Game4_Loser_Score INTEGER,"
    "Game5_Winner_Score INTEGER,"
    "Game5_Loser_Score INTEGER,"
    "Retired INTEGER,"
    "PSA INTEGER);")
# c.execute("COMMIT")

player_select_str = "SELECT Player_ID " +\
                      "FROM Player " +\
                      "WHERE First_Name=? " +\
                      "AND Middle_Name=? " +\
                      "AND Last_Name=?"

for root, dirs, files in os.walk(player_folder):
    for name in files:
        fname = os.path.join(root, name)
        print(fname)
        player_dict = parse_player(fname)
        if player_dict is not None:
            match_list = parse_matches(fname)
            # print(match_list)
            #print(player_dict)
            if match_list is not None:
                nmatches = len(match_list)
            else:
                nmatches = 0

            player_select_tup = (
                          player_dict.get('First_Name'),
                          player_dict.get('Middle_Name'),
                          player_dict.get('Last_Name'))
            #print([player_dict['First_Name'], player_dict['Middle_Name'],
            #       player_dict['Last_Name'], nmatches])
            # insert the player into the database
            c.execute(player_select_str, player_select_tup)
            Player_ID = c.fetchone()
            if Player_ID is None:
                c.execute(
                    "INSERT INTO Player("
                    "First_Name,"
                    "Middle_Name,"
                    "Last_Name,"
                    "Country,"
                    "Current_Rank,"
                    "Highest_Rank,"
                    "DOB,"
                    "Male,"
                    "Height,"
                    "Weight,"
                    "Right_Handed,"
                    "Refresh_Date)"
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (player_dict.get('First_Name'),
                     player_dict.get('Middle_Name'),
                     player_dict.get('Last_Name'),
                     player_dict.get('Country'),
                     player_dict.get('Current_Rank'),
                     player_dict.get('Highest_Rank'),
                     player_dict.get('DOB'),
                     player_dict.get('Male'),
                     player_dict.get('Height'),
                     player_dict.get('Weight'),
                     player_dict.get('Right_Handed'),
                     player_dict.get('Refresh_Date')))
                c.execute(player_select_str, player_select_tup)
                Player_ID = c.fetchone()[0]
            else:  # player already exists
                c.execute(
                    "UPDATE Player "
                    "SET Country=?, "
                    "Current_Rank=?, "
                    "Highest_Rank=?, "
                    "DOB=?, "
                    "Male=?, "
                    "Height=?, "
                    "Weight=?, "
                    "Right_Handed=?, "
                    "Refresh_Date=? "
                    "WHERE Player_ID=?",
                    (player_dict.get('Country'),
                     player_dict.get('Current_Rank'),
                     player_dict.get('Highest_Rank'),
                     player_dict.get('DOB'),
                     player_dict.get('Male'),
                     player_dict.get('Height'),
                     player_dict.get('Weight'),
                     player_dict.get('Right_Handed'),
                     player_dict.get('Refresh_Date'),
                     Player_ID[0]))
                Player_ID = Player_ID[0]
            # c.execute("SELECT Player_ID FROM Player WHERE First_Name=? AND Middle_Name=? AND Last_Name=?", (player_dict['First_Name'], player_dict['Middle_Name'], player_dict['Last_Name']))
            # Player_ID = c.fetchone()[0]
            # insert the matches into the database
            #print(match_list)
            # c.execute("BEGIN")
            nmatches_added = 0
            for match_dict in match_list:
                #print(match_dict)
                ngames = len(match_dict['Scores']) if match_dict['Scores'] else 0
                opponent_select_tup = (
                          match_dict.get('Opponent_First_Name'),
                          match_dict.get('Opponent_Middle_Name'),
                          match_dict.get('Opponent_Last_Name'))
                c.execute(player_select_str, opponent_select_tup)
                Opponent_ID = c.fetchone()
                if Opponent_ID is None:  # temporarily insert the opponent name into the player table so we can refer to it
                    c.execute("INSERT INTO "
                              "Player(First_Name, Middle_Name, Last_Name) "
                              "VALUES(?, ?, ?)", opponent_select_tup)
                    c.execute(player_select_str, opponent_select_tup)
                    Opponent_ID = c.fetchone()
                Opponent_ID = Opponent_ID[0]
                if match_dict['Scores'] is None:
                    scores = [[None, None], [None, None], [None, None],
                              [None, None], [None, None]]
                else:
                    scores = match_dict['Scores'] + \
                             [[None, None]] * (5 - len(match_dict['Scores']))
                if match_dict['Result'] == 'W':
                    Winner_ID = Player_ID
                    Loser_ID = Opponent_ID
                # scores = match_dict['Scores']
                else:
                    Winner_ID = Opponent_ID
                    Loser_ID = Player_ID
                    scores = [s[::-1] for s in scores]  # reverse them because the player is always first
                # if scores is None:
                #	scores = [[None, None], [None, None], [None, None], [None, None], [None, None]]
                # else:
                #	scores = scores + [[None, None]]*(5-len(scores))
                # TODO: try to find if either record has more info and update if so
                c.execute("SELECT Match_ID "
                           "FROM Match "
                           "WHERE Winner_ID=? "
                           "AND Loser_ID=? "
                           "AND (Event IS NULL OR Event=?) "
                           "AND (Date IS NULL OR Date=?) "
                           "AND (Country IS NULL OR Country=?) "
                           "AND (Round IS NULL OR Round=?) "
                           "AND (Number_of_Games IS NULL OR Number_of_Games=?) "
                           "AND (Duration IS NULL OR Duration=?) "
                           "AND (Game1_Winner_Score IS NULL "
                           "OR Game1_Winner_Score=?) "
                           "AND (Game1_Loser_Score IS NULL "
                           "OR Game1_Loser_Score=?) "
                           "AND (Game2_Winner_Score IS NULL "
                           "OR Game2_Winner_Score=?) "
                           "AND (Game2_Loser_Score IS NULL "
                           "OR Game2_Loser_Score=?) "
                           "AND (Game3_Winner_Score IS NULL "
                           "OR Game3_Winner_Score=?) "
                           "AND (Game3_Loser_Score IS NULL "
                           "OR Game3_Loser_Score=?) "
                           "AND (Game4_Winner_Score IS NULL "
                           "OR Game4_Winner_Score=?) "
                           "AND (Game4_Loser_Score IS NULL "
                           "OR Game4_Loser_Score=?) "
                           "AND (Game5_Winner_Score IS NULL "
                           "OR Game5_Winner_Score=?) "
                           "AND (Game5_Loser_Score IS NULL "
                           "OR Game5_Loser_Score=?) "
                           "AND (Retired IS NULL "
                           "OR Retired=?) "
                           "AND (PSA IS NULL "
                           "OR PSA=?)", (
                           Winner_ID,
                           Loser_ID,
                           match_dict.get('Event'),
                           match_dict.get('Date'),
                           match_dict.get('Country'),
                           match_dict.get('Round'),
                           ngames,
                           match_dict.get('Duration'),
                           scores[0][0],
                           scores[0][1],
                           scores[1][0],
                           scores[1][1],
                           scores[2][0],
                           scores[2][1],
                           scores[3][0],
                           scores[3][1],
                           scores[4][0],
                           scores[4][1],
                           match_dict.get('Retired'),
                           match_dict.get('PSA')))
                Game_ID = c.fetchone()
                if Game_ID is None:
                    nmatches_added += 1
                    c.execute(
                        "INSERT INTO Match("
                        "Winner_ID, "
                        "Loser_ID, "
                        "Event, "
                        "Date, "
                        "Country, "
                        "Round, "
                        "Number_of_Games, "
                        "Duration, "
                        "Game1_Winner_Score, "
                        "Game1_Loser_Score, "
                        "Game2_Winner_Score, "
                        "Game2_Loser_Score, "
                        "Game3_Winner_Score, "
                        "Game3_Loser_Score, "
                        "Game4_Winner_Score, "
                        "Game4_Loser_Score, "
                        "Game5_Winner_Score, "
                        "Game5_Loser_Score, "
                        "Retired, "
                        "PSA) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                        "?, ?)", (
                        Winner_ID,
                        Loser_ID,
                        match_dict.get('Event'),
                        match_dict.get('Date'),
                        match_dict.get('Country'),
                        match_dict.get('Round'),
                        ngames,
                        match_dict.get('Duration'),
                        scores[0][0],
                        scores[0][1],
                        scores[1][0],
                        scores[1][1],
                        scores[2][0],
                        scores[2][1],
                        scores[3][0],
                        scores[3][1],
                        scores[4][0],
                        scores[4][1],
                        match_dict.get('Retired'),
                        match_dict.get('PSA')))
            infostr = '{} {} {} played {} games and {} are new'.format(
                player_dict['First_Name'], player_dict['Middle_Name'],
                player_dict['Last_Name'], nmatches, nmatches_added)
            print(infostr)
            # c.execute("COMMIT")

c.execute("ANALYZE")  # analyzer database for faster queries
c.close()
conn.close()
