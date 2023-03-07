"""
This data extraction program is derived from
https://www.kaggle.com/code/iliaskydyraliev/nba-hall-of-fame-prediction-project 
under the Apache 2.0 open source license. See
https://www.apache.org/licenses/LICENSE-2.0

This program is also released under the Apache 2.0
open source license.

Author: edgexyz
"""

import pandas as pd
import os

player_totals=pd.read_csv('./NBA-stats/Player Totals.csv')
player_season_info=pd.read_csv('./NBA-stats/Player Season Info.csv')
player_career_info=pd.read_csv('./NBA-stats/Player Career Info.csv')
player_award_shares=pd.read_csv('./NBA-stats/Player Award Shares.csv')
all_star=pd.read_csv('./NBA-stats/All-Star Selections.csv')
advanced=pd.read_csv('./NBA-stats/Advanced.csv')
end_of_season=pd.read_csv('./NBA-stats/End of Season Teams.csv')
players = player_career_info

namelist = players['player']

points = []
assists = []
rebound = []
blocks = []
steals = []
games = []
minutes = []
freegoal = []
freegoal_percent = []
efg_percent = []

for name in namelist:
    points.append(player_totals[player_totals['player'] == name].sum()['pts'])
    assists.append(player_totals[player_totals['player'] == name].sum()['ast'])
    rebound.append(player_totals[player_totals['player'] == name].sum()['trb'])
    blocks.append(player_totals[player_totals['player'] == name].sum()['blk'])
    steals.append(player_totals[player_totals['player'] == name].sum()['stl'])
    games.append(player_totals[player_totals['player'] == name].sum()['g'])
    minutes.append(player_totals[player_totals['player'] == name].sum()['mp'])
    freegoal.append(player_totals[player_totals['player'] == name].sum()['fg'])
    freegoal_percent.append(player_totals[player_totals['player'] == name].sum()['fg_percent'])
    efg_percent.append(player_totals[player_totals['player'] == name].sum()['e_fg_percent'])
    

players['points']=points
players['assists']=assists
players['rebound']=rebound
players['blocks']=blocks
players['steals']=steals
players['games']=games
players['minutes']=minutes
players['FG'] = freegoal
players['FG%'] = freegoal_percent
players['eFG%'] = efg_percent

allnba1 = []
allnba2 = []
allnba3 = []

alldef1 = []
alldef2 = []

allrooks1 = []
allrooks2 = []

allaba1 = []
allaba2 = []

allstar = []

mvp = []
dpoy = []
nba_roy = []
mip = []
smoy = []
aba_mvp = []
aba_rook = []

for name in namelist:
    allnba = end_of_season[(end_of_season['type'] == 'All-NBA') & (end_of_season['player'] == name)]
    alldefense = end_of_season[(end_of_season['type'] == 'All-Defense')&(end_of_season['player'] == name)]
    allrooks = end_of_season[(end_of_season['type'] == 'All-Rookie')&(end_of_season['player'] == name)]
    allaba = end_of_season[(end_of_season['type'] == 'All-ABA') & (end_of_season['player'] == name)]

    allstars = all_star[all_star['player'] == name]

    mvps = player_award_shares[(player_award_shares['award'] == 'nba mvp') & (player_award_shares['player'] == name)]
    dpoys = player_award_shares[(player_award_shares['award'] == 'dpoy') & (player_award_shares['player'] == name)]
    nba_roys = player_award_shares[(player_award_shares['award'] == 'nba roy') & (player_award_shares['player'] == name)]
    mips = player_award_shares[(player_award_shares['award'] == 'mip') & (player_award_shares['player'] == name)]
    smoys = player_award_shares[(player_award_shares['award'] == 'nba mvp') & (player_award_shares['player'] == name)]
    aba_mvps = player_award_shares[(player_award_shares['award'] == 'aba mvp') & (player_award_shares['player'] == name)]
    aba_rooks = player_award_shares[(player_award_shares['award'] == 'aba roy') & (player_award_shares['player'] == name)]
    
    try:
        tmpnba1 = allnba['number_tm'].value_counts()['1st']
    except:
        tmpnba1 = 0
    
    try:
        tmpnba2 = allnba['number_tm'].value_counts()['2nd']
    except:
        tmpnba2 = 0

    try:
        tmpnba3 = allnba['number_tm'].value_counts()['3rd']
    except:
        tmpnba3 = 0

    allnba1.append(tmpnba1)
    allnba2.append(tmpnba2)
    allnba3.append(tmpnba3)

    try:
        tmpdef1 = alldefense['number_tm'].value_counts()['1st']
    except:
        tmpdef1 = 0
        
    try:
        tmpdef2 = alldefense['number_tm'].value_counts()['2nd']
    except:
        tmpdef2 = 0
    
    alldef1.append(tmpdef1)
    alldef2.append(tmpdef2)

    try:
        tmprooks1 = allrooks['number_tm'].value_counts()['1st']
    except:
        tmprooks1 = 0

    try:
        tmprooks2 = allrooks['number_tm'].value_counts()['2nd']
    except:
        tmprooks2 = 0

    allrooks1.append(tmprooks1)
    allrooks2.append(tmprooks2)

    try:
        tmpaba1 = allaba['number_tm'].value_counts()['1st']
    except:
        tmpaba1 = 0

    try:
        tmpaba2 = allaba['number_tm'].value_counts()['2nd']
    except:
        tmpaba2 = 0
    
    allaba1.append(tmpaba1)
    allaba2.append(tmpaba2)

    try:
        tmpAllStar = allstars['replaced'].count()
    except:
        tmpAllStar = 0
    
    allstar.append(tmpAllStar)

    try:
        tmpMVP = sum(mvps['winner'])
    except:
        tmpMVP = 0
    
    mvp.append(tmpMVP)

    try:
        tmpDPOY = sum(dpoys['winner'])
    except:
        tmpDPOY = 0
    
    dpoy.append(tmpDPOY)

    try:
        tmpROY = sum(nba_roys['winner'])
    except:
        tmpROY = 0
    
    nba_roy.append(tmpROY)

    try:
        tmpMIP = sum(mips['winner'])
    except:
        tmpMIP = 0

    mip.append(tmpMIP)

    try:
        tmpSMOY = sum(smoys['winner'])
    except:
        tmpSMOY = 0
    
    smoy.append(tmpSMOY)

    try:
        tmpABAmvp = sum(aba_mvps['winner'])
    except:
        tmpABAmvp = 0
    
    aba_mvp.append(tmpABAmvp)

    try:
        tmpABArook = sum(aba_rooks['winner'])
    except:
        tmpABArook = 0
    
    aba_rook.append(tmpABArook)

players['All NBA 1st team'] = allnba1
players['All NBA 2nd team'] = allnba2
players['All NBA 3rd team'] = allnba3

players['All Defense 1st team'] = alldef1
players['All Defense 2nd team'] = alldef2

players['All Rookie 1st team'] = allrooks1
players['All Rookie 2nd team'] = allrooks2

players['All ABA 1st team'] = allaba1
players['All ABA 2nd team'] = allaba2

players['All Star appearances'] = allstar

players['MVPs'] = mvp
players['DPOY'] = dpoy
players['NBA ROY'] = nba_roy
players['MIP'] = mip
players['SMOY'] = smoy
players['ABA MVP'] = aba_mvp
players['ABA ROY'] = aba_rook

isHOF = list(players['hof'])
hof = players.loc[isHOF]
players.drop(hof.index, inplace=True)
players = players.reset_index(drop=True)
hof = hof.reset_index(drop=True)

os.makedirs('results', exist_ok=True)
hof.to_csv('results/hof.csv', index=False)
players.to_csv('results/players.csv', index=False)