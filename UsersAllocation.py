import numpy as np
import pandas as pd
import math
import random
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Defining types
IOT_TYPE = 'iot'
MMB_TYPE = 'mmb'

# Stores user generation order strategy
class UserGenOrder(Enum):
    FIRST_MMB = 'firstMmb'
    FIRST_IOT = 'firstIot'
    SHUFFLE = 'shuffle'

# Generates
def generateUsers(maxX, maxY, totUsers, iotUsersPercent, userGenOrder = UserGenOrder.SHUFFLE):

    # MMB users percent (of totUsers)
    mmbUsersPercent = 1.0 - iotUsersPercent

    # Number of IoT users
    numIotUsers = round(totUsers * iotUsersPercent)

    # Number of MMB users
    numMmbUsers = round(totUsers * mmbUsersPercent)

    # Generating users in random locations
    users = pd.DataFrame()
    for r in range(totUsers):
        users.at[r, 'x'] = random.random() * maxX
        users.at[r, 'y'] = random.random() * maxY

        # Assigning user type according to current index and userGenOrder value
        if (r >= numMmbUsers and userGenOrder is UserGenOrder.FIRST_MMB) or \
                (r < numIotUsers and userGenOrder is not UserGenOrder.FIRST_MMB):
            users.at[r, 'type'] = IOT_TYPE
        else:
            users.at[r, 'type'] = MMB_TYPE

    # Shuffling users
    if (userGenOrder is UserGenOrder.SHUFFLE):
        users = users.sample(frac=1).reset_index(drop=True)

    print('Generated', numIotUsers, 'IoT users and', numMmbUsers, 'MMB users')

    return users

# Maximum number of physical resource blocks
MAX_PRB = 25

class User:
    def __init__(self, nprb, nSubcarrier, dfc, f, modOrder, overhead, color, marker):

        # Number of pyshical resource blocks
        self.nprb = nprb

        # Number of subcarrier
        self.nSubcarrier = nSubcarrier

        # Frequency slice [kHz]
        self.dfc = dfc

        # [kHz]
        self.wprb = dfc * nSubcarrier

        # [dB]
        self.f = f

        # Mod order [bit/sym]
        self.modOrder = modOrder

        # Noise [dBm]
        self.noise = -173.977 + f + 10 * math.log10(self.wprb * nprb)

        # 1 - overhead
        self.overhead = overhead

        # Plot color
        self.color = color

        # Plot marker
        self.marker = marker

        # Bandwidth [kHz]
        self.bandwidth = 4500

        # Number of symbol
        self.nSymbol = 13

        # Number of symbols for slot
        self.nSymSlot = self.nSymbol / 14

        # Number of layer (MIMO)
        self.nLayer = 1

        # Maximum scaling factor
        self.scalingFactor = 1

        # R coding
        self.rCoding = 948 / 1024

    # Gets the maximum bitrate [Mbit/s]
    def getMaxBitRate(self):
        maxBitRate = self.nLayer * self.modOrder * self.scalingFactor * self.rCoding
        maxBitRate *= self.nSymSlot * self.overhead * MAX_PRB
        maxBitRate *= 1000 * 14 * 10 / 1000000
        return maxBitRate


# Defining l0 (used in path loss computation)
d0 = 1  # m
f = 24  # Gh
l0 = 10 * math.log10(((4 * math.pi * d0) / f) ** 2)

# True: a base station could connect only a user type (IoT or MMB)
isBSExclusive = False

# Creating MMB user
mmbUser = User(9, 12, 15, 4, 6, 1, 'b', ',')

# Creating IoT user
iotUser = User(1, 12, 15, 7, 2, 0.86, 'g', '.')

# Gets user by type
def getUser(type):
    return mmbUser if type == MMB_TYPE else iotUser

# Gets user plot color by type
def getColor(type):
    return getUser(type).color

# Gets user plot marker by type
def getMarker(type):
    return getUser(type).marker

# Plots base stations and users position
def plotPositions(bs, users, connections=None):
    # Getting MMB/IoT users
    mmbUsers = users[users.type == MMB_TYPE]
    iotUsers = users[users.type == IOT_TYPE]

    fig, ax = plt.subplots()

    # Plotting BS, MMB and IoT Users positions
    ax.scatter(bs.y, bs.x, color='y', marker='^')
    ax.scatter(mmbUsers.y, mmbUsers.x, color=getColor(MMB_TYPE), marker=getMarker(MMB_TYPE))
    ax.scatter(iotUsers.y, iotUsers.x, color=getColor(IOT_TYPE), marker=getMarker(IOT_TYPE))

    # Plotting labels and legend
    plt.ylabel('y')
    plt.xlabel('x')
    ax.legend(['Base Stations', 'MMB Users', 'IoT Users'])

    # Getting delta x/y to plot BS index on top/right of BS position
    deltaX, deltaY = bs.x.max()*0.01, bs.y.max()*0.01

    # Plotting base station indexes
    for i, j in bs.iterrows():
        ax.annotate(i, (j.y + deltaY, j.x + deltaX), color='y')

    # Plotting user/base station connections
    if (connections is not None):
        for ui, conn in connections.iterrows():
            if (conn.bsIdx is not None):
                bsi = int(conn.bsIdx)

                # Getting user/base station locations
                x1, x2 = users.iloc[int(conn.uIdx)].x, bs.iloc[bsi].x
                y1, y2 = users.iloc[int(conn.uIdx)].y, bs.iloc[bsi].y

                if (not math.isinf(x1) and not math.isinf(y1)):
                    ax.plot([y1, y2], [x1, x2], color=getColor(conn.type), alpha=0.2)
            else:
                # Highlighting users not connected
                pos = (users.loc[int(ui), 'y'], users.loc[int(ui), 'x'])
                circle = plt.Circle(pos, 8, color='r', fill=False, alpha=0.8)
                ax.add_artist(circle)


# Computes cartesian distance
def computeDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Computes path loss
def computePathLoss(distance, gamma=3):
    return l0 + 10 * gamma * math.log10(distance)

# Computes path losses for all users/base statiions combinations
def computePathLosses(bs, users):
    pathLosses = []
    for ui, user in users.iterrows():
        for bsi, base in bs.iterrows():
            # Computing distance between user and base station
            dist = computeDistance(user.x, user.y, base.x, base.y)

            # Computing path loss between user and base station
            pathLoss = computePathLoss(dist)

            pathLosses.append([bsi, ui, user.type, pathLoss])

    return pd.DataFrame(pathLosses, columns=['bsIdx', 'uIdx', 'type', 'pathLoss'])

# Computes BS/users minimum path loss connections
def computeMinPathLossesConnections(bs, pathLosses):
    # Setting to max the PRB available for the base stations
    bs['freePrb'] = MAX_PRB

    if (isBSExclusive):
        bs['bsType'] = None

    minPathLossConn = pd.DataFrame({'uIdx': pathLosses.uIdx.unique()})
    minPathLossConn['bsIdx'] = None
    minPathLossConn['type'] = None
    minPathLossConn['pathLoss'] = math.inf
    minPathLossConn['bitRate'] = 0

    userPathLosses = pathLosses.sort_values(by='pathLoss').groupby('uIdx')

    # Iterating on user path loss grouped by index
    for uIdx, userPathLoss in userPathLosses:
        # Iterating on single users
        for rowIdx, row in userPathLoss.iterrows():

            # Computing free PRB for current base station
            freePrb = bs.at[row.bsIdx, 'freePrb'] - getUser(row.type).nprb

            # A BS is considered valid (just in case of exclusive BS hypothesis)
            # when its type is the same as the user
            isValidBS = True
            if (isBSExclusive and bs.at[row.bsIdx, 'bsType'] is not None):
                isValidBS = bs.at[row.bsIdx, 'bsType'] == row.type

            if (freePrb >= 0 and isValidBS):

                # Reducing the available PRB for current base station
                bs.at[row.bsIdx, 'freePrb'] = freePrb

                if (isBSExclusive):
                    bs.at[row.bsIdx, 'bsType'] = row.type

                # Allocating current user (uIdx) to current base station (row.bsIdx)
                minPathLossConn.at[uIdx, 'pathLoss'] = row.pathLoss
                minPathLossConn.at[uIdx, 'type'] = row.type
                minPathLossConn.at[uIdx, 'bsIdx'] = int(row.bsIdx)
                minPathLossConn.at[uIdx, 'uIdx'] = row.uIdx
                minPathLossConn.at[uIdx, 'bitRate'] = getUser(row.type).getMaxBitRate()

                break

    return minPathLossConn

class GeneticAllocation:
    def __init__(self, bs, users):
        self.bs = bs
        self.users = users
        self.population = {}

    def generatePopulation(self, population_size = 100):

        self.population = {}

        max_bs_index = len(self.bs)

        for individual_index in range(population_size):

            # Setting to max the PRB available for the base stations
            self.bs['freePrb'] = MAX_PRB

            individual = pd.DataFrame()

            # Assigning users to a random base station
            for uIdx, user in self.users.iterrows():

                individual.at[uIdx, "bsIdx"] = None
                individual.at[uIdx, "uIdx"] = int(uIdx)
                individual.at[uIdx, "type"] = user.type
                individual.at[uIdx, 'bitRate'] = 0

                bs_indexes = list(range(0, max_bs_index))
                while bs_indexes:
                    bs_index = random.choice(bs_indexes)

                    # Computing free PRB for current base station
                    free_prb = self.bs.at[bs_index, 'freePrb'] - getUser(user.type).nprb

                    # A BS is considered valid (just in case of exclusive BS hypothesis)
                    # when its type is the same as the user
                    is_valid_bs = True
                    if (isBSExclusive and self.bs.at[bs_index, 'type'] is not None):
                        is_valid_bs = self.bs.at[bs_index, 'type'] == user.type

                    if (free_prb >= 0 and is_valid_bs):

                        # Reducing the available PRB for current base station
                        self.bs.at[bs_index, 'freePrb'] = free_prb

                        if (isBSExclusive):
                            self.bs.at[bs_index, 'type'] = user.type

                        # Populating individual values
                        individual.at[uIdx, "bsIdx"] = bs_index
                        individual.at[uIdx, 'bitRate'] = getUser(user.type).getMaxBitRate()

                        break
                    else:
                        # BS not valid or not available physical blocks
                        bs_indexes.remove(bs_index)

            # TODO: capire perché i bsIdx vengono impostati come NaN e non come None (poi rimuovere questa riga)
            individual = individual.where(pd.notnull(individual), None)

            # Inserting individual in population
            self.population[individual_index] = individual

    def fitness(self):
        population_size = len(self.population)

        population_fitness = pd.DataFrame()
        for individual_index in self.population:
            individual = self.population[individual_index]
            population_fitness.at[individual_index, 'tot_bit_rate'] = individual.bitRate.sum()
            bs_not_alloc = individual[individual.bsIdx.isnull()].shape[0]
            population_fitness.at[individual_index, 'users_disc_percent'] = bs_not_alloc / population_size * 100

        population_fitness = population_fitness.sort_values(['tot_bit_rate', 'users_disc_percent'],
                                                            ascending=[False, True])

        #print(population_fitness)

        first_best_fitness = self.population[population_fitness.index.values[0]]
        second_best_fitness = self.population[population_fitness.index.values[1]]

        return first_best_fitness, second_best_fitness
