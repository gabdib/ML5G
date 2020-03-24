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


# Generates users
def generateUsers(max_x, max_y, tot_users, iot_users_percent, user_gen_order=UserGenOrder.SHUFFLE):

    # MMB users percent (of totUsers)
    mmb_users_percent = 1.0 - iot_users_percent

    # Number of IoT users
    num_iot_users = round(tot_users * iot_users_percent)

    # Number of MMB users
    num_mmb_users = round(tot_users * mmb_users_percent)

    # Generating users in random locations
    users = pd.DataFrame()
    for r in range(tot_users):
        users.at[r, 'x'] = random.random() * max_x
        users.at[r, 'y'] = random.random() * max_y

        # Assigning user type according to current index and userGenOrder value
        if (r >= num_mmb_users and user_gen_order is UserGenOrder.FIRST_MMB) or \
                (r < num_iot_users and user_gen_order is not UserGenOrder.FIRST_MMB):
            users.at[r, 'type'] = IOT_TYPE
        else:
            users.at[r, 'type'] = MMB_TYPE

    # Shuffling users
    if user_gen_order is UserGenOrder.SHUFFLE:
        users = users.sample(frac=1).reset_index(drop=True)

    print('Generated', num_iot_users, 'IoT users and', num_mmb_users, 'MMB users')

    return users


# Maximum number of physical resource blocks
MAX_PRB = 25

# Computes cartesian distance
def compute_distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if dist < 3:
        dist = 3

    return dist


# Computes l0 (used in path loss computation)
def getl0(d0 = 1, f = 2.4e9):
    c = 3e8
    l0 = 10 * math.log10(((4 * math.pi * d0) * f / c) ** 2)
    return l0

def compute_path_loss_by_distance(x1, y1, x2, y2, gamma=3):
    dist = compute_distance(x1, y1, x2, y2)
    return compute_path_loss(dist, gamma)

# Computes path loss
def compute_path_loss(distance, gamma=3):
    return getl0() + 10 * gamma * math.log10(distance)


class User:
    def __init__(self, parameters):

        # Number of physical resource blocks
        self.num_prb = parameters['num_prb']

        # Number of subcarrier
        self.num_of_subcarrier = parameters['num_of_subcarrier']

        # Frequency slice [kHz]
        self.dfc = parameters['dfc']

        # [dB]
        self.f = parameters['f']

        # Power [dBm]
        self.pt = parameters['pt']

        # 1 - overhead
        self.overhead = parameters['overhead']

        # Plot color
        self.color = parameters['color']

        # Plot marker
        self.marker = parameters['marker']

        self.ebn0_t = parameters['ebn0_t']

        # [dB]
        self.qpsk = None
        if 'qpsk' in parameters:
            self.qpsk = parameters['qpsk']

        # [dB]
        self.qam16 = None
        if '16qam' in parameters:
            self.qam16 = parameters['16qam']

        # [dB]
        self.qam64 = None
        if 'qam64' in parameters:
            self.qam64 = parameters['64qam']

        # Bandwidth [kHz]
        self.bandwidth = 4500

        # Number of symbol
        self.num_symbol = 13

        # Number of symbols for slot
        self.num_sym_slot = self.num_symbol / 14

        # Number of layer (MIMO)
        self.num_layer = 1

        # Maximum scaling factor
        self.scaling_factor = 1

        # R coding
        self.r_coding = 948 / 1024

    def get_mod_order(self, path_loss):

        ebn0 = self.get_eb_n0(path_loss)

        if self.qpsk is not None:
            if ebn0 < self.qpsk:
                return 2
            elif self.qam16 is not None and self.qpsk <= ebn0 < self.qam16:
                return 4
            elif self.qam64 is not None:
                if self.qam16 <= ebn0 < self.qam64:
                    return 6
                elif ebn0 > self.qam64:
                    return 8

        if ebn0 > self.ebn0_t:
            return 1
        else:
            return 0

    # Gets width of a PRB [kHz]
    def get_wprb(self):
        return self.dfc * self.num_of_subcarrier

    # Gets noise [dBm]
    def get_noise(self):
        # [dBm]
        return -173.977 + self.f + 10 * math.log10(self.get_wprb() * self.num_prb)

    def get_eb_n0(self, path_loss):
        return self.pt - path_loss - self.get_noise()

    def is_visible(self, user_x, user_y, bs_x, bs_y, gamma=3):

        # Computing distance
        dist = compute_distance(user_x, user_y, bs_x, bs_y)

        # Computing path loss
        path_loss = compute_path_loss(dist, gamma)

        ebn0 = self.get_eb_n0(path_loss)

        return ebn0 >= self.ebn0_t

    # Gets the maximum bit rate [Mbit/s]
    def get_bit_rate(self, path_loss = 100):
        max_bit_rate = self.num_layer * self.scaling_factor * self.r_coding
        max_bit_rate *= self.num_sym_slot * self.get_mod_order(path_loss) * self.overhead * self.num_prb
        max_bit_rate *= 1000 * 14 * 10 / 1000000
        return max_bit_rate


# True: a base station could connect only a user type (IoT or MMB)
is_bs_exclusive = False

mmbUser = User({
    'num_prb': 9,
    'num_of_subcarrier': 12,
    'dfc': 15,
    'f': 4,
    'pt': 20,
    'mod_order': 6,
    'ebn0_t': 10,
    'qpsk':6,
    '16qam':12,
    '64qam':18,
    'overhead': 1,
    'color': 'b',
    'marker': ','
})
# Creating MMB user

# Creating IoT user
iotUser = User({
    'num_prb': 1,
    'num_of_subcarrier': 12,
    'dfc': 15,
    'f': 7,
    'pt': 0,
    'mod_order': 2,
    'ebn0_t': 0,
    'qpsk':0,
    'overhead': 0.86,
    'color': 'g',
    'marker': '.'
})

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

                if not math.isinf(x1) and not math.isinf(y1):
                    ax.plot([y1, y2], [x1, x2], color=getColor(conn.type), alpha=0.2)
            else:
                # Highlighting users not connected
                pos = (users.loc[int(ui), 'y'], users.loc[int(ui), 'x'])
                circle = plt.Circle(pos, 8, color='r', fill=False, alpha=0.8)
                ax.add_artist(circle)


def evaluate_solution(connections, tot_users):

    # Getting the total network bit rate
    tot_bit_rate = connections.bitRate.sum()

    # Getting percentage of users non connected
    disc_users_percent = connections[connections.bsIdx.isnull()].shape[0] / tot_users * 100

    solution_result = {
        'tot_bit_rate': tot_bit_rate,
        'disc_users_percent': disc_users_percent
    }

    return solution_result


# Computes path losses for all users/base stations combinations
def compute_path_losses(bs, users):
    path_losses = []
    for ui, user in users.iterrows():
        for bsi, base in bs.iterrows():

            # Computing distance between user and base station
            dist = compute_distance(user.x, user.y, base.x, base.y)

            # Computing path loss between user and base station
            path_loss = compute_path_loss(dist)

            path_losses.append([bsi, ui, user.type, path_loss])

    return pd.DataFrame(path_losses, columns=['bsIdx', 'uIdx', 'type', 'pathLoss'])


# Computes BS/users minimum path loss connections
def compute_min_path_losses_connections(bs, path_losses):
    # Setting to max the PRB available for the base stations
    bs['freePrb'] = MAX_PRB

    if is_bs_exclusive:
        bs['bsType'] = None

    connections = pd.DataFrame({'uIdx': path_losses.uIdx.unique()})
    connections['bsIdx'] = None
    connections['type'] = None
    connections['pathLoss'] = math.inf
    connections['bitRate'] = 0

    user_path_losses = path_losses.sort_values(by='pathLoss').groupby('uIdx')

    # Iterating on user path loss grouped by index
    for user_index, user_path_loss in user_path_losses:
        # Iterating on single users
        for row_index, row in user_path_loss.iterrows():

            # Computing free PRB for current base station
            free_prb = bs.at[row.bsIdx, 'freePrb'] - getUser(row.type).num_prb

            # A BS is considered valid (just in case of exclusive BS hypothesis)
            # when its type is the same as the user
            is_valid_bs = True
            if is_bs_exclusive and bs.at[row.bsIdx, 'bsType'] is not None:
                is_valid_bs = bs.at[row.bsIdx, 'bsType'] == row.type

            if (free_prb >= 0 and is_valid_bs):

                # Reducing the available PRB for current base station
                bs.at[row.bsIdx, 'freePrb'] = free_prb

                if is_bs_exclusive:
                    bs.at[row.bsIdx, 'bsType'] = row.type

                # Allocating current user (uIdx) to current base station (row.bsIdx)
                connections.at[user_index, 'pathLoss'] = row.pathLoss
                connections.at[user_index, 'type'] = row.type
                connections.at[user_index, 'bsIdx'] = int(row.bsIdx)
                connections.at[user_index, 'uIdx'] = row.uIdx
                connections.at[user_index, 'bitRate'] = getUser(row.type).get_bit_rate(row.pathLoss)

                break

    return connections


class GeneticAllocation:

    def __init__(self, bs, users):
        self.bs = bs
        self.users = users
        self.population = {}

    def genetic_evolution(self, num_of_generation=10, population_size=100):

        # Generating the first generation of the population
        self.generate_population()

        solution_size = len(self.population)

        # Evolving population for num_of_generation generations
        for actual_generation in range(num_of_generation):

            # Computing best two individuals
            first_best_individual, second_best_individual = self.fitness()

            # Crossing-over the best two individuals
            next_gen_individual = self.crossover(first_best_individual, second_best_individual)

            # Evaluating current solution
            solution = evaluate_solution(next_gen_individual, solution_size)

            print("Generation:", actual_generation + 1, "results")
            print('Total bit rate:', solution['tot_bit_rate'], '[Mbit/s]')
            print('Users disconnected:', solution['disc_users_percent'], '%')

            if actual_generation is not num_of_generation:
                # Mutating next generation individual prototype
                self.population = self.mutation(next_gen_individual)

        return next_gen_individual

    def generate_population(self, population_size = 100):

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
                    free_prb = self.bs.at[bs_index, 'freePrb'] - getUser(user.type).num_prb

                    # A BS is considered valid (just in case of exclusive BS hypothesis)
                    # when its type is the same as the user
                    is_valid_bs = True
                    if (is_bs_exclusive and self.bs.at[bs_index, 'type'] is not None):
                        is_valid_bs = self.bs.at[bs_index, 'type'] == user.type

                    if (free_prb >= 0 and is_valid_bs):

                        # Reducing the available PRB for current base station
                        self.bs.at[bs_index, 'freePrb'] = free_prb

                        if is_bs_exclusive:
                            self.bs.at[bs_index, 'type'] = user.type

                        base_station = self.bs.iloc[bs_index]
                        path_loss = compute_path_loss_by_distance(user.x, user.y, base_station.x, base_station.y)

                        # Populating individual values
                        individual.at[uIdx, "bsIdx"] = bs_index
                        individual.at[uIdx, 'bitRate'] = getUser(user.type).get_bit_rate(path_loss)

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

        # Sorting population by best fitness
        population_fitness = population_fitness.sort_values(['tot_bit_rate', 'users_disc_percent'],
                                                            ascending=[False, True])
        
        # Getting first/second best fitness individual
        first_best_fitness = self.population[population_fitness.index.values[0]]
        second_best_fitness = self.population[population_fitness.index.values[1]]

        return first_best_fitness, second_best_fitness

    def crossover(self, first_best_fitness, second_best_fitness):
        next_gen_individual = first_best_fitness

        #TODO fare il merge del first e del second (considerando i vincoli su i PRB delle BS disponibili)

        return next_gen_individual

    def mutation(self, next_gen_individual):
        population_size = len(self.population)
        new_population = {}
        for individual_index in range(population_size):
            mutant = next_gen_individual
            # TODO: mutazione individuo
            new_population[individual_index] = mutant

        return new_population
