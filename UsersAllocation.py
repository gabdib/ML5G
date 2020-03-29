import math
import random
from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd

# Defining types
IOT_TYPE = 'iot'
MMB_TYPE = 'mmb'


# Stores user generation order strategy
class UserGenOrder(Enum):
    FIRST_MMB = 'firstMmb'
    FIRST_IOT = 'firstIot'
    SHUFFLE = 'shuffle'


# Stores modulation order
class ModOrder:
    NO_MOD = 0
    QPSK = 2
    QAM16 = 4
    QAM64 = 6
    QAM256 = 8


# Generates users
def generate_users(max_x, max_y, tot_users, iot_users_percent, user_gen_order=UserGenOrder.SHUFFLE):

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
def get_l0(d0=1, f=2.4e9):
    c = 3e8
    l0 = 10 * math.log10(((4 * math.pi * d0) * f / c) ** 2)
    return l0


# Computes path loss
def compute_path_loss(distance, gamma=3):
    return get_l0() + 10 * gamma * math.log10(distance)


# Compute path loss by distance
def compute_path_loss_by_distance(x1, y1, x2, y2, gamma=3):
    dist = compute_distance(x1, y1, x2, y2)
    return compute_path_loss(dist, gamma)


class User:
    def __init__(self, user_type, parameters):

        # User type [iot or mmb]
        self.user_type = user_type

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

    # Gets mod order by path loss
    def get_mod_order(self, path_loss):

        ebn0 = self.get_eb_n0(path_loss)

        if self.user_type is MMB_TYPE:
            if ebn0 <= self.ebn0_t:
                return ModOrder.NO_MOD
            elif self.ebn0_t < ebn0 <= self.qpsk:
                return ModOrder.QPSK
            elif self.qpsk < ebn0 <= self.qam16:
                return ModOrder.QAM16
            elif self.qam16 <= ebn0 < self.qam16:
                return ModOrder.QAM64
            else:
                return ModOrder.QAM256
        elif self.user_type is IOT_TYPE:
            if ebn0 <= self.ebn0_t:
                return ModOrder.NO_MOD
            else:
                return ModOrder.QPSK
        else:
            print('Unknown user type', self.user_type)
            return None

    # Gets width of a PRB [kHz]
    def get_wprb(self):
        return self.dfc * self.num_of_subcarrier

    # Gets noise [dBm]
    def get_noise(self):
        return -173.977 + self.f + 10 * math.log10(self.get_wprb()*1e3 * self.num_prb)

    def get_eb_n0(self, path_loss):
        return self.pt - path_loss - self.get_noise()

    def is_visible_by_path_loss(self, path_loss, gamma=3):

        # Computing Eb/N0
        ebn0 = self.get_eb_n0(path_loss)

        return ebn0 >= self.ebn0_t

    def is_visible_by_position(self, user_x, user_y, bs_x, bs_y, gamma=3):

        # Computing distance
        dist = compute_distance(user_x, user_y, bs_x, bs_y)

        # Computing path loss
        path_loss = compute_path_loss(dist, gamma)

        return self.is_visible_by_path_loss(path_loss, gamma)

    # Gets the max bit rate [Mbit/s]
    def get_max_bit_rate(self):
        return self.get_bit_rate(0)

    # Gets the bit rate [Mbit/s]
    def get_bit_rate(self, path_loss):
        bit_rate = self.num_layer * self.scaling_factor * self.r_coding
        bit_rate *= self.num_sym_slot * self.get_mod_order(path_loss) * self.overhead * self.num_prb
        bit_rate *= 1000 * 14 * 10 / 1000000
        return bit_rate

    # Gets the bit rate by user/bs position [Mbit/s]
    def get_bit_rate_by_position(self, user_x, user_y, bs_x, bs_y):
        path_loss = compute_path_loss_by_distance(user_x, user_y, bs_x, bs_y)
        return self.get_bit_rate(path_loss)


# True: a base station could connect only a user type (IoT or MMB)
is_bs_exclusive = False

# Creating MMB user
mmb_user = User(MMB_TYPE, {
    'num_prb': 9,
    'num_of_subcarrier': 12,
    'dfc': 15,
    'f': 4,
    'pt': 20,
    'ebn0_t': 10,
    'qpsk': 12,
    '16qam': 18,
    '64qam': 24,
    'overhead': 1,
    'color': 'b',
    'marker': ','
})

# Creating IoT user
iot_user = User(IOT_TYPE, {
    'num_prb': 1,
    'num_of_subcarrier': 12,
    'dfc': 15,
    'f': 7,
    'pt': 0,
    'ebn0_t': 0,
    'qpsk': 6,
    'overhead': 0.86,
    'color': 'g',
    'marker': '.'
})


# Gets user by type
def get_user(user_type):
    return mmb_user if user_type == MMB_TYPE else iot_user


# Gets user plot color by type
def get_color(user_type):
    return get_user(user_type).color


# Gets user plot marker by type
def get_marker(user_type):
    return get_user(user_type).marker


# Plots base stations and users position
def plot_positions(bs, users, connections=None):
    # Getting MMB/IoT users
    mmb_users = users[users.type == MMB_TYPE]
    iot_users = users[users.type == IOT_TYPE]

    fig, ax = plt.subplots()

    # Plotting BS, MMB and IoT Users positions
    ax.scatter(bs.y, bs.x, color='y', marker='^')
    ax.scatter(mmb_users.y, mmb_users.x, color=get_color(MMB_TYPE), marker=get_marker(MMB_TYPE))
    ax.scatter(iot_users.y, iot_users.x, color=get_color(IOT_TYPE), marker=get_marker(IOT_TYPE))

    # Plotting labels and legend
    plt.ylabel('y')
    plt.xlabel('x')
    ax.legend(['Base Stations', 'MMB Users', 'IoT Users'])

    # Getting delta x/y to plot BS index on top/right of BS position
    delta_x, delta_y = bs.x.max()*0.01, bs.y.max()*0.01

    circle_size = (bs.x.max() + bs.y.max()) * 0.01

    # Plotting base station indexes
    for i, j in bs.iterrows():
        ax.annotate(i, (j.y + delta_y, j.x + delta_x), color='y')

    # Plotting user/base station connections
    if connections is not None:
        for ui, conn in connections.iterrows():
            if conn.bsIdx is not None:
                bsi = int(conn.bsIdx)

                # Getting user/base station locations
                x1, x2 = users.iloc[int(conn.uIdx)].x, bs.iloc[bsi].x
                y1, y2 = users.iloc[int(conn.uIdx)].y, bs.iloc[bsi].y

                if not math.isinf(x1) and not math.isinf(y1):
                    ax.plot([y1, y2], [x1, x2], color=get_color(conn.type), alpha=0.2)
            else:
                # Highlighting users not connected
                pos = (users.loc[int(ui), 'y'], users.loc[int(ui), 'x'])
                circle = plt.Circle(pos, circle_size, color='r', fill=False, alpha=0.8)
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


def show_solution(bs, users, connections, tot_users):

    # Evaluating solution
    solution = evaluate_solution(connections, tot_users)

    print('Network bit rate is', solution['tot_bit_rate'], '[Mbit/s]')
    print('Users disconnected:', solution['disc_users_percent'], '%')

    # Plotting users/base stations and related connections
    plot_positions(bs, users, connections)


# Computes path losses for all users/base stations combinations
def compute_path_losses(bs, users):
    path_losses = []
    for ui, user in users.iterrows():
        for bsi, base in bs.iterrows():

            # Computing distance between user and base station
            dist = compute_distance(user.x, user.y, base.x, base.y)

            # Computing path loss between user and base station
            path_loss = compute_path_loss(dist)

            path_losses.append([bsi, ui, user.type, user.x, user.y, path_loss])

    return pd.DataFrame(path_losses, columns=['bsIdx', 'uIdx', 'type', 'x', 'y', 'pathLoss'])


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
            free_prb = bs.at[row.bsIdx, 'freePrb'] - get_user(row.type).num_prb

            # A BS is considered valid (just in case of exclusive BS hypothesis)
            # when its type is the same as the user
            is_valid_bs = True
            if is_bs_exclusive and bs.at[row.bsIdx, 'bsType'] is not None:

                is_valid_bs = bs.at[row.bsIdx, 'bsType'] == row.type

            # Computing visibility according to bs/users path loss
            is_visible = get_user(row.type).is_visible_by_path_loss(row.pathLoss)

            if free_prb >= 0 and is_valid_bs and is_visible:

                # Reducing the available PRB for current base station
                bs.at[row.bsIdx, 'freePrb'] = free_prb

                if is_bs_exclusive:
                    bs.at[row.bsIdx, 'bsType'] = row.type

                # Allocating current user (uIdx) to current base station (row.bsIdx)
                connections.at[user_index, 'pathLoss'] = row.pathLoss
                connections.at[user_index, 'type'] = row.type
                connections.at[user_index, 'bsIdx'] = int(row.bsIdx)
                connections.at[user_index, 'uIdx'] = row.uIdx
                connections.at[user_index, 'bitRate'] = get_user(row.type).get_bit_rate(row.pathLoss)

                break

    return connections


class GeneticAllocation:

    def __init__(self, bs, users):
        self.bs = bs
        self.users = users
        self.population = {}

    def genetic_evolution(self, num_of_generation=10, population_size=100):

        # Generating the first generation of the population
        self.generate_population(population_size)

        solution_size = len(self.population)

        next_gen_individual = None

        # Evolving population for num_of_generation generations
        for actual_generation in range(num_of_generation):

            print("Generation:", actual_generation + 1)

            # Computing best two individuals
            first_best_individual, second_best_individual = self.fitness()

            # Crossing-over the best two individuals
            next_gen_individual = self.crossover(first_best_individual, second_best_individual)

            # Evaluating current solution
            solution = evaluate_solution(next_gen_individual, solution_size)

            print('Total bit rate:', solution['tot_bit_rate'], '[Mbit/s]')
            print('Users disconnected:', solution['disc_users_percent'], '%')

            if actual_generation is not num_of_generation:
                # Mutating next generation individual prototype
                self.population = self.mutation(next_gen_individual)

        return next_gen_individual

    def generate_population(self, population_size=100):

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
                individual.at[uIdx, 'x'] = user.x
                individual.at[uIdx, 'y'] = user.y

                bs_indexes = list(range(0, max_bs_index))
                while bs_indexes:
                    bs_index = random.choice(bs_indexes)
                    base_station = self.bs.iloc[bs_index]

                    # Computing free PRB for current base station
                    free_prb = self.bs.at[bs_index, 'freePrb'] - get_user(user.type).num_prb

                    # A BS is considered valid (just in case of exclusive BS hypothesis)
                    # when its type is the same as the user
                    is_valid_bs = True
                    if is_bs_exclusive and base_station.type is not None:
                        is_valid_bs = base_station.type == user.type

                    # Computing path loss
                    path_loss = compute_path_loss_by_distance(user.x, user.y, base_station.x, base_station.y)

                    # Computing visibility between current bs and current user
                    is_visible = get_user(user.type).is_visible_by_path_loss(path_loss)

                    if free_prb >= 0 and is_valid_bs and is_visible:

                        # Reducing the available PRB for current base station
                        self.bs.at[bs_index, 'freePrb'] = free_prb

                        if is_bs_exclusive:
                            self.bs.at[bs_index, 'type'] = user.type

                        # Populating individual values
                        individual.at[uIdx, "bsIdx"] = bs_index
                        individual.at[uIdx, 'bitRate'] = get_user(user.type).get_bit_rate(path_loss)

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

    def mutation(self, next_gen_individual, mutation_rate=0.1):
        population_size = len(self.population)
        new_population = {}

        max_user_index = len(self.users)

        mutation_size = int(max_user_index*mutation_rate)

        for individual_index in range(population_size):
            mutant = next_gen_individual

            individual = self.population[individual_index]

            source_user_indexes = list(range(0, max_user_index))
            mutation_done = 0

            while source_user_indexes and mutation_size >= mutation_done:

                # Getting source user
                source_user_index = random.choice(source_user_indexes)
                source_user = individual.iloc[source_user_index]
                source_user_prb = get_user(source_user.type).num_prb

                if source_user['bsIdx'] is None:
                    source_user_indexes.remove(source_user_index)
                    continue

                target_user_indexes = list(range(0, max_user_index))
                while target_user_indexes:

                    # Getting target user
                    target_user_index = random.choice(target_user_indexes)
                    target_user = individual.iloc[target_user_index]
                    target_user_prb = get_user(target_user.type).num_prb

                    if source_user_prb == target_user_prb:

                        # Getting source/target bs
                        source_bs = self.bs.iloc[int(source_user['bsIdx'])]

                        is_source_visible = get_user(source_user.type).is_visible_by_position(
                            source_bs.x, source_bs.y, target_user.x, target_user.y)

                        if target_user['bsIdx'] is not None:
                            target_bs = self.bs.iloc[int(target_user['bsIdx'])]
                            is_target_visible = get_user(target_user.type).is_visible_by_position(
                                source_user.x, source_user.y, target_bs.x, target_bs.y)
                        else:
                            target_bs = None
                            is_target_visible = True

                        if is_source_visible and is_target_visible:
                            # Computing new source bit rate
                            if target_bs is not None:
                                mutant.at[source_user_index, 'bitRate'] = get_user(source_user.type)\
                                    .get_bit_rate_by_position( source_user.x, source_user.y, target_bs.x, target_bs.y)
                            else:
                                mutant.at[source_user_index, 'bitRate'] = 0

                            # Computing new target bit rate
                            mutant.at[target_user_index, 'bitRate'] = get_user(source_user.type)\
                                .get_bit_rate_by_position(target_user.x, target_user.y, source_bs.x, source_bs.y)

                            # Mutating individual
                            mutant.at[target_user_index, 'bsIdx'] = source_user['bsIdx']
                            mutant.at[source_user_index, 'bsIdx'] = target_user['bsIdx']

                            mutation_done = mutation_done + 1

                            break
                    else:
                        target_user_indexes.remove(target_user_index)

                source_user_indexes.remove(source_user_index)

            new_population[individual_index] = mutant

        return new_population
