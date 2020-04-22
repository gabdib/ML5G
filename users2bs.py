import math
import random
from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
class UsersGenerator:

    def __init__(self, max_x, max_y, tot_users, iot_users_percent):

        self.max_x = max_x

        self.max_y = max_y

        self.tot_users = tot_users

        self.num_iot_users = iot_users_percent

        # MMB users percent (of totUsers)
        self.mmb_users_percent = 1.0 - iot_users_percent

        # Number of IoT users
        self.num_iot_users = round(tot_users * iot_users_percent)

        # Number of MMB users
        self.num_mmb_users = round(tot_users * self.mmb_users_percent)

    # Generates a population of num_generation size
    def generate_population(self, num_generation, user_gen_order=UserGenOrder.SHUFFLE):

        population = {}

        # Generating for num_generation times users
        for generation_index in range(num_generation):
            population[generation_index] = self.generate_users(user_gen_order)

        return population

    # Generates users
    def generate_users(self, user_gen_order=UserGenOrder.SHUFFLE):

        # Generating users in random locations
        users = pd.DataFrame()
        for user_index in range(self.tot_users):
            users.at[user_index, 'x'] = random.random() * self.max_x
            users.at[user_index, 'y'] = random.random() * self.max_y
            users.at[user_index, 'u_index'] = int(user_index)

            # Assigning user type according to current index and userGenOrder value
            if (user_index >= self.num_mmb_users and user_gen_order is UserGenOrder.FIRST_MMB) or \
                    (user_index < self.num_iot_users and user_gen_order is not UserGenOrder.FIRST_MMB):
                users.at[user_index, 'type'] = IOT_TYPE
            else:
                users.at[user_index, 'type'] = MMB_TYPE

        # Shuffling users
        if user_gen_order is UserGenOrder.SHUFFLE:
            users = users.sample(frac=1).reset_index(drop=True)

        return users

    @staticmethod
    def order_users(users, user_gen_order):

        if user_gen_order is UserGenOrder.SHUFFLE:
            # Shuffling users
            ordered_users = users.sample(frac=1).reset_index(drop=True)
        else:
            if user_gen_order is UserGenOrder.FIRST_IOT:
                ascending = True
            else:
                ascending = False

            # Sorting users by type
            ordered_users = users.sort_values(by=['type'], ascending=ascending)

        ordered_users.reset_index(inplace=True)

        return ordered_users


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

        # Eb/No t
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

    def is_visible_by_path_loss(self, path_loss):

        # Computing Eb/N0
        ebn0 = self.get_eb_n0(path_loss)

        return ebn0 >= self.ebn0_t

    def is_visible_by_position(self, user_x, user_y, bs_x, bs_y, gamma=3):

        # Computing distance
        dist = compute_distance(user_x, user_y, bs_x, bs_y)

        # Computing path loss
        path_loss = compute_path_loss(dist, gamma)

        return self.is_visible_by_path_loss(path_loss)

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

    # Gets the bit rate by user/bs index [Mbit/s]
    def get_bit_rate_by_bs(self, user_x, user_y, bs, bs_index):
        if bs_index is not None:
            base_station = bs.iloc[bs_index]
            return self.get_bit_rate_by_position(user_x, user_y, base_station.x, base_station.y)
        else:
            return 0


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
            if conn.bs_index is not None:
                bsi = int(conn.bs_index)

                # Getting user/base station locations
                x1, x2 = users.iloc[int(conn.u_index)].x, bs.iloc[bsi].x
                y1, y2 = users.iloc[int(conn.u_index)].y, bs.iloc[bsi].y

                if not math.isinf(x1) and not math.isinf(y1):
                    ax.plot([y1, y2], [x1, x2], color=get_color(conn.type), alpha=0.2)
            else:
                # Highlighting users not connected
                pos = (users.loc[int(ui), 'y'], users.loc[int(ui), 'x'])
                circle = plt.Circle(pos, circle_size, color='r', fill=False, alpha=0.8)
                ax.add_artist(circle)


def evaluate_solution(connections, tot_users):

    # Getting the total network bit rate
    tot_bit_rate = connections.bit_rate.sum()

    # Getting percentage of users non connected
    disc_users_percent = connections[connections.bs_index.isnull()].shape[0] / tot_users * 100

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

    return pd.DataFrame(path_losses, columns=['bs_index', 'u_index', 'type', 'x', 'y', 'path_loss'])


def compute_bs_free_prb(bs, connections):

    bs['free_prb'] = MAX_PRB

    for connection_index, connection in connections.iterrows():
        if connection['bs_index'] is not None:
            bs_index = int(connection['bs_index'])
            bs.at[bs_index, 'free_prb'] -= get_user(connection.type).num_prb

    return bs


def get_free_prb_bs(bs, connections):
    bs = compute_bs_free_prb(bs, connections)
    return bs[bs['free_prb'] > 0]


# Computes BS/users minimum path loss connections
def compute_min_path_losses_connections(bs, path_losses, consider_max_prb=True):

    # Setting to max the PRB available for the base stations
    bs['free_prb'] = MAX_PRB

    if is_bs_exclusive:
        bs['bs_type'] = None

    connections = pd.DataFrame({'u_index': path_losses.u_index.unique()})
    connections['bs_index'] = None
    connections['type'] = None
    connections['path_loss'] = math.inf
    connections['bit_rate'] = 0.0
    connections['x'] = None
    connections['y'] = None

    if consider_max_prb:
        user_path_losses = path_losses.groupby('u_index')
    else:
        user_path_losses = path_losses.sort_values(by='path_loss').groupby('u_index')

    # Iterating on user path loss grouped by index
    for user_index, user_path_loss in user_path_losses:

        # Iterating on single users
        for row_index, row in user_path_loss.iterrows():

            # Computing free PRB for current base station
            free_prb = bs.at[row.bs_index, 'free_prb'] - get_user(row.type).num_prb

            is_free_prb = free_prb >= 0 or not consider_max_prb

            # A BS is considered valid (just in case of exclusive BS hypothesis)
            # when its type is the same as the user
            is_valid_bs = True
            if is_bs_exclusive and bs.at[row.bs_index, 'bs_type'] is not None:

                is_valid_bs = bs.at[row.bs_index, 'bs_type'] == row.type

            # Computing visibility according to bs/users path loss
            is_visible = get_user(row.type).is_visible_by_path_loss(row.path_loss)

            connections.at[user_index, 'u_index'] = row.u_index
            connections.at[user_index, 'x'] = row.x
            connections.at[user_index, 'y'] = row.y
            connections.at[user_index, 'type'] = row.type
            connections.at[user_index, 'path_loss'] = row.path_loss

            if is_free_prb and is_valid_bs and is_visible:

                # Reducing the available PRB for current base station
                bs.at[row.bs_index, 'free_prb'] = free_prb

                if is_bs_exclusive:
                    bs.at[row.bs_index, 'bs_type'] = row.type

                # Allocating current user (u_index) to current base station (row.bs_index)
                connections.at[user_index, 'bs_index'] = int(row.bs_index)
                connections.at[user_index, 'bit_rate'] = get_user(row.type).get_bit_rate(row.path_loss)

                break

    return connections


class UsersAllocationPredictor:

    def __init__(self, bs, training='genetic', classifier=svm.SVC()):

        # Base stations
        self.bs = bs

        # Training type (genetic or min_path_loss)
        self.training = training

        # Create a Classifier
        self.classifier = classifier

    def train(self, tot_populations, tot_users, iot_users_percent, training_percent=0.7):

        # Generating users
        users_generator = UsersGenerator(self.bs.x.max(), self.bs.y.max(), tot_users, iot_users_percent)

        if self.training == 'min_path_loss':
            # Minimum path loss training
            population = users_generator.generate_population(tot_populations)
        else:
            # Genetic training
            users = users_generator.generate_users()
            genetic_allocation = GeneticAllocation(self.bs, users)

            # Computing best fitness population
            population = genetic_allocation.generate_best_solutions(tot_populations, tot_users, iot_users_percent)

        # Computing min path loss connections
        for population_index in population:
            users = population[population_index]
            path_losses = compute_path_losses(self.bs, users)
            population[population_index] = compute_min_path_losses_connections(self.bs, path_losses, False)

        # Adapting users pandas data frame to python array
        X, y = self._adapt_population(population, True)

        # Pre-train data adapting
        X, y = UsersAllocationPredictor._pre_train_adapting(X, y)

        # Splitting data set into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_percent)

        # Training the model using the training sets
        self.classifier.fit(X_train, y_train)

        # Predicting the response for test dataset
        y_pred = self.classifier.predict(X_test)

        return metrics.accuracy_score(y_test, y_pred)

    def predict(self, users):

        # Computing min path loss connections
        path_losses = compute_path_losses(self.bs, users)
        users = compute_min_path_losses_connections(self.bs, path_losses, False)

        # Adapting users
        X, y = self._adapt_users(users)
        X, y = self._pre_train_adapting(X, y)

        # Predicting BS
        y = self.classifier.predict(X)

        X, y = UsersAllocationPredictor._post_train_adapting(X, y)

        # Assigning to users the BS prediction
        for index in range(0, len(y)):
            user = users.iloc[index]
            bs_index = int(y[index]) if y[index] != -1 else None
            users.at[index, 'bs_index'] = bs_index
            users.at[index, 'u_index'] = user['u_index']
            users.at[index, 'bit_rate'] = get_user(user.type).get_bit_rate_by_bs(user.x, user.y, self.bs, bs_index)

        users = users.where(pd.notnull(users), None)

        return users

    def _adapt_population(self, population, generate_also_bs_indexes=False):

        X = list()
        y = None
        if generate_also_bs_indexes:
            y = list()

        for population_index in population:
            users = population[population_index]

            for user_index, user in users.iterrows():
                x_user = list()
                x_user.insert(0, user.x)
                x_user.insert(1, user.y)
                x_user.insert(2, user.type)

                X.insert(population_index, x_user)

            if generate_also_bs_indexes:
                y += users['bs_index'].tolist()

        return X, y

    def _adapt_users(self, users, generate_also_bs_indexes=False):

        X = list()
        y = None
        if generate_also_bs_indexes:
            y = list()

        for user_index, user in users.iterrows():
            x_user = list()
            x_user.insert(0, user.x)
            x_user.insert(1, user.y)
            x_user.insert(2, user.type)

            X.append(x_user)

        if generate_also_bs_indexes:
            y += users['bs_index'].tolist()

        return X, y

    @staticmethod
    def _replace_values(array, from_value, to_value):
        for index in range(0, len(array)):

            if isinstance(array[index], list):
                UsersAllocationPredictor._replace_values(array[index], from_value, to_value)

            if array[index] is from_value:
                array[index] = to_value

        return array

    @staticmethod
    def _pre_train_adapting(X, y):
        X = UsersAllocationPredictor._replace_values(X, IOT_TYPE, 0)
        X = UsersAllocationPredictor._replace_values(X, MMB_TYPE, 1)
        if y is not None:
            y = UsersAllocationPredictor._replace_values(y, None, -1)
        return X, y

    @staticmethod
    def _post_train_adapting(X, y):
        X = UsersAllocationPredictor._replace_values(X, 0, IOT_TYPE)
        X = UsersAllocationPredictor._replace_values(X, 1, MMB_TYPE)
        if y is not None:
            y = UsersAllocationPredictor._replace_values(y, -1, None)
        return X, y


class GeneticAllocation:

    def __init__(self, bs, users):
        self.bs = bs
        self.users = users
        self.population = {}
        self.evolutions = {}

    def genetic_evolution(self, num_of_generation=10, population_size=100, mutation_rate=0.05):

        # Generating the first generation of the population
        self.generate_population(population_size)

        tot_users = len(self.users)

        next_gen_individual = None

        # Evolving population for num_of_generation generations
        for generation_index in range(num_of_generation):

            actual_generation = generation_index + 1
            print("Generation:", actual_generation)

            # Computing best two individuals
            first_best_individual, second_best_individual = self.fitness()

            # Crossing-over the best two individuals
            next_gen_individual = self.crossover(first_best_individual, second_best_individual)
            self.evolutions[generation_index] = next_gen_individual
            
            # Evaluating current solution
            solution = evaluate_solution(next_gen_individual, tot_users)

            print('Total bit rate:', solution['tot_bit_rate'], '[Mbit/s]')
            print('Users disconnected:', solution['disc_users_percent'], '%')

            if actual_generation < num_of_generation:

                # Mutating next generation individual prototype
                self.population = self.mutation(next_gen_individual, mutation_rate)

        return next_gen_individual

    def generate_best_solutions(self, num_solutions, tot_users, iot_users_percent):
        users_generator = UsersGenerator(self.bs.x.max(), self.bs.y.max(), tot_users, iot_users_percent)

        best_solutions = {}
        for population_index in range(0, num_solutions):
            self.users = users_generator.generate_users()
            solution = self.genetic_evolution(1, tot_users)
            best_solutions[population_index] = solution

        return best_solutions

    def generate_population(self, population_size=100):

        self.population = {}

        max_bs_index = len(self.bs)

        for individual_index in range(population_size):

            # Setting to max the PRB available for the base stations
            self.bs['free_prb'] = MAX_PRB

            individual = pd.DataFrame()

            # Assigning users to a random base station
            for user_index, user in self.users.iterrows():

                bs_index, bit_rate = self.__alloc_user_to_bs(user_index, user)

                individual.at[user_index, "u_index"] = int(user_index)
                individual.at[user_index, "type"] = user.type
                individual.at[user_index, 'x'] = user.x
                individual.at[user_index, 'y'] = user.y
                individual.at[user_index, 'bs_index'] = bs_index
                individual.at[user_index, 'bit_rate'] = bit_rate

            # TODO: capire perché i bs_index vengono impostati come NaN e non come None (poi rimuovere questa riga)
            individual = individual.where(pd.notnull(individual), None)

            # Inserting individual in population
            self.population[individual_index] = individual

    def fitness(self):
        users_size = len(self.users)

        population_fitness = pd.DataFrame()
        for individual_index in self.population:
            individual = self.population[individual_index]

            population_fitness.at[individual_index, 'tot_bit_rate'] = individual.bit_rate.sum()
            bs_not_alloc = individual[individual.bs_index.isnull()].shape[0]
            population_fitness.at[individual_index, 'users_disc_percent'] = bs_not_alloc / users_size * 100

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

    def mutation(self, next_gen_individual, mutation_rate=0.05):
        population_size = len(self.population)
        new_population = {}

        max_user_index = len(self.users)

        mutation_size = int(max_user_index*mutation_rate)

        for individual_index in range(population_size):

            mutant = next_gen_individual
            self.bs = compute_bs_free_prb(self.bs, mutant)
            user_indexes = list(range(0, max_user_index))
            mutation_done = 1

            while mutation_size >= mutation_done:

                # Getting user
                user_index = random.choice(user_indexes)
                user = mutant.iloc[user_index]

                if user['bs_index'] is not None:

                    # Getting bs index
                    bs_index = int(user['bs_index'])

                    # Disconnecting user
                    mutant.at[user_index, 'bs_index'] = None
                    mutant.at[user_index, 'bit_rate'] = 0.0

                    # Getting user PRB
                    user_prb = get_user(user.type).num_prb

                    # Updating free PRB
                    self.bs.at[bs_index, 'free_prb'] += user_prb
                    if self.bs.at[bs_index, 'free_prb'] > MAX_PRB:
                        self.bs.at[bs_index, 'free_prb'] = MAX_PRB

                    mutation_done += 1

            mmb_users = self.users[self.users.type == MMB_TYPE]
            iot_users = self.users[self.users.type == IOT_TYPE]

            mmb_user_indexes = list(mmb_users.index)
            iot_user_indexes = list(iot_users.index)

            mmb_user_num_prb = get_user(MMB_TYPE).num_prb

            max_mutation_num = mutation_size
            while len(get_free_prb_bs(self.bs, mutant)) and mutation_done < max_mutation_num:

                # Getting user index (first MMB)
                if mmb_user_indexes and self.bs['free_prb'].max() >= mmb_user_num_prb:
                    user_index = random.choice(mmb_user_indexes)
                    mmb_user_indexes.remove(user_index)
                else:
                    user_index = random.choice(iot_user_indexes)
                    iot_user_indexes.remove(user_index)

                user = mutant.iloc[user_index]

                if user['bs_index'] is None:

                    # Allocating user
                    bs_index, bit_rate = self.__alloc_user_to_bs(user_index, user)
                    if bs_index is not None:
                        mutant.at[user_index, 'bs_index'] = bs_index
                        mutant.at[user_index, 'bit_rate'] = bit_rate

                    mutation_done += 1

            new_population[individual_index] = mutant

        return new_population

    def __alloc_user_to_bs(self, user_index, user):

        result_bs_index = None
        result_bit_rate = 0.0

        max_bs_index = len(self.bs)
        bs_indexes = list(range(0, max_bs_index))

        while bs_indexes:
            bs_index = random.choice(bs_indexes)

            base_station = self.bs.iloc[bs_index]

            # Computing free PRB for current base station
            free_prb = self.bs.at[bs_index, 'free_prb'] - get_user(user.type).num_prb

            is_free_prb = free_prb >= 0

            # A BS is considered valid (just in case of exclusive BS hypothesis)
            # when its type is the same as the user
            is_valid_bs = True
            if is_bs_exclusive and base_station.type is not None:
                is_valid_bs = base_station.type == user.type

            # Computing path loss
            path_loss = compute_path_loss_by_distance(user.x, user.y, base_station.x, base_station.y)

            # Computing visibility between current bs and current user
            is_visible = get_user(user.type).is_visible_by_path_loss(path_loss)

            if is_free_prb and is_valid_bs and is_visible:

                # Reducing the available PRB for current base station
                self.bs.at[bs_index, 'free_prb'] = free_prb

                if is_bs_exclusive and self.bs.at[bs_index, 'type'] is None:
                    self.bs.at[bs_index, 'type'] = user.type

                # Populating result values
                result_bs_index = bs_index
                result_bit_rate = get_user(user.type).get_bit_rate(path_loss)

                break
            else:
                # BS not valid or not available physical blocks
                bs_indexes.remove(bs_index)

        return result_bs_index, result_bit_rate
