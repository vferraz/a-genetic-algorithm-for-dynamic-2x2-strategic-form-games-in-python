# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:33:16 2017

@author: Vinícius Ferraz / Thomas Pitz

Genetic Algorithms on 2x2 Strategic Form Games - Agent Based Simuation Model

Version 1.1

"""

#%% Imports

import random as rd
import numpy as np
import pandas as pd
import nashpy as nash
import copy

# import winsound

import time

start_time = time.time()

import complete_pools_edit as cp  # data source

#%%

# Parameters - Strategy Selection: ################################

pool_of_games = cp.data["L4"]
# entire pool = complete
# layers = L1, L2, L3, L4
# families = winwin, biased, secondbest, unfair, prison, cyclic

strategy = 1  # Strategy selector
# 1 = nash
# 2 = hurwicz
# 3 = random

eq_sel = 2  # Nash Equilibrium Selection Rule
# 1 = Payoff dominant (max)
# 2 = Random

alpha = 1  # Hurwicz alpha
# between 0 (risk averse) and 1 (risk prone)

# Parameters - Genetic Algorithm: ################################

evolution_target = 10  # iterations

mutation_rate = 0.01  # mutation probability

crossover_points = 1  # 1 or 2

n = 1  # exclusion rate (sorted population slicing position)


#%% ALGORITHM:


# When strategy = 1 - Nash Equilibrium strategy selection (payoff dominance and random)


def NashEquilibrium(pool_of_games):
    gci = 0
    fitness_list, game_pool, output_pool, eq_l, r_l, c_l = [], [], [], [], [], []

    for game in range(len(pool_of_games)):
        vector = pool_of_games[game][0]
        pr1 = vector[:2]  # player ROW's 1st set of strategies
        pr2 = vector[2:4]  # player ROW's 2nd set of strategies
        pc1 = vector[4:6]  # player COL's 1st set of strategies
        pc2 = vector[6:]  # player COL's 2nd set of strategies
        matrix_r = [pr1, pr2]  # player ROW's strategy profile
        matrix_c = [pc1, pc2]  # player COL's strategy profile
        game_matrix = nash.Game(
            matrix_r, matrix_c
        )  # generation of the 2x2 strategic-form matrix
        nash_array = (
            game_matrix.support_enumeration()
        )  # Probability distribution array of ALL possible Nash Equilibria
        nash_list = list(nash_array)  # Array to list

        for s1, s2 in nash_list:
            r_eu = np.dot(np.dot(s1, matrix_r), s2)  # Player ROW - expected utility
            c_eu = np.dot(np.dot(s1, matrix_c), s2)  # Player COL - expected utility
            r_l.append(r_eu)
            c_l.append(c_eu)
            eq = [r_eu, c_eu]
            eq_l.append(eq)
            fitness = (
                r_eu + c_eu
            )  # aggregating the expected utility (fitness) for each player
            fitness_list.append(
                fitness
            )  # list containing all the fitness values for every possible Nash Equilibrium set of strategies

        if eq_sel == 1:
            utility = max(
                fitness_list
            )  # Equilibrium Selection Rule - Payoff Maximizig (Selten)
        elif eq_sel == 2:
            utility = rd.choice(fitness_list)  # Equilibrium Selection Rule - Random

        eq_count = len(fitness_list)
        eq_s = copy.copy(eq_l)
        r_s = copy.copy(r_l)
        c_s = copy.copy(c_l)
        fitness_list.clear(), eq_l.clear(), r_l.clear(), c_l.clear()
        nash_strategy = [
            vector,
            [utility],
            [gci],
        ]  # population individual: [[game],[fitness->max(eU)],[gci]]
        out_row = [
            generation,
            vector,
            utility,
            gci,
            eq_count,
            nash_list,
            eq_s,
            r_s,
            c_s,
            r_eu,
            c_eu,
        ]
        game_pool.append(
            nash_strategy
        )  # pool of al games and expected utilities based on the Nash selected strategies
        output_pool.append(out_row)
        gci = gci + 1  # increase the counter

    return (game_pool, output_pool)


# When strategy = 2 - Hurwicz criterion strategy selection


def HurwiczSelection(pool_of_games):
    gci = 0
    game_pool, output_pool = [], []

    for game in range(len(pool_of_games)):
        vector = pool_of_games[game][0]
        prs1 = vector[:2]  # player ROW's 1st set of strategies
        prs2 = vector[2:4]  # player ROW's 2nd set of strategies
        pcs1 = vector[4], vector[6]  # player COL's 1st set of strategies
        pcs2 = vector[5], vector[7]  # player COL's 2nd set of strategies
        matrix_r = prs1, prs2  # row strategy profile (vector)
        matrix_c = vector[4:6], vector[6:]  #    #player COLUMN's 1st set of strategies

        hc_prs1 = alpha * max(prs1) + (1 - alpha) * min(
            prs1
        )  # Hurwicz coefficient ROW strategy set 1
        hc_prs2 = alpha * max(prs2) + (1 - alpha) * min(
            prs2
        )  # Hurwicz coefficient ROW strategy set 2
        hc_pcs1 = alpha * max(pcs1) + (1 - alpha) * min(
            pcs1
        )  # Hurwicz coefficient COL strategy set 1
        hc_pcs2 = alpha * max(pcs2) + (1 - alpha) * min(
            pcs2
        )  # Hurwicz coefficient COL strategy set 2
        pa_1 = np.asarray(
            [1, 0]
        )  # probability distribution 1 - play 1st set of strategie
        pa_2 = np.asarray(
            [0, 1]
        )  # probability distribution 2 - play 2nd set of strategies

        r_probdist = HurwiczRule(
            hc_prs1, hc_prs2, pa_1, pa_2
        )  # ROW probability distribution over strategy profiles - SELECTION
        c_probdist = HurwiczRule(
            hc_pcs1, hc_pcs2, pa_1, pa_2
        )  # COL probability distribution over strategy profiles - SELECTION

        r_eu = np.dot(np.dot(r_probdist, matrix_r), c_probdist)  # ROW expected utility
        c_eu = np.dot(np.dot(r_probdist, matrix_c), c_probdist)  # COL expected utility
        fitness = (
            r_eu + c_eu
        )  # aggregating the expected utility (fitness) for each player
        hurwicz_strategy = [vector, [fitness], [gci]]
        out_row = [
            generation,
            vector,
            fitness,
            gci,
            hc_prs1,
            hc_prs2,
            hc_pcs1,
            hc_pcs2,
            r_probdist,
            c_probdist,
            r_eu,
            c_eu,
        ]
        game_pool.append(
            hurwicz_strategy
        )  # pool of al games and expected utilities based on the Nash selected strategies
        output_pool.append(out_row)
        gci = gci + 1  # increase the counter

    return (game_pool, output_pool)


def HurwiczRule(hc_1, hc_2, pa_1, pa_2):
    if hc_1 > hc_2:
        return pa_1
    elif hc_1 < hc_2:
        return pa_2
    else:
        pd_sp, probdist = [
            pa_1,
            pa_2,
        ], None  # probability distribution over the strategy profiles
        for value in range(len(pd_sp)):
            probdist = rd.choice(
                pd_sp
            )  # random choice if Hurwicz coefficients are the same
        return probdist


# When strategy = 3 -Random strategy selection


def RandomSelection(pool_of_games):
    gci = 0
    game_pool, output_pool = [], []

    for game in range(len(pool_of_games)):
        vector = pool_of_games[game][0]
        prs1 = vector[:2]  # player ROW's 1st set of strategies
        prs2 = vector[2:4]  # player ROW's 2nd set of strategies
        pcs1 = vector[4], vector[6]  # player COLUMN's 1st set of strategies
        pcs2 = vector[5], vector[7]  # player COLUMN's 2nd set of strategies
        matrix_c = vector[4:6], vector[6:]  # player COL's matrix
        prsp = prs1, prs2  # row strategy profile (vector) / player ROW's matrix
        pcsp = pcs1, pcs2  # column strategy profile (vector)
        pa_1 = np.asarray(
            [1, 0]
        )  # probability distribution 1 - play 1st set of strategie
        pa_2 = np.asarray(
            [0, 1]
        )  # probability distribution 2 - play 2nd set of strategies
        rd_choice_r = rd.choice(prsp)  # ROW random choice
        rd_choice_c = rd.choice(pcsp)  # COL random choice

        r_probdist = (
            pa_1 if rd_choice_r == prs1 else pa_2
        )  # ROW probability distribution over strategy profiles - SELECTION
        c_probdist = (
            pa_1 if rd_choice_c == pcs1 else pa_2
        )  # COL probability distribution over strategy profiles - SELECTION

        r_eu = np.dot(np.dot(r_probdist, prsp), c_probdist)  # ROW expected utility
        c_eu = np.dot(np.dot(r_probdist, matrix_c), c_probdist)  # COL expected utility
        fitness = r_eu + c_eu
        random_strategy = [vector, [fitness], [gci]]
        out_row = [
            generation,
            vector,
            fitness,
            gci,
            rd_choice_r,
            rd_choice_c,
            r_probdist,
            c_probdist,
            r_eu,
            c_eu,
        ]
        game_pool.append(
            random_strategy
        )  # pool of al games and expected utilities based on the Nash selected strategies
        output_pool.append(out_row)
        gci = gci + 1  # increase the counter

    return (game_pool, output_pool)


# Strategy selection method alternation


def StrategySelection(pool_of_games):
    if strategy == 1:  # Nash Equilibrium
        return NashEquilibrium(pool_of_games)
    elif strategy == 2:  # Hurwicz
        return HurwiczSelection(pool_of_games)
    elif strategy == 3:  # Random
        return RandomSelection(pool_of_games)


# Binary conversion of game vectors


def BinaryConversion(game_pool):
    binary_list = []
    binary_index = {
        1: (0, 0),
        2: (0, 1),
        3: (1, 0),
        4: (1, 1),
    }  # possible binary combinations for the range of the strategy vectors
    for game in range(len(game_pool)):
        converted = [
            s for num in game_pool[game][0] for s in binary_index[num]
        ]  # generates binary converted individual
        binary_list.append(converted)  # creating a list of binary individuals
    return binary_list


# Creating a populatin of binary strings


def GeneratePopulation(
    game_pool, binary_list
):  # creates a new list replacing games in vector form for their binary form
    population = []  # list of all individuals
    for ind in range(len(binary_list)):
        individual = [
            binary_list[ind],
            game_pool[ind][1],
            game_pool[ind][2],
        ]  # generating individual = [[chromosome],[fitness],[counter]]
        population.append(
            individual
        )  # aggregating all new individuals in order to form the population
    return population


# Calculating weights for parents selection (fitness maximizing)


def WeightsSelection(population):
    weights_s = []
    for chromosome in range(len(population)):
        ind_fit_s = np.square(
            population[chromosome][1][0]
        )  # squared fitnes for weights distribution (scaling)
        weights_s.append(ind_fit_s)
    total_weight_s = sum(weights_s)
    weights_selection = [ind_fit_s / total_weight_s for ind_fit_s in weights_s]
    return weights_selection


# Parents election - Fitness Proportionate Selection method -  ROULETTE WHEEL


def FitnessProportionateSelection(population, weights_selection):
    population_clone = copy.copy(
        population
    )  # CLONE of the population to extract the selected individuals without repetions allowed
    selected_chromosomes = []

    selected1 = rd.choices(
        population_clone, weights=weights_selection, k=1
    )  # select 1st individual
    index1 = selected1[0][2][0]  # isolate the counter
    del weights_selection[index1]  # adjust the probability list
    del population_clone[index1]  # adjust the population

    selected2 = rd.choices(
        population_clone, weights=weights_selection, k=1
    )  # select 2nd individual
    index2 = selected2[0][2][0]  # isolate the counter

    if (
        index2 > index1
    ):  # adjusting the counter index to the new population(P) lenght (P-1)
        index2 -= 1
    del weights_selection[index2]
    del population_clone[index2]

    selected = selected1, selected2
    selected_chromosomes = [
        i[0] for i in selected
    ]  # flatten the list so the individuals follow the population structure
    return selected_chromosomes  # return selected individuals to generate children


# Crossover (two-point / random points) - Split selected individuals and combine the parts to form new individuals (offspring)


def Crossover(selected_chromosomes):
    rd.shuffle(selected_chromosomes)
    parent1, parent2 = selected_chromosomes[0][0], selected_chromosomes[1][0]

    if crossover_points == 1:  # 1-point crossover
        point1 = rd.randint(
            0, len(parent1) - 1
        )  # random index cut for the crossover - point 1
        child = parent1[:point1] + parent2[point1:]  # creation of the 1st child

    elif crossover_points == 2:  # 2-point crossover
        point1 = rd.randint(
            0, len(parent1)
        )  # random index cut for the crossover - point 1
        point2 = rd.randint(
            0, len(parent2)
        )  # random index cut for the crossover - point 2
        if point1 > point2:
            point1, point2 = point2, point1
        child = (
            parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        )  # creation of the 1st child
        point1 = rd.randint(
            0, len(parent1)
        )  # random index cut for the crossover - point 1
        child = parent1[:point1] + parent2[point1:]  # creation of the 1st child

    offspring = [[child]]
    return offspring


# Mutation


def Mutation(offspring):
    for i in range(len(offspring)):
        if (
            rd.random() < mutation_rate
        ):  # return the next random floating point number in the range 0-1
            gene = rd.randrange(len(offspring[i][0]))  # random bit selection
            offspring[i][0][gene] = 0 if offspring[i][0][gene] else 1  # mutate
    return offspring  # with possible mutation


# Calculating weights for deletion - inverse rule of the selection function


def WeightsDeletion(population):
    weights_d = []
    for chromosome in range(len(population)):
        ind_fit_d = population[chromosome][1][0]
        weights_d.append((10 - ind_fit_d) ** 2)
    total_weight_d = sum(weights_d)
    weights_deletion = [ind_fit_d / total_weight_d for ind_fit_d in weights_d]
    return weights_deletion


# Deletion - keeping population size constant


def PopulationAdjustment(population, offspring, weights_deletion):
    selected_del = rd.choices(
        population, weights=weights_deletion, k=n
    )  # select 1st individual
    index1 = selected_del[0][2][0]  # isolate the counter
    del population[index1]  # adjust the population
    new_population = population
    new_population.extend(
        offspring
    )  # concatenate the offspring to the population after the genetic operators
    return new_population


# Re-assembling game vectors as a population input for the next iteration


def StrategyVectorConversion(new_population):
    final_population = []
    binary_index = {
        (0, 0): 1,
        (0, 1): 2,
        (1, 0): 3,
        (1, 1): 4,
    }  # indexing the bit pairs to their respective value
    for game in new_population:
        vectorized_game = [
            binary_index[(i, j)] for i, j in zip(game[0][0::2], game[0][1::2])
        ]  # indexing every 2 bits to their respective value
        final_game_matrix = [vectorized_game]
        final_population.append(
            final_game_matrix
        )  # aggregating the individuals in strategy vector form
    return final_population


# Genetic Loop - starts the evolution process to reach the evolution target (number of specified iterations)


def GeneticLoop():
    global generation
    output = []
    final_population = pool_of_games
    print_ = lambda item: list(map(print, item))  # prints every element in list item
    for generation in range(1, evolution_target + 1):
        print("\n--------------------------------------------------")
        print("Generation " + str(generation) + ":\n")
        game_pool, output_pool = StrategySelection(final_population)
        print("\n--------------------------------------------------")
        print("Fitness assignment - Strategy Selection:\n")
        print_(game_pool)
        for item in output_pool:
            output.append(item)
        binary_list = BinaryConversion(game_pool)
        print("\n--------------------------------------------------")
        print("Game vectors converted to binary strings:\n")
        print_(binary_list)
        population = GeneratePopulation(game_pool, binary_list)
        print("\n--------------------------------------------------")
        print("Population:\n")
        print_(population)
        weights_selection = WeightsSelection(population)
        print("\n--------------------------------------------------")
        print("Relative probabilities of being selected:\n")
        print_(weights_selection)
        selected_chromosomes = FitnessProportionateSelection(
            population, weights_selection
        )
        print("\n--------------------------------------------------")
        print("Selected chromosomes:\n")
        print_(selected_chromosomes)
        offspring = Crossover(selected_chromosomes)
        print("\n--------------------------------------------------")
        print("Crossover - offspring generated from the selected parents: \n")
        print_(offspring)
        mutation = Mutation(offspring)
        print("\n--------------------------------------------------")
        print("Mutation:\n")
        print_(mutation)
        weights_deletion = WeightsDeletion(population)
        print("\n--------------------------------------------------")
        print("Relative probabilities of being deleted:\n")
        print_(weights_deletion)
        new_population = PopulationAdjustment(population, offspring, weights_deletion)
        print("\n--------------------------------------------------")
        print("Chromosome replacement - Adjusted population: \n")
        print_(new_population)
        final_population = StrategyVectorConversion(new_population)
        print("\n--------------------------------------------------")
        print(
            "Binary strings returned to strategy-vector form (final adjusted population): \n"
        )
        print_(final_population)

        # output generator
    poolname = cp.keys[cp.vals.index(pool_of_games)]

    df = pd.DataFrame(output)

    if strategy == 1:  # Nash Equilibrium
        df.columns = [
            "generation",
            "game",
            "fitness",
            "counter",
            "eq_count",
            "nash_list",
            "eq_list",
            "R",
            "C",
            "r_utility",
            "c_utility",
        ]
        if eq_sel == 1:
            eq_name = "pdom"
        elif eq_sel == 2:
            eq_name = "rand"
        filename = "nash" + poolname + eq_name + ".csv"

    elif strategy == 2:  # Hurwicz
        df.columns = [
            "generation",
            "game",
            "fitness",
            "counter",
            "hc_prs1",
            "hc_prs2",
            "hc_pcs1",
            "hc_pcs2",
            "prsp_prop_dist",
            "pcsp_prob_dist",
            "r_utility",
            "c_utility",
        ]
        filename = "hurwicz" + poolname + str(int(alpha * 100)) + ".csv"

    elif strategy == 3:  # Random
        df.columns = [
            "generation",
            "game",
            "fitness",
            "counter",
            "rd_choice_r",
            "rd_choice_c",
            "prsp_prop_dist",
            "pcsp_prob_dist",
            "r_utility",
            "c_utility",
        ]
        filename = "random" + poolname + ".csv"

    df.to_csv(filename, index=False)
    # winsound.Beep(frequency, duration)    #beep
    return final_population


# Algorithm-end beep parameters:
frequency = 5000  # Hz
duration = 200  # ms


evolution = GeneticLoop()

print("\n--------------------------------------------------")
print("Final Population (evolved from", evolution_target, "generations): \n")
for item in evolution:
    print(item[0])
print("\n")

print(
    "The simulaton took",
    (round(((time.time() - start_time) / 60), 2)),
    "minutes to run",
)
