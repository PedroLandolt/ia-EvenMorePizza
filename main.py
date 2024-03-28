import numpy as np
from itertools import combinations
from itertools import combinations_with_replacement
import random



def clearScreen():
    print("\033[H\033[J")
    return

"""
def menuAll():
    choice = ""
    while choice != "f":
        print("\nChoose a file to read:")
        print("a. a_example")
        print("b. b_little_bit_of_everything")
        print("c. c_many_ingredients")
        print("d. d_many_pizzas")
        print("e. e_many_teams")
        print("f. Exit")
        choice = input("\nEnter your choice: ")

        if choice == "f":
            print("Exiting program...")
            break

        file_mapping = {
            "a": "inputs/a_example.in",
            "b": "inputs/b_little_bit_of_everything.in",
            "c": "inputs/c_many_ingredients.in",
            "d": "inputs/d_many_pizzas.in",
            "e": "inputs/e_many_teams.in"
        }

        filePath = file_mapping.get(choice)
        if filePath is None:
            print("\nInvalid choice!")
            continue

        with open(filePath, "r") as file:
            lines = file.readlines()

        initialInfo = lines[0].split()

        teams = {}
        print(initialInfo[0] + " pizzas")
        for team in range(1, len(initialInfo)):
            teams.update({team + 1: initialInfo[team]})

        pizzas = {}
        i = 0
        for line in lines[1:]:
            items = line.split()
            pizzas.update({(i, items[0]): items[1:]})
            i += 1


        print(teams)
        print("")
        print(pizzas)


        # Random Solution
        print("\nRandom Solution:")
        random_solution = randomSolution(pizzas, teams)
        print(random_solution)
        print("Scores from each team:")
        scores = evaluateSolution(random_solution, pizzas)
        print(scores)
        print("Total score: " + str(sum(scores)))


        # Neighbour Solution
        print("\nNeighbour Solution:")
        new_solution = getNeighbourSolution(random_solution)
        print(new_solution)
        print("Scores from each team:")
        scores = evaluateSolution(new_solution, pizzas)
        print(scores)
        print("Total score: " + str(sum(scores)))


        # Hill Climbing Solution
        print("\nHill Climbing Solution:")
        iterations = 1000
        best_solution = hillClimbing(iterations, pizzas, teams)
        print(best_solution)
        print("Scores from each team:")
        scores = evaluateSolution(best_solution, pizzas)
        print(scores)
        print("Total score: " + str(sum(scores)))


        # Simulated Annealing Solution
        print("\nSimulated Annealing Solution:")
        initialTemperature = 1000
        finalTemperature = 0.1
        coolingRate = 0.003
        best_solution = simulatedAnnealing(pizzas, teams, initialTemperature, finalTemperature, coolingRate)
        print(best_solution)
        print("Scores from each team:")
        scores = evaluateSolution(best_solution, pizzas)
        print(scores)
        print("Total score: " + str(sum(scores)))

        # Tabu Search Solution
        print("\nTabu Search Solution:")
        tabuListSize = 10
        maxIterations = 1000
        best_solution = tabuSearch(pizzas, teams, tabuListSize, maxIterations)
        print(best_solution)
        print("Scores from each team:")
        scores = evaluateSolution(best_solution, pizzas)
        print(scores)
        print("Total score: " + str(sum(scores)))

        # Genetic Algorithm Solution
        print("\nGenetic Algorithm Solution:")
        population_size = 100
        tournament_size = 10
        mutation_rate = 0.1
        max_generations = 100
        best_solution = genetic_algorithm(pizzas, teams, population_size, tournament_size, mutation_rate, max_generations)
        print(best_solution)
        print("Scores from each team:")
        scores = evaluateSolution(best_solution, pizzas)
        print(scores)
        print("Total score: " + str(sum(scores)))
        

        input("\nPress Enter to continue...")

    return
"""

def menuRunRandomSolution(teams, pizzas):

    clearScreen()
    print("Random Solution:")
    random_solution = randomSolution(pizzas, teams)
    print(random_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(random_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    input("\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas)


def menuRunHillClimbing(teams, pizzas):

    clearScreen()
    print("Hill Climbing Solution:")
    print("\n")
    iterations = int(input("Enter the number of iterations: "))
    best_solution = hillClimbing(iterations, pizzas, teams)
    print(best_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(best_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    input("\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas)


def menuRunSimulatedAnnealing(teams, pizzas):

    clearScreen()
    print("Simulated Annealing Solution:")
    print("\n")
    initialTemperature = float(input("Enter the initial temperature: "))
    finalTemperature = float(input("Enter the final temperature: "))
    coolingRate = float(input("Enter the cooling rate: "))
    best_solution = simulatedAnnealing(pizzas, teams, initialTemperature, finalTemperature, coolingRate)
    print(best_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(best_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    input("\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas)


def menuRunTabuSearch(teams, pizzas):
    
    clearScreen()
    print("Tabu Search Solution:")
    print("\n")
    tabuListSize = int(input("Enter the tabu list size: "))
    maxIterations = int(input("Enter the maximum number of iterations: "))
    best_solution = tabuSearch(pizzas, teams, tabuListSize, maxIterations)
    print(best_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(best_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    input("\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas)


def menuRunGeneticAlgorithm(teams, pizzas):
    
    clearScreen()
    print("Genetic Algorithm Solution:")
    print("\n")
    population_size = int(input("Enter the population size: "))
    tournament_size = int(input("Enter the tournament size: "))
    mutation_rate = float(input("Enter the mutation rate: "))
    max_generations = int(input("Enter the maximum number of generations: "))
    best_solution = genetic_algorithm(pizzas, teams, population_size, tournament_size, mutation_rate, max_generations)
    print(best_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(best_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    input("\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas)


def menuRunAll(teams, pizzas):

    clearScreen()
    print("Running all optimization algorithms with default parameters...")
    print("\n")
    print("Random Solution:")
    random_solution = randomSolution(pizzas, teams)
    print(random_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(random_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    print("\n")
    print("Hill Climbing Solution:")
    iterations = 1000
    best_solution = hillClimbing(iterations, pizzas, teams)
    print(best_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(best_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    print("\n")
    print("Simulated Annealing Solution:")
    initialTemperature = 1000
    finalTemperature = 0.1
    coolingRate = 0.003
    best_solution = simulatedAnnealing(pizzas, teams, initialTemperature, finalTemperature, coolingRate)
    print(best_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(best_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    print("\n")
    print("Tabu Search Solution:")
    tabuListSize = 10
    maxIterations = 1000
    best_solution = tabuSearch(pizzas, teams, tabuListSize, maxIterations)
    print(best_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(best_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    print("\n")
    print("Genetic Algorithm Solution:")
    population_size = 100
    tournament_size = 10
    mutation_rate = 0.1
    max_generations = 100
    best_solution = genetic_algorithm(pizzas, teams, population_size, tournament_size, mutation_rate, max_generations)
    print(best_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(best_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    input("\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas)

def menuChooseOptimization(teams, pizzas):

    clearScreen()
    print("Choose an optimization algorithm:")
    print("\n")
    print("1. Random Solution")
    print("2. Hill Climbing")
    print("3. Simulated Annealing")
    print("4. Tabu Search")
    print("5. Genetic Algorithm")
    print("7. Run all with default parameters")
    print("\n")
    print("0. Go back")
    print("\n")
    choice = input("\nEnter your choice: ")

    if choice == "0":
        menu()
        return
    
    if choice == "1":
        menuRunRandomSolution(teams, pizzas)
    elif choice == "2":
        menuRunHillClimbing(teams, pizzas)
    elif choice == "3":
        menuRunSimulatedAnnealing(teams, pizzas)
    elif choice == "4":
        menuRunTabuSearch(teams, pizzas)
    elif choice == "5":
        menuRunGeneticAlgorithm(teams, pizzas)
    elif choice == "7":
        menuRunAll(teams, pizzas)
    else:
        print("\nInvalid choice!")
        print("Press Enter to go to main menu...")
        choice = input()
        menu()
    
    return
    

def menu():

    clearScreen()
    print("Choose a file to read:")
    print("\n")
    print("1. a_example")
    print("2. b_little_bit_of_everything")
    print("3. c_many_ingredients")
    print("4. d_many_pizzas")
    print("5. e_many_teams")
    print("\n")
    print("0. Exit")
    print("\n")
    choice = input("\nEnter your choice: ")

    if choice == "0":
        print("Exiting program...")
        return
    
    file_mapping = {
        "1": "inputs/a_example.in",
        "2": "inputs/b_little_bit_of_everything.in",
        "3": "inputs/c_many_ingredients.in",
        "4": "inputs/d_many_pizzas.in",
        "5": "inputs/e_many_teams.in"
    }

    filePath = file_mapping.get(choice)
    if filePath is None:
        print("\nInvalid choice!")
        return
    
    with open(filePath, "r") as file:
        lines = file.readlines()

    initialInfo = lines[0].split()

    teams = {}
    print(initialInfo[0] + " pizzas")
    for team in range(1, len(initialInfo)):
        teams.update({team + 1: initialInfo[team]})

    pizzas = {}
    i = 0
    for line in lines[1:]:
        items = line.split()
        pizzas.update({(i, items[0]): items[1:]})
        i += 1

    menuChooseOptimization(teams, pizzas)

    return

def randomSolution(pizzas, teams):
    # Randomly assign pizzas to teams
    solution = []
    for team, pizza_count in teams.items():
        team_pizzas = random.sample(pizzas.keys(), int(pizza_count))
        solution.append((team, team_pizzas))
    return solution


def evaluateSolution(solution, pizzas):
    scores = []
    for team in solution:
        team_pizzas = team[1]
        team_pizzas_set = set()
        for pizza in team_pizzas:
            team_pizzas_set.update(pizzas[pizza])
        scores.append(len(team_pizzas_set) ** 2)
    return scores

# swap a random pizza between two teams of the current solution
def getNeighbourSolution(solution):
    new_solution = solution.copy()

    if len(new_solution) < 2:
        return new_solution  # Not enough teams for swapping

    team1_index, team2_index = random.sample(range(len(new_solution)), 2)

    team1 = new_solution[team1_index]
    team2 = new_solution[team2_index]

    pizza1 = random.choice(team1[1])
    pizza2 = random.choice(team2[1])

    team1[1].remove(pizza1)
    team2[1].remove(pizza2)
    team1[1].append(pizza2)
    team2[1].append(pizza1)

    return new_solution

def hillClimbing(iterations, pizzas, teams):
    best_solution = randomSolution(pizzas, teams)
    best_score = sum(evaluateSolution(best_solution, pizzas))

    i = 0
    while i < iterations:
        i += 1
        new_solution = getNeighbourSolution(best_solution)
        new_score = sum(evaluateSolution(new_solution, pizzas))

        if new_score > best_score:
            best_solution = new_solution
            best_score = new_score
        else:
            break

    return best_solution

# simulated annealing
def simulatedAnnealing(pizzas, teams, initialTemperature, finalTemperature, coolingRate):
    currentSolution = randomSolution(pizzas, teams)
    currentScore = sum(evaluateSolution(currentSolution, pizzas))
    bestSolution = currentSolution
    bestScore = currentScore
    temperature = initialTemperature

    while temperature > finalTemperature:
        newSolution = getNeighbourSolution(currentSolution)
        newScore = sum(evaluateSolution(newSolution, pizzas))
        scoreDifference = newScore - currentScore

        if scoreDifference > 0:
            currentSolution = newSolution
            currentScore = newScore
            if newScore > bestScore:
                bestSolution = newSolution
                bestScore = newScore
        else:
            if random.random() < np.exp(scoreDifference / temperature):
                currentSolution = newSolution
                currentScore = newScore

        temperature *= 1 - coolingRate

    return bestSolution

# tabu search

def tabuSearch(pizzas, teams, tabuListSize, maxIterations):
    currentSolution = randomSolution(pizzas, teams)
    currentScore = sum(evaluateSolution(currentSolution, pizzas))
    bestSolution = currentSolution
    bestScore = currentScore
    tabuList = []
    tabuList.append(currentSolution)
    tabuListSize = tabuListSize
    maxIterations = maxIterations
    i = 0

    while i < maxIterations:
        i += 1
        newSolution = getNeighbourSolution(currentSolution)
        newScore = sum(evaluateSolution(newSolution, pizzas))

        if newSolution not in tabuList:
            if newScore > bestScore:
                bestSolution = newSolution
                bestScore = newScore
            currentSolution = newSolution
            currentScore = newScore
            tabuList.append(newSolution)
            if len(tabuList) > tabuListSize:
                tabuList.pop(0)

    return bestSolution

def initialize_population(pizzas, teams, population_size):
    population = []
    for _ in range(population_size):
        population.append(randomSolution(pizzas, teams))
    return population

def tournament_selection(population, tournament_size, pizzas):
    selected_parents = []
    for _ in range(2):  # Select 2 parents
        tournament_candidates = random.sample(population, tournament_size)
        selected_parents.append(max(tournament_candidates, key=lambda x: sum(evaluateSolution(x, pizzas))))
    return selected_parents

def single_point_crossover(parents):
    min_length = min(len(parents[0]), len(parents[1]))
    
    if min_length <= 1:
        # If either parent has length 1 or less, cannot perform crossover
        return parents

    crossover_point = random.randint(1, min_length - 1)
    child1 = parents[0][:crossover_point] + parents[1][crossover_point:]
    child2 = parents[1][:crossover_point] + parents[0][crossover_point:]
    return child1, child2


def mutation(solution, mutation_rate, pizzas):
    mutated_solution = solution.copy()
    for i, team in enumerate(mutated_solution):
        if random.random() < mutation_rate:
            random_pizza_index = random.randint(0, len(team[1]) - 1)
            mutated_solution[i][1][random_pizza_index] = random.choice(list(pizzas.keys()))
    return mutated_solution

def genetic_algorithm(pizzas, teams, population_size, tournament_size, mutation_rate, max_generations):
    population = initialize_population(pizzas, teams, population_size)
    best_solution = None
    best_score = float('-inf')
    generation = 0

    while generation < max_generations:
        new_population = []

        for _ in range(population_size // 2):
            parents = tournament_selection(population, tournament_size, pizzas)
            offspring1, offspring2 = single_point_crossover(parents)
            offspring1 = mutation(offspring1, mutation_rate, pizzas)
            offspring2 = mutation(offspring2, mutation_rate, pizzas)
            new_population.extend([offspring1, offspring2])

        population = new_population
        generation += 1

        # Evaluate population and update best solution
        for solution in population:
            score = sum(evaluateSolution(solution, pizzas))
            if score > best_score:
                best_solution = solution
                best_score = score

    return best_solution

menu()
