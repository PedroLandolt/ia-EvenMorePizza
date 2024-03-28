import numpy as np
from itertools import combinations
from itertools import combinations_with_replacement
import random



def clearScreen():
    print("\033[H\033[J")
    return

#--------------------------------------------------------------------------------------------------------------#

def menuRunRandomSolution(teams, pizzas, fileName):

    clearScreen()
    print("Random Solution for " + fileName + " :")
    random_solution = randomSolution(pizzas, teams)
    print(random_solution)
    print("\n")
    print("Scores from each team:")
    scores = evaluateSolution(random_solution, pizzas)
    print(scores)
    print("Total score: " + str(sum(scores)))
    input("\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas, fileName)


def menuRunHillClimbing(teams, pizzas, fileName):

    clearScreen()
    print("Hill Climbing Solution for " + fileName + " :")
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
    menuChooseOptimization(teams, pizzas, fileName)


def menuRunSimulatedAnnealing(teams, pizzas, fileName):

    clearScreen()
    print("Simulated Annealing Solution for " + fileName + " :")
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
    menuChooseOptimization(teams, pizzas, fileName)


def menuRunTabuSearch(teams, pizzas, fileName):
    
    clearScreen()
    print("Tabu Search Solution for " + fileName + " :")
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
    menuChooseOptimization(teams, pizzas, fileName)


def menuRunGeneticAlgorithm(teams, pizzas, fileName):
    
    clearScreen()
    print("Genetic Algorithm Solution for " + fileName + " :")
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
    menuChooseOptimization(teams, pizzas, fileName)


def menuRunAll(teams, pizzas, fileName):

    clearScreen()
    print("Running all optimization algorithms with default parameters for " + fileName + " :")
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
    menuChooseOptimization(teams, pizzas, fileName)

def menuChooseOptimization(teams, pizzas, fileName):

    clearScreen()
    print("Choose an optimization algorithm for " + fileName + " :")
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
        menuRunRandomSolution(teams, pizzas, fileName)
    elif choice == "2":
        menuRunHillClimbing(teams, pizzas, fileName)
    elif choice == "3":
        menuRunSimulatedAnnealing(teams, pizzas, fileName)
    elif choice == "4":
        menuRunTabuSearch(teams, pizzas, fileName)
    elif choice == "5":
        menuRunGeneticAlgorithm(teams, pizzas, fileName)
    elif choice == "7":
        menuRunAll(teams, pizzas, fileName)
    else:
        print("\nInvalid choice!")
        print("Press Enter to go to main menu...")
        choice = input()
        menu()
    
    return
    

def menu():

    clearScreen()
    fileName= ""

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
    
    if choice not in ["1", "2", "3", "4", "5"]:
        print("\nInvalid choice!")
        print("Press Enter to go to main menu...")
        choice = input()
        menu()
    
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
        menu()
    
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

    fileName = filePath.split("/")[1].split(".")[0]

    menuChooseOptimization(teams, pizzas, fileName)

    return

#--------------------------------------------------------------------------------------------------------------#

### Evaluation Functions ###

### Evaluate Solution ###
def evaluateSolution(solution, pizzas):
    """Summary: Evaluate the solution by calculating the score of each team

    Args:
        solution (list): A list of tuples containing the team and the pizzas assigned to it
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)

    Returns:
        list: A list of scores for each team
    """
    scores = []
    for team in solution:
        team_pizzas = team[1]
        team_pizzas_set = set()
        for pizza in team_pizzas:
            team_pizzas_set.update(pizzas[pizza])
        scores.append(len(team_pizzas_set) ** 2)
    return scores


### Neighbour Solution ###
def getNeighbourSolution(solution):
    """Summary: Swap a pizza between two random teams

    Args:
        solution (list): A list of tuples containing the team and the pizzas assigned to it

    Returns:
        list: A new solution with a pizza swapped between two random teams
    """

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

### Neighbour Solution ###
def getNeighbourSolution2(solution, pizzas):
    """Summary: Remove a pizza from a team and assign a not used pizza to it"""
    
    new_solution = solution.copy()
    
    team_index = random.randint(0, len(new_solution) - 1)
    team = new_solution[team_index]
    
    if len(team[1]) == 0:
        return new_solution
    
    pizza = random.choice(team[1])
    team[1].remove(pizza)
    
    not_used_pizzas = set(pizzas.keys()) - set([pizza for team in new_solution for pizza in team[1]])
    new_pizza = random.choice(list(not_used_pizzas))
    team[1].append(new_pizza)
    
    return new_solution

### Neighbour Solution ###
def getNeighbourSolution3(solution, pizzas):
    """Summary : Use both neighbour solutions 1 and 2"""
    
    new_solution = solution.copy()
    
    if random.random() < 0.5:
        new_solution = getNeighbourSolution(new_solution)
    else:
        new_solution = getNeighbourSolution2(new_solution, pizzas)
    
    return new_solution

#--------------------------------------------------------------------------------------------------------------#

### Optimization Algorithms ###

### Random Solution ###
def randomSolution(pizzas, teams):
    """Summary: Randomly assign pizzas to teams

    Args:
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)
        teams (dictionary): Contains the number of pizzas each team can have (key: team, value: number of pizzas)

    Returns:
        list: A list of tuples containing the team and the pizzas assigned to it
    """

    lenPizza = len(pizzas)
    cpPizzas = pizzas.copy()
    cpTeams = teams.copy()

    lenPizza -= 1

    pizzasForTeams = []

    solution = []

    while (cpTeams != {}):
        team = random.choice(list(cpTeams.keys()))
        teamName = "Team with " + str(team) + " members"
        if lenPizza < int(team):
            break
        else:
            while int(cpTeams[team]) > 0:
                if lenPizza == 0:
                    break
                tempNumMembers = int(team)
                for _ in range(int(tempNumMembers)):
                    if lenPizza == 0:
                        break

                    #sort cpPizzas by number of ingredients
                    cpPizzas = dict(sorted(cpPizzas.items(), key=lambda item: len(item[1])))
                    pizza = list(cpPizzas.keys())[0]
                    pizzasForTeams.append(pizza)
                    cpPizzas.pop(pizza)
                    lenPizza -= 1
                solution.append((teamName, pizzasForTeams))
                cpTeams[team] = int(cpTeams[team]) - 1
                pizzasForTeams = []
            cpTeams.pop(team)

    return solution

### Hill Climbing ###
def hillClimbing(iterations, pizzas, teams):
    """Summary: Hill climbing optimization algorithm that consists of generating a random solution 
            and then improving it by swapping pizzas between teams until no improvement is made

    Args:
        iterations (int): Number of iterations
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)
        teams (dictionary): Contains the number of pizzas each team can have (key: team, value: number of pizzas)

    Returns:
        list: A list of tuples containing the team and the pizzas assigned to it
    """

    best_solution = randomSolution(pizzas, teams)
    best_score = sum(evaluateSolution(best_solution, pizzas))

    i = 0
    while i < iterations:
        i += 1
        #new_solution = getNeighbourSolution(best_solution)
        #new_solution = getNeighbourSolution2(best_solution, pizzas)
        new_solution = getNeighbourSolution3(best_solution, pizzas)
        new_score = sum(evaluateSolution(new_solution, pizzas))

        if new_score > best_score:
            best_solution = new_solution
            best_score = new_score
        else:
            break

    return best_solution


### Simulated Annealing ###
def simulatedAnnealing(pizzas, teams, initialTemperature, finalTemperature, coolingRate):
    """Summary: Simulated annealing optimization algorithm that consists of generating a random solution 
            and then improving it by swapping pizzas between teams until no improvement is made

    Args:
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)
        teams (dictionary): Contains the number of pizzas each team can have (key: team, value: number of pizzas)
        initialTemperature (float): Initial temperature
        finalTemperature (float): Final temperature
        coolingRate (float): Cooling rate

    Returns:
        list: A list of tuples containing the team and the pizzas assigned to it
    """

    currentSolution = randomSolution(pizzas, teams)
    currentScore = sum(evaluateSolution(currentSolution, pizzas))
    bestSolution = currentSolution
    bestScore = currentScore
    temperature = initialTemperature

    while temperature > finalTemperature:
        #newSolution = getNeighbourSolution(currentSolution)
        #newSolution = getNeighbourSolution2(currentSolution, pizzas)
        newSolution = getNeighbourSolution3(currentSolution, pizzas)

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


### Tabu Search ###
def tabuSearch(pizzas, teams, tabuListSize, maxIterations):
    """Summary: Tabu search optimization algorithm that consists of generating a random solution 
            and then improving it by swapping pizzas between teams until no improvement is made

    Args:
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)
        teams (dictionary): Contains the number of pizzas each team can have (key: team, value: number of pizzas)
        tabuListSize (int): Size of the tabu list
        maxIterations (int): Maximum number of iterations

    Returns:
        list: A list of tuples containing the team and the pizzas assigned to it
    """

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
        #newSolution = getNeighbourSolution(currentSolution)
        #newSolution = getNeighbourSolution2(currentSolution, pizzas)
        newSolution = getNeighbourSolution3(currentSolution, pizzas)

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


### Genetic Algorithm ###

### Initialize Population ###
def initialize_population(pizzas, teams, population_size):
    """Summary: Initialize the population with random solutions

    Args:
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)
        teams (dictionary): Contains the number of pizzas each team can have (key: team, value: number of pizzas)
        population_size (int): Size of the population

    Returns:
        list: A list of the population
    """

    population = []
    for _ in range(population_size):
        population.append(randomSolution(pizzas, teams))
    return population

### Tournament Selection ###
def tournament_selection(population, tournament_size, pizzas):
    """Summary: Select 2 parents using tournament selection

    Args:
        population (list): A list of solutions
        tournament_size (int): Size of the tournament
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)

    Returns:
        list: A list of 2 selected parents
    """

    selected_parents = []
    for _ in range(2):  # Select 2 parents
        tournament_candidates = random.sample(population, tournament_size)
        selected_parents.append(max(tournament_candidates, key=lambda x: sum(evaluateSolution(x, pizzas))))
    return selected_parents

### Single Point Crossover ###
def single_point_crossover(parents):
    """Summary: Perform single point crossover on the parents

    Args:
        parents (list): A list of 2 parents

    Returns:
        list: A list of 2 children
    """

    min_length = min(len(parents[0]), len(parents[1]))
    
    if min_length <= 1:
        # If either parent has length 1 or less, cannot perform crossover
        return parents

    crossover_point = random.randint(1, min_length - 1)
    child1 = parents[0][:crossover_point] + parents[1][crossover_point:]
    child2 = parents[1][:crossover_point] + parents[0][crossover_point:]
    return child1, child2

### Mutation ###
def mutation(solution, mutation_rate, pizzas):
    """Summary: Perform mutation on the solution

    Args:
        solution (list): A list of tuples containing the team and the pizzas assigned to it
        mutation_rate (float): Mutation rate
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)

    Returns:
        list: A mutated solution
    """

    mutated_solution = solution.copy()
    for i, team in enumerate(mutated_solution):
        if random.random() < mutation_rate:
            random_pizza_index = random.randint(0, len(team[1]) - 1)
            mutated_solution[i][1][random_pizza_index] = random.choice(list(pizzas.keys()))
    return mutated_solution

### Genetic Algorithm ###
def genetic_algorithm(pizzas, teams, population_size, tournament_size, mutation_rate, max_generations):
    """Summary: Genetic algorithm optimization algorithm that consists of generating a random population 
            and then improving it by performing selection, crossover, and mutation operations

    Args:
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)
        teams (dictionary): Contains the number of pizzas each team can have (key: team, value: number of pizzas)
        population_size (int): Size of the population
        tournament_size (int): Size of the tournament
        mutation_rate (float): Mutation rate
        max_generations (int): Maximum number of generations

    Returns:
        list: A list of tuples containing the team and the pizzas assigned to it
    """

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


#--------------------------------------------------------------------------------------------------------------#

### Main Function ###
if __name__ == "__main__":
    menu()

#--------------------------------------------------------------------------------------------------------------#
