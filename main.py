import numpy as np
import random
import matplotlib.pyplot as plt
import time


def clearScreen():
    for _ in range(100):
        print("\n")
    return

#--------------------------------------------------------------------------------------------------------------#

def menuRunRandomSolution(teams, pizzas, fileName):

    clearScreen()
    print("Random Solution for " + fileName + " :")
    random_solution = randomSolution(pizzas, teams)
    print("\nCheck the output folder for the solution file: " + fileName + "_random.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_random.out", "w") as file:
        file.write(str(len(random_solution)) + "\n")
        for team in random_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")

    scores = evaluateSolution(random_solution, pizzas)
    print("Total score: " + str(sum(scores)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in random_solution])))
    
    input("\n\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas, fileName)


def menuRunHillClimbing(teams, pizzas, fileName):
    clearScreen()
    print("Hill Climbing Solution for " + fileName + " :")
    
    run_multiple_times = input("\nDo you want to run the algorithm multiple times for comparison? (y/n): ").lower()
    
    if run_multiple_times == "yes" or run_multiple_times == "y":
        num_iterations = input("Enter the number of iterations for each run, separated by whitespaces (default: 10 100 1000): ")
        if num_iterations == "":
            iterations_list = [1000, 5000, 10000]
        else:
            iterations_list = [int(i) for i in num_iterations.split()]
            if len(iterations_list) < 2 or len(iterations_list) > 3:
                print("Please enter minimum 2 and maximum 3 values.")
                return

    else:
        num_iterations = int(input("Enter the number of iterations: "))
        iterations_list = [num_iterations]
    
    # Lists to store scores obtained for each set of parameters
    scores_list = []
    iterations_used_list = []

    best_solution = []
    best_score = float('-inf')
    best_iterations = 0
    
    for iterations in iterations_list:
        current_solution = hillClimbing(iterations, pizzas, teams)
        current_score = sum(evaluateSolution(current_solution, pizzas))
        
        # Store the scores obtained and the number of iterations used
        scores_list.append(current_score)
        iterations_used_list.append(iterations)
        
        if current_score > best_score:
            best_solution = current_solution
            best_score = current_score
            best_iterations = iterations

    
    print("\nCheck the output folder for the solution file: " + fileName + "_hill_climbing.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_hill_climbing.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")
    
    print("\nNumber of iterations that produced the best solution:", best_iterations)
    scores = evaluateSolution(best_solution, pizzas)
    print("Total score: " + str(sum(scores)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))
    
    # Plotting the comparison
    plt.figure(figsize=(10, 6))
    plt.bar([str(iterations) for iterations in iterations_used_list], scores_list)
    plt.title("Score Comparison for Different Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Score")
    plt.grid(axis='y')
    plt.show()
    
    input("\n\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas, fileName)



def menuRunSimulatedAnnealing(teams, pizzas, fileName):

    clearScreen()
    print("Simulated Annealing Solution for " + fileName + " :")
    
    run_multiple_times = input("\nDo you want to run the algorithm multiple times for comparison? (y/n): ").lower()
    
    if run_multiple_times == "yes" or run_multiple_times == "y":
        initialTemperatures_list = get_parameters("Enter the initial temperatures for each run, separated by whitespaces (default: 1000 5000 10000): ", [1000, 5000, 10000])
        if initialTemperatures_list is None:
            return
        
        finalTemperatures_list = get_parameters("Enter the final temperatures for each run, separated by whitespaces (default: 0.001 0.005 0.01): ", [0.001, 0.005, 0.01])
        if finalTemperatures_list is None:
            return
        
        coolingRates_list = get_parameters("Enter the cooling rates for each run, separated by whitespaces (default: 0.95 0.97 0.99): ", [0.95, 0.97, 0.99])
        if coolingRates_list is None:
            return
    else:
        initialTemperature = float(input("Enter the initial temperature: "))
        finalTemperature = float(input("Enter the final temperature: "))
        coolingRate = float(input("Enter the cooling rate: "))

    best_scores_list = []
    parameter_sets = []

    if run_multiple_times == "yes" or run_multiple_times == "y":
        for initialTemperature, finalTemperature, coolingRate in zip(initialTemperatures_list, finalTemperatures_list, coolingRates_list):
            best_solution = simulatedAnnealing(pizzas, teams, initialTemperature, finalTemperature, coolingRate)
            best_scores_list.append(sum(evaluateSolution(best_solution, pizzas)))
            parameter_sets.append((initialTemperature, finalTemperature, coolingRate))
    else:
        best_solution = simulatedAnnealing(pizzas, teams, initialTemperature, finalTemperature, coolingRate)
        best_scores_list.append(sum(evaluateSolution(best_solution, pizzas)))
        parameter_sets.append((initialTemperature, finalTemperature, coolingRate))

    print("\nCheck the output folder for the solution file: " + fileName + "_simulated_annealing.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_simulated_annealing.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")
            

    scores = evaluateSolution(best_solution, pizzas)
    print("\nTotal score: " + str(sum(scores)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))
    

    # Plotting the comparison
    plt.figure(figsize=(10, 6))
    plt.bar([str(parameters) for parameters in parameter_sets], best_scores_list)
    plt.title("Score Comparison for Different Parameter Sets")
    plt.xlabel("Parameter Set")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    input("\n\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas, fileName)



def menuRunTabuSearch(teams, pizzas, fileName):
    
    clearScreen()
    print("Tabu Search Solution for " + fileName + " :")
    
    run_multiple_times = input("\nDo you want to run the algorithm multiple times for comparison? (y/n): ").lower()
    
    if run_multiple_times == "yes" or run_multiple_times == "y":
        tabuListSizes = get_parameters("Enter the tabu list sizes for each run, separated by whitespaces (default: 15 20 40): ", [15, 20, 40], int)
        if tabuListSizes is None:
            return
        
        maxIterationsList = get_parameters("Enter the maximum iterations for each run, separated by whitespaces (default: 8000 10000 20000): ", [8000, 10000, 20000], int)
        if maxIterationsList is None:
            return
    else:
        tabuListSize = int(input("Enter the tabu list size: "))
        maxIterations = int(input("Enter the maximum number of iterations: "))

    best_scores_list = []
    parameter_sets = []
    
    if run_multiple_times == "yes" or run_multiple_times == "y":
        for tabuListSize, maxIterations in zip(tabuListSizes, maxIterationsList):
            best_solution = tabuSearch(pizzas, teams, tabuListSize, maxIterations)
            best_scores_list.append(sum(evaluateSolution(best_solution, pizzas)))
            parameter_sets.append((tabuListSize, maxIterations))
    else:
        best_solution = tabuSearch(pizzas, teams, tabuListSize, maxIterations)
        best_scores_list.append(sum(evaluateSolution(best_solution, pizzas)))
        parameter_sets.append((tabuListSize, maxIterations))

    print("\nCheck the output folder for the solution file: " + fileName + "_tabu_search.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_tabu_search.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")
            
    scores = evaluateSolution(best_solution, pizzas)
    
    print("Total score: " + str(sum(scores)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))

    # Plotting the comparison
    plt.figure(figsize=(10, 6))
    plt.bar([str(parameters) for parameters in parameter_sets], best_scores_list)
    plt.title("Score Comparison for Different Parameter Sets")
    plt.xlabel("Parameter Set")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    input("\n\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas, fileName)



def menuRunGeneticAlgorithm(teams, pizzas, fileName):
    
    clearScreen()
    print("Genetic Algorithm Solution for " + fileName + " :")
    
    run_multiple_times = input("\nDo you want to run the algorithm multiple times for comparison? (y/n): ").lower()

    # list of inputs - balanced values, large population, small tournament size, high mutation rate, extended generations
    
    parameter_descriptions = [
        "Balanced Values", 
        "Large Population", 
        "Small Tournament Size", 
        "High Mutation Rate", 
        "Extended Generations"
    ]

    if run_multiple_times == "yes" or run_multiple_times == "y":
        population_sizes = get_parameters_genetic("Enter the population sizes for each run, separated by whitespaces (default: 50 100 50 75 60): ", [50, 100, 50, 75, 60], int)
        if population_sizes is None:
            return
        
        tournament_sizes = get_parameters_genetic("Enter the tournament sizes for each run, separated by whitespaces (default: 5 10 3 8 6): ", [5, 10, 3, 8, 6], int)
        if tournament_sizes is None:
            return
        
        mutation_rates = get_parameters_genetic("Enter the mutation rates for each run, separated by whitespaces (default: 0.05 0.1 0.05 0.2 0.1): ", [0.05, 0.1, 0.05, 0.2, 0.1])
        if mutation_rates is None:
            return
        
        max_generations_list = get_parameters_genetic("Enter the maximum generations for each run, separated by whitespaces (default: 100 200 150 120 300): ", [100, 100, 100, 100, 100], int)
        if max_generations_list is None:
            return
    else:
        population_size = int(input("Enter the population size: "))
        tournament_size = int(input("Enter the tournament size: "))
        mutation_rate = float(input("Enter the mutation rate: "))
        max_generations = int(input("Enter the maximum number of generations: "))

    best_scores_list = []
    parameter_sets = []
    
    if run_multiple_times == "yes" or run_multiple_times == "y":
        for i, (population_size, tournament_size, mutation_rate, max_generations) in enumerate(zip(population_sizes, tournament_sizes, mutation_rates, max_generations_list)):
            best_solution = genetic_algorithm(pizzas, teams, population_size, tournament_size, mutation_rate, max_generations)
            best_scores_list.append(sum(evaluateSolution(best_solution, pizzas)))
            parameter_sets.append(parameter_descriptions[i])
    else:
        best_solution = genetic_algorithm(pizzas, teams, population_size, tournament_size, mutation_rate, max_generations)
        best_scores_list.append(sum(evaluateSolution(best_solution, pizzas)))
        parameter_sets.append(parameter_descriptions[0])

    print("\nCheck the output folder for the solution file: " + fileName + "_genetic.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_genetic.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")

    scores = evaluateSolution(best_solution, pizzas)

    print("Total score: " + str(sum(scores)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))

    # Plotting the comparison
    plt.figure(figsize=(10, 6))
    plt.bar(parameter_sets, best_scores_list)
    plt.title("Score Comparison for Different Parameter Sets")
    plt.xlabel("Parameter Set Description")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    input("\n\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas, fileName)



def menuRunHybridTabuGenetic(teams, pizzas, fileName):
    
    clearScreen()
    print("Hybrid Tabu Search + Genetic Algorithm Solution for " + fileName + " :")
    
    tabuListSize    = int(input("\nEnter the tabu list size: "))
    maxIterations   = int(input("Enter the maximum number of iterations: "))
    population_size = int(input("Enter the population size: "))
    tournament_size = int(input("Enter the tournament size: "))
    mutation_rate   = float(input("Enter the mutation rate: "))
    max_generations = int(input("Enter the maximum number of generations: "))
    best_solution = hybrid_tabu_genetic(pizzas, teams, tabuListSize, maxIterations, population_size, tournament_size, mutation_rate, max_generations)
    
    print("\nCheck the output folder for the solution file: " + fileName + "_hybrid_tabu_genetic.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_hybrid_tabu_genetic.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")
            
    scores = evaluateSolution(best_solution, pizzas)
    print("Total score: " + str(sum(scores)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))
    
    
    input("\n\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas, fileName)


def menuRunOptimalSolution(teams, pizzas, fileName):

    clearScreen()
    print("Optimal Solution for " + fileName + " :")
    best_solution = localOptimalSolution(pizzas, teams)
    
    print("\nCheck the output folder for the solution file: " + fileName + "_optimal.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_optimal.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")
            
    scores = evaluateSolution(best_solution, pizzas)
    print("Total score: " + str(sum(scores)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))
    
    input("\n\nPress Enter to continue...")
    menuChooseOptimization(teams, pizzas, fileName)

def menuRunAll(teams, pizzas, fileName):

    clearScreen()
    print("Running all optimization algorithms with default parameters for " + fileName + " :")
    print("\n")
    print("Random Solution:")
    random_solution = randomSolution(pizzas, teams)
    
    print("Check the output folder for the solution file: " + fileName + "_default_random.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension    
    with open("outputs/" + fileName + "_default_random.out", "w") as file:
        file.write(str(len(random_solution)) + "\n")
        for team in random_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")

    ra_score = evaluateSolution(random_solution, pizzas)
    print("Total score: " + str(sum(ra_score)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in random_solution])))


    print("\n")
    print("Hill Climbing Solution:")
    iterations = 10000
    best_solution = hillClimbing(iterations, pizzas, teams)
    
    print("Check the output folder for the solution file: " + fileName + "_default_hill_climbing.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_default_hill_climbing.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")
            
    hc_score = evaluateSolution(best_solution, pizzas)
    print("Total score: " + str(sum(hc_score)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))



    print("\n")
    print("Simulated Annealing Solution:")
    initialTemperature = 10000
    finalTemperature = 0.001
    coolingRate = 0.95
    best_solution = simulatedAnnealing(pizzas, teams, initialTemperature, finalTemperature, coolingRate)
    
    print("Check the output folder for the solution file: " + fileName + "_default_simulated_annealing.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_default_simulated_annealing.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")
            
    sa_score = evaluateSolution(best_solution, pizzas)
    print("Total score: " + str(sum(sa_score)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))


    print("\n")
    print("Tabu Search Solution:")
    tabuListSize = 20
    maxIterations = 10000
    best_solution = tabuSearch(pizzas, teams, tabuListSize, maxIterations)
    
    print("Check the output folder for the solution file: " + fileName + "_default_tabu_search.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_default_tabu_search.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")

    tabu_score = evaluateSolution(best_solution, pizzas)
    print("Total score: " + str(sum(tabu_score)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))



    print("\n")
    print("Genetic Algorithm Solution:")
    population_size = 100
    tournament_size = 10
    mutation_rate = 0.1
    max_generations = 100
    
    best_solution = genetic_algorithm(pizzas, teams, population_size, tournament_size, mutation_rate, max_generations)
    
    print("Check the output folder for the solution file: " + fileName + "_default_genetic_algorithm.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_default_genetic_algorithm.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")
            
    ga_score = evaluateSolution(best_solution, pizzas)
    print("Total score: " + str(sum(ga_score)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))
    
    
    
    print("\n")
    print("Hybrid Tabu Search + Genetic Algorithm Solution for " + fileName + " :")
    tabuListSize = 20       
    maxIterations = 10000   
    population_size = 100   
    tournament_size = 5     
    mutation_rate = 0.2     
    max_generations = 100   
    best_solution = hybrid_tabu_genetic(pizzas, teams, tabuListSize, maxIterations, population_size, tournament_size, mutation_rate, max_generations)
    
    print("Check the output folder for the solution file: " + fileName + "_hybrid_tabu_genetic.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_hybrid_tabu_genetic.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")
            
    hy_score = evaluateSolution(best_solution, pizzas)
    print("Total score: " + str(sum(hy_score)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))
    
    
    
    print("\n")
    print("Optimal Solution for " + fileName + " :")
    best_solution = localOptimalSolution(pizzas, teams)
    print("Check the output folder for the solution file: " + fileName + "_default_optimal.out")
    # Output the solution to a file in the outputs folder with the same name as the input file but with a .out extension
    with open("outputs/" + fileName + "_default_optimal.out", "w") as file:
        file.write(str(len(best_solution)) + "\n")
        for team in best_solution:
            file.write(str(team[0][10:11:1]) + " " + " ".join([str(pizza[0]) for pizza in team[1]]) + "\n")
            
    lo_scores = evaluateSolution(best_solution, pizzas)
    print("Total score: " + str(sum(lo_scores)))
    print("Number of pizzas assigned: " + str(sum([len(team[1]) for team in best_solution])))
    
    # List of algorithm names
    algorithms = ["Random Solution", "Hill Climbing", "Simulated Annealing", "Tabu Search", "Genetic Algorithm", "Hybrid Tabu Search + Genetic Algorithm", "Local Optimal Solution"]

    # List of scores corresponding to each algorithm
    scores = [sum(ra_score), sum(hc_score), sum(sa_score), sum(tabu_score), sum(ga_score), sum(hy_score), sum(lo_scores)]

    plt.figure(figsize=(10, 6))

    # Creating a bar chart
    plt.bar(algorithms, scores)

    plt.title("Score Comparison of Optimization Algorithms")
    plt.xlabel("Algorithm")
    plt.ylabel("Score")
    plt.grid(axis='y')
    plt.show()
        
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
    print("6. Hybrid Tabu Search + Genetic Algorithm")
    print("7. Local Optimal Solution")
    print("8. Run all with default parameters")
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
    elif choice == "6":
        menuRunHybridTabuGenetic(teams, pizzas, fileName)
    elif choice == "7":
        menuRunOptimalSolution(teams, pizzas, fileName)
    elif choice == "8":
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

    pizzasForTeams = []

    totalTeams = int(cpTeams[2]) + int(cpTeams[3]) + int(cpTeams[4])
    solution = []

    while (cpTeams != {} and lenPizza > 0 and totalTeams > 0):
        team = random.choice(list(cpTeams.keys()))
        teamName = "Team with " + str(team) + " members"
        if lenPizza < int(team):
            cpTeams.pop(team)
            continue
        else:
            if lenPizza == 0:
                break
            num_members = min(int(team), lenPizza)
            for _ in range(num_members):
                if lenPizza == 0:
                    break

                #sort cpPizzas by number of ingredients
                cpPizzas = dict(sorted(cpPizzas.items(), key=lambda item: len(item[1])))
                pizza = list(cpPizzas.keys())[0]
                pizzasForTeams.append(pizza)
                cpPizzas.pop(pizza)
                lenPizza -= 1
            solution.append((teamName, pizzasForTeams))
            totalTeams -= 1
            pizzasForTeams = []
            cpTeams[team] = str(int(cpTeams[team]) - 1)
            if cpTeams[team] == '0':
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
    start_time = time.time()
    new_solution = []
    new_score = 0
    

    for _ in range(iterations):
        new_solution = getNeighbourSolution3(best_solution, pizzas)
        new_score = sum(evaluateSolution(new_solution, pizzas))
        
        if new_score > best_score:
            best_solution = new_solution
            best_score = new_score

        else:
            break

    end_time = time.time()
    print("Time taken by Hill Climbing: ", "{:.4f}".format(end_time - start_time), " seconds")
            
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
    start_time = time.time()
    bestSolution = currentSolution
    bestScore = currentScore
    temperature = initialTemperature

    while temperature > finalTemperature:
        newSolution = getNeighbourSolution3(currentSolution, pizzas)
        newScore = sum(evaluateSolution(newSolution, pizzas))
        scoreDifference = newScore - currentScore

        # Accept the new solution if it's better or based on the acceptance probability
        if scoreDifference > 0 or np.random.rand() < np.exp(scoreDifference / temperature):
            currentSolution = newSolution
            currentScore = newScore

            # Update the best solution if applicable
            if newScore > bestScore:
                bestSolution = newSolution
                bestScore = newScore

        temperature *= 1 - coolingRate
        temperature = max(temperature, finalTemperature)

    end_time = time.time()
    print("Time taken by Annealing: ", "{:.4f}".format(end_time - start_time), " seconds")

    return bestSolution


### Tabu Search ###
def tabuSearch(pizzas, teams, tabuListSize, maxIterations):
    """
    Tabu search optimization algorithm for the Even More Pizzas problem.

    Args:
        pizzas (dict): Dictionary containing pizzas and their ingredients (key: pizza, value: list of ingredients).
        teams (dict): Dictionary containing the number of pizzas each team can have (key: team, value: number of pizzas).
        tabuListSize (int): Size of the tabu list.
        maxIterations (int): Maximum number of iterations.

    Returns:
        list: A list of tuples containing the team and the pizzas assigned to it.
    """

    # Initialize current solution and best solution
    currentSolution = bestSolution = randomSolution(pizzas, teams)
    bestScore = sum(evaluateSolution(currentSolution, pizzas))
    start_time = time.time()

    # Initialize tabu list with the initial solution
    tabuList = [currentSolution]

    for _ in range(maxIterations):
        # Generate a new neighbor solution
        newSolution = getNeighbourSolution3(currentSolution, pizzas)
        newScore = sum(evaluateSolution(newSolution, pizzas))

        # Check if the new solution is not tabu or leads to improvement
        if newSolution not in tabuList or newScore > bestScore:
            # Update the best solution if the new solution is better
            if newScore > bestScore:
                bestSolution = newSolution
                bestScore = newScore
                
            # Update the current solution
            currentSolution = newSolution
            
            # Add the new solution to the tabu list
            tabuList.append(newSolution)

            # Limit the size of the tabu list
            if len(tabuList) > tabuListSize:
                tabuList.pop(0)

    end_time = time.time()
    print("Time taken by Tabu: ", "{:.4f}".format(end_time - start_time), " seconds")

    return bestSolution


### Genetic Algorithm ###

### Initialize Population ###
def initialize_population(pizzas, teams, population_size):
    """Initialize the population with random solutions.

    Args:
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)
        teams (dictionary): Contains the number of pizzas each team can have (key: team, value: number of pizzas)
        population_size (int): Size of the population

    Returns:
        list: A list of the population
    """
    
    population = [randomSolution(pizzas, teams) for _ in range(population_size)]
    return population

### Tournament Selection ###
def tournament_selection(population, tournament_size, pizzas):
    
    """ Select 2 parents using tournament selection.

    Args:
        population (list): A list of solutions
        tournament_size (int): Size of the tournament
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)

    Returns:
        list: A list of 2 selected parents
    """
    tournament = random.sample(population, tournament_size)
    tournament_scores = [sum(evaluateSolution(solution, pizzas)) for solution in tournament]
    parent1 = tournament[tournament_scores.index(max(tournament_scores))]
    tournament.remove(parent1)
    tournament_scores.remove(max(tournament_scores))
    parent2 = tournament[tournament_scores.index(max(tournament_scores))]
    return parent1, parent2

### Single Point Crossover ###
def single_point_crossover(parents):
    """Perform single point crossover on the parents.

    Args:
        parents (list): A list of 2 parents

    Returns:
        list: A list of 2 children
    """
    
    parent1, parent2 = parents
    crossover_point = random.randint(0, min(len(parent1), len(parent2)))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

### Mutation ###
def mutation(solution, mutation_rate, pizzas):
    """Perform mutation on the solution.

    Args:
        solution (list): A list of tuples containing the team and the pizzas assigned to it
        mutation_rate (float): Mutation rate
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)

    Returns:
        list: A mutated solution
    """

    mutated_solution = solution.copy()
    for i, (team, pizzas_assigned) in enumerate(mutated_solution):
        if random.random() < mutation_rate:
            if pizzas_assigned:
                random_pizza = random.choice(pizzas_assigned)
                mutated_solution[i] = (team, [random_pizza])
    return mutated_solution

def genetic_algorithm(pizzas, teams, population_size, tournament_size, mutation_rate, max_generations):
    """Genetic algorithm optimization algorithm.

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

    start_time = time.time()

    population = initialize_population(pizzas, teams, population_size)
    best_solution = max(population, key=lambda x: sum(evaluateSolution(x, pizzas)))
    best_score = sum(evaluateSolution(best_solution, pizzas))

    for generation in range(max_generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = tournament_selection(population, tournament_size, pizzas)
            child1, child2 = single_point_crossover([parent1, parent2])
            new_population.extend([mutation(child1, mutation_rate, pizzas), mutation(child2, mutation_rate, pizzas)])
        population = new_population
        scores = [sum(evaluateSolution(solution, pizzas)) for solution in population]
        best_solution_index = scores.index(max(scores))
        if scores[best_solution_index] > best_score:
            best_solution = population[best_solution_index]
            best_score = scores[best_solution_index]

    end_time = time.time()
    print("Time taken by Genetic: ", "{:.4f}".format(end_time - start_time), " seconds")
    return best_solution


### Hybrid Algorithm (Tabu + Genetic) ###
def hybrid_tabu_genetic(pizzas, teams, tabuListSize, maxIterations, population_size, tournament_size, mutation_rate, max_generations):
    """
    Hybrid algorithm combining Tabu Search and Genetic Algorithm.

    Args:
        pizzas (dictionary): Contains the pizzas and their ingredients (key: pizza, value: list of ingredients)
        teams (dictionary): Contains the number of pizzas each team can have (key: team, value: number of pizzas)
        tabuListSize (int): Size of the tabu list in Tabu Search
        maxIterations (int): Maximum number of iterations in Tabu Search
        population_size (int): Size of the population in Genetic Algorithm
        tournament_size (int): Size of the tournament selection in Genetic Algorithm
        mutation_rate (float): Mutation rate in Genetic Algorithm
        max_generations (int): Maximum number of generations in Genetic Algorithm

    Returns:
        list: A list of tuples containing the team and the pizzas assigned to it
    """
    # Initialize with a solution from Tabu Search
    start_time = time.time()
    best_solution = tabuSearch(pizzas, teams, tabuListSize, maxIterations)
    best_score = sum(evaluateSolution(best_solution, pizzas))

    # Further optimize using Genetic Algorithm
    for _ in range(max_generations):
        # Generate initial population
        population = initialize_population(pizzas, teams, population_size)

        # Run Genetic Algorithm
        for _ in range(max_generations):
            new_population = []
            for _ in range(population_size // 2):
                parent1, parent2 = tournament_selection(population, tournament_size, pizzas)
                child1, child2 = single_point_crossover([parent1, parent2])
                new_population.extend([mutation(child1, mutation_rate, pizzas), mutation(child2, mutation_rate, pizzas)])
            population = new_population

            # Evaluate and update best solution
            scores = [sum(evaluateSolution(solution, pizzas)) for solution in population]
            best_solution_index = scores.index(max(scores))
            if scores[best_solution_index] > best_score:
                best_solution = population[best_solution_index]
                best_score = scores[best_solution_index]

    end_time = time.time()
    print("Time taken by Hybrid: ", "{:.4f}".format(end_time - start_time), " seconds")

    return best_solution


def localOptimalSolution(pizzas, teams):
    
    start_time = time.time()
    # Sort pizzas by number of ingredients
    pizzas = dict(sorted(pizzas.items(), key=lambda item: len(item[1]), reverse=True))
    
    solution = []
    
    # Assign pizzas to teams with 4 members
    for _ in range(int(teams[4])):
        if len(pizzas) < 4:
            break
        team_pizzas = []
        for _ in range(4):
            pizza = list(pizzas.keys())[0]
            team_pizzas.append(pizza)
            pizzas.pop(pizza)
        solution.append(("Team with 4 members", team_pizzas))
        
    # Assign pizzas to teams with 3 members
    for _ in range(int(teams[3])):
        if len(pizzas) < 3:
            break
        team_pizzas = []
        for _ in range(3):
            pizza = list(pizzas.keys())[0]
            team_pizzas.append(pizza)
            pizzas.pop(pizza)
        solution.append(("Team with 3 members", team_pizzas))
        
    # Assign pizzas to teams with 2 members
    for _ in range(int(teams[2])):
        if len(pizzas) < 2:
            break
        team_pizzas = []
        for _ in range(2):
            pizza = list(pizzas.keys())[0]
            team_pizzas.append(pizza)
            pizzas.pop(pizza)
        solution.append(("Team with 2 members", team_pizzas))

    end_time = time.time()
    print("Time taken by Local Optimal: ", "{:.4f}".format(end_time - start_time), " seconds")
        
    return solution



#--------------------------------------------------------------------------------------------------------------#



def get_parameters(prompt, default_values, data_type=float):
    user_input = input(prompt)
    if user_input == "":
        values_list = default_values
    else:
        values_list = [data_type(i) for i in user_input.split()]
        if len(values_list) < 2 or len(values_list) > 3:
            print(f"Please enter minimum 2 and maximum 3 values.")
            return None
    return values_list

def get_parameters_genetic(prompt, default_values, data_type=float):
    user_input = input(prompt)
    if user_input == "":
        values_list = default_values
    else:
        values_list = [data_type(i) for i in user_input.split()]
        if len(values_list) < 2 or len(values_list) > 5:
            print(f"Please enter minimum 2 and maximum 5 values.")
            return None
    return values_list


#--------------------------------------------------------------------------------------------------------------#

### Main Function ###
if __name__ == "__main__":
    menu()

#--------------------------------------------------------------------------------------------------------------#
