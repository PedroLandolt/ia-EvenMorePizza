import numpy as np
from itertools import combinations
from itertools import combinations_with_replacement
import random

def menu():
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
        print("\nScores from each team:")
        scores = evaluateSolution(best_solution, pizzas)
        print(scores)
        print("Total score: " + str(sum(scores)))
        

        input("\nPress Enter to continue...")

def subsetSum(target_sum, teams):
    team_counts = {int(k): int(v) for k, v in teams.items()}  # Convert keys and values to integers
    
    # Generate all possible combinations of teams
    combinations_list = []
    for team_size, count in team_counts.items():
        for combo in combinations_with_replacement([team_size], count):
            if sum(combo) <= target_sum:  # Ensure the combination doesn't exceed the target sum
                combinations_list.append(combo)
    
    # Calculate the sum of each combination and count the occurrences
    count_dict = {}
    for combo in combinations_list:
        combo_sum = sum(combo)
        count_dict[combo_sum] = count_dict.get(combo_sum, 0) + 1
    
    return count_dict

def randomSolution(pizzas, teams):
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
                for i in range(int(tempNumMembers)):
                    pizza = random.choice(list(cpPizzas.keys()))
                    pizzasForTeams.append(pizza)
                    cpPizzas.pop(pizza)
                    lenPizza -= 1
                solution.append((teamName, pizzasForTeams))
                cpTeams[team] = int(cpTeams[team]) - 1
                pizzasForTeams = []

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

    team1 = random.choice(new_solution)
    team2 = random.choice(new_solution)

    if len(new_solution) == 1:
        return new_solution
    
    while team1 == team2:
        team2 = random.choice(new_solution)

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

menu()


