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

        best_solution = initial_solution(pizzas, teams)
        print("\nBest Solution:")
        print(best_solution)
        
        input("\nPress Enter to continue...")



def initial_solution(pizzas, teams):
    solution = {}
    for (team_id, num_pizzas) in teams.items():
        solution[team_id] = random.sample(list(pizzas.keys()), min(int(num_pizzas), len(pizzas)))
    return solution


menu()