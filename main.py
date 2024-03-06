filePath = "b_little_bit_of_everything.in"

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
