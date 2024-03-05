filePath = "input.txt"

with open(filePath, "r") as file:
    lines = file.readlines()

initialInfo = lines[0].split()

print(initialInfo[0] + " pizzas")
for team in range(1, len(initialInfo)):
    print(initialInfo[team] + " team of " + str(team + 1))

print("")
for line in lines[1:]:
    print(line.strip())
