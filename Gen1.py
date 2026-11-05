
import random
import time
import json

import matplotlib.pyplot as plt
import seaborn as sns

from deap import base, creator, tools
print("Enter Enviroment: Enviroment1/Enviroment2/Enviroment3")
Enviroment = input()
if Enviroment == "Enviroment1":
    with open("enviroment1.json", "r") as f:
        enviroment = json.load(f)
elif Enviroment == "Enviroment2":
    with open("enviroment2.json", "r") as f:
        enviroment = json.load(f)
elif Enviroment == "Enviroment3":
    with open("enviroment3.json", "r") as f:
        enviroment = json.load(f)
else:
    print("No valid Enviroment selected")
temp_env = enviroment["temperature"]
ph_Env = enviroment["ph"]
optimal_nutrition_efficiency = enviroment["nutrition"]
salt_concentration = enviroment["salt"]


NGEN = 50
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
chromosome_length = 400

N_COL = 10
N_ROW = 10

mutProb = 0.5
indpb = 0.01
transProb = 0.01

bestTempFitness = []
meanTempFitness = []

bestPhFitness = []
meanPhFitness = []

bestNutFitness = []
meanNutFitness = []

bestSaltFitness = []
meanSaltFitness = []

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0 ))
creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMulti
    )

toolbox = base.Toolbox()

toolbox.register("zeroOrOne", random.randint, 0, 1)
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.zeroOrOne,
    n=chromosome_length
)

toolbox.register("row", tools.initRepeat, list, toolbox.individual, n=N_COL)
toolbox.register("population", tools.initRepeat, list, toolbox.row, n=N_ROW)
toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
toolbox.register("select", tools.selTournament, tournsize=3)
start_time = time.time()

def transduction(recipient, donor):

    chromosome_length = len(recipient)

    # Zufällige Startposition
    start = random.randint(0, chromosome_length - 1)

    # Zufällige Länge des übertragenen Abschnitts
    segment_length = random.randint(1, 20)

    # Ende des Abschnitts bestimmen
    end = min(start + segment_length, chromosome_length)

    # Genabschnitt übertragen
    recipient[start:end] = donor[start:end]

    return recipient

toolbox.register("transduction", transduction)
pop = toolbox.population()


def evaluate(individual):
        temp_fitness = abs(
            temp_env - sum(individual[0:100])
        ) / 100

        ph_fitness = abs(
            ph_Env - sum(individual[100:200]) / 100 * 14
        )

        nut_fitness = abs(
            sum(individual[200:300]) / 100
            - optimal_nutrition_efficiency
        )

        salt_fitness = abs(
            sum(individual[300:400]) / 100
            - salt_concentration
        )

        return temp_fitness, ph_fitness, nut_fitness, salt_fitness


toolbox.register("evaluate", evaluate)
def get_neighbors(grid, row, col):
    neighbors = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:

            if dr == 0 and dc == 0:
                continue

            r = row + dr
            c = col + dc

            if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                neighbors.append(grid[r][c])

    return neighbors



for g in range(NGEN):
    print(f"Generation: {g}")
    # Create an empty grid for the next generation
    new_grid = [
        [None for c in range(N_COL)]
        for r in range(N_ROW)
    ]

    # Go through every position
    for r in range(N_ROW):
        for c in range(N_COL):

            # Find local neighbors
            neighbors = get_neighbors(pop, r, c)

            # Get current bacterium
            current = pop[r][c]

            # Combine current bacterium and neighbors
            selList = [current] + neighbors

            # Select one parent
            parent = toolbox.select(selList, 1)[0]

            # Clone the selected parent
            child = toolbox.clone(parent)

            #Transduction
            # Transduction
            if random.random() < transProb:
                donor = random.choice(neighbors)

                toolbox.transduction(child, donor)

                del child.fitness.values
            # Apply mutation
            if random.random() < mutProb:
                toolbox.mutate(child)
                del child.fitness.values

            # Evaluate if fitness is invalid
            if not child.fitness.valid:
                child.fitness.values = toolbox.evaluate(child)

            # Place child at the same position
            new_grid[r][c] = child

    # Replace old population
    pop = new_grid

    # --------------------------------
    # STATISTICS
    # --------------------------------

    fitnessValues = [
        [bacterium.fitness.values[i] for row in pop for bacterium in row]
        for i in range(4)
    ]

    # Temperatur
    bestTempFitness.append(min(fitnessValues[0]))
    meanTempFitness.append(sum(fitnessValues[0]) / len(fitnessValues[0]))

    # pH-Wert
    bestPhFitness.append(min(fitnessValues[1]))
    meanPhFitness.append(sum(fitnessValues[1]) / len(fitnessValues[1]))

    # Nährstoffeffizienz
    bestNutFitness.append(min(fitnessValues[2]))
    meanNutFitness.append(sum(fitnessValues[2]) / len(fitnessValues[2]))

    # Salzkonzentration
    bestSaltFitness.append(min(fitnessValues[3]))
    meanSaltFitness.append(sum(fitnessValues[3]) / len(fitnessValues[3]))
end_time = time.time()

calculation_time = end_time - start_time

print(f"Calculation time: {calculation_time:.4f} seconds")
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Temperatur
axes[0, 0].plot(bestTempFitness, label="Best Fitness")
axes[0, 0].plot(meanTempFitness, label="Average Fitness")
axes[0, 0].set_title("Temperature Fitness")

# pH-Wert
axes[0, 1].plot(bestPhFitness, label="Best Fitness")
axes[0, 1].plot(meanPhFitness, label="Average Fitness")
axes[0, 1].set_title("pH Fitness")

# Nährstoffe
axes[1, 0].plot(bestNutFitness, label="Best Fitness")
axes[1, 0].plot(meanNutFitness, label="Average Fitness")
axes[1, 0].set_title("Nutrition Fitness")

# Salz
axes[1, 1].plot(bestSaltFitness, label="Best Fitness")
axes[1, 1].plot(meanSaltFitness, label="Average Fitness")
axes[1, 1].set_title("Salt Fitness")

for ax in axes.flat:
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
