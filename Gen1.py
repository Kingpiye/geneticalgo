
import random
import time

import matplotlib.pyplot as plt
import seaborn as sns

from deap import base, creator, tools

# --------------------------------
# 1. PARAMETERS
# --------------------------------

temp_env = 40
ph_Env = 7
optimal_nutrition_efficiency = 1

NGEN = 50
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

chromosome_length = 100

N_COL = 100
N_ROW = 100

mutProb = 0.5
indpb = 0.1
transProb = 0

bestFitnessValues = []
meanFitnessValues = []



creator.create("FitnessMin", base.Fitness, weights=(-1.0,))


creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMin
    )

toolbox = base.Toolbox()

toolbox.register("zeroOrOne", random.randint, 0, 1)
toolbox.register("transduction", tools.cxTwoPoint)
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
toolbox.register("evaluate", lambda individual: (
    abs(temp_env - sum(individual)) / 1000,
))
start_time = time.time()

pop = toolbox.population()


def evaluate(individual):
    temp_fitness = abs(temp_env - sum(individual)) / 1000
    return (temp_fitness,)


toolbox.unregister("evaluate")
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
            if random.random() < transProb:
                donor = random.choice(neighbors)
                toolbox.transduction(child, donor)
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
    # 8. STATISTICS
    # --------------------------------

    fitnessValues = [
        bacterium.fitness.values[0]
        for row in pop
        for bacterium in row
    ]

    minFitness = min(fitnessValues)
    meanFitness = sum(fitnessValues) / len(fitnessValues)

    bestFitnessValues.append(minFitness)
    meanFitnessValues.append(meanFitness)

    print(
        f"Generation {g + 1}: "
        f"Best Fitness = {minFitness:.4f}, "
        f"Average Fitness = {meanFitness:.4f}"
    )

end_time = time.time()

calculation_time = end_time - start_time

print(f"Calculation time: {calculation_time:.4f} seconds")
sns.set_style("whitegrid")

plt.plot(bestFitnessValues, color="red", label="Best Fitness")
plt.plot(meanFitnessValues, color="green", label="Average Fitness")

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Local Evolution - Best and Average Fitness")

plt.legend()
plt.show()
