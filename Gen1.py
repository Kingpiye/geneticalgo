
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

            if 0 <= r < N_ROW and 0 <= c < N_COL:
                neighbors.append(grid[r][c])

    return neighbors


for row in pop:
    for bacterium in row:
        bacterium.fitness.values = toolbox.evaluate(bacterium)

# --------------------------------
# 7. EVOLUTION
# --------------------------------
start_time = time.time()

for g in range(NGEN):

    # Create a copy of the old grid
    new_grid = [
        [toolbox.clone(bacterium) for bacterium in row]
        for row in pop
    ]

    for row in range(N_ROW):
        for col in range(N_COL):

            current = pop[row][col]

            # Get local neighbors
            neighbors = get_neighbors(pop, row, col)

            # Current bacterium + neighbors
            candidates = neighbors + [current]

            # Local tournament selection
            parent = toolbox.select(candidates, 1)[0]

            # Clone selected parent
            child = toolbox.clone(parent)

            # Mutation
            if random.random() < mutProb:
                toolbox.mutate(child)
                del child.fitness.values

            # Evaluate if necessary
            if not child.fitness.valid:
                child.fitness.values = toolbox.evaluate(child)

            # Keep original position
            new_grid[row][col] = child

    # Replace old grid
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
