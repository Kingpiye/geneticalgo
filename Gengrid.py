
import random
import time
import json

import matplotlib.pyplot as plt
import seaborn as sns

from deap import base, creator, tools
environments = {
    "1": "enviroment1.json",
    "2": "enviroment2.json",
    "3": "enviroment3.json"
}

print("Select Environment (1, 2 or 3):")
choice = input().strip()

if choice not in environments:
    raise ValueError("Invalid environment selected!")

with open(environments[choice], "r") as f:
    enviroment = json.load(f)
temp_env = enviroment["temperature"]
ph_Env = enviroment["ph"]
optimal_nutrition_efficiency = enviroment["nutrition"]
salt_concentration = enviroment["salt"]


NGEN = 100
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
chromosome_length = 400

initial_COL = 100
initial_ROW = 100
space_COL = 1000
space_ROW = 1000
mutProb = 0.5
indpb = 0.01
transProb = 0.01
max_trans_laenge = 20
repb = 0.2

bestTempFitness = []
meanTempFitness = []

bestPhFitness = []
meanPhFitness = []

bestNutFitness = []
meanNutFitness = []

bestSaltFitness = []
meanSaltFitness = []
popSize = []

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



toolbox.register("row", tools.initRepeat, list, toolbox.individual, n=initial_ROW)
toolbox.register("population", tools.initRepeat, list, toolbox.row, n=initial_COL)
toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)

def transduction(recipient, donor):
    start = random.randint(0, len(recipient) - 1)

    segment_length = random.randint(
        1,
        max_trans_laenge
    )

    end = min(start + segment_length, len(recipient))
    recipient[start:end] = donor[start:end]

    return recipient
toolbox.register("transduction", transduction)
toolbox.register("select", tools.selTournament, tournsize=3)

start_time = time.time()



def evaluate(individual):
        temp_fitness = abs(
            temp_env - sum(individual[0:100])/ 100
        )

        ph_fitness = abs(
            ph_Env - sum(individual[100:200]) / 100
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

def collect_stats(grid):

    # Get all bacteria from the 2D grid
    pop = [
        grid[r][c]
        for r in range(space_ROW)
        for c in range(space_COL)
        if grid[r][c] is not None
    ]

    if not pop:
        return

    fitnessValues = [
        [ind.fitness.values[i] for ind in pop]
        for i in range(4)
    ]

    # Temperature
    bestTempFitness.append(min(fitnessValues[0]))
    meanTempFitness.append(sum(fitnessValues[0]) / len(pop))

    # pH
    bestPhFitness.append(min(fitnessValues[1]))
    meanPhFitness.append(sum(fitnessValues[1]) / len(pop))

    # Nutrition
    bestNutFitness.append(min(fitnessValues[2]))
    meanNutFitness.append(sum(fitnessValues[2]) / len(pop))

    # Salt
    bestSaltFitness.append(min(fitnessValues[3]))
    meanSaltFitness.append(sum(fitnessValues[3]) / len(pop))

    # Population size
    popSize.append(len(pop))

def get_neighbors(grid, row, col):

    neighbors = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:

            if dr == 0 and dc == 0:
                continue

            r = row + dr
            c = col + dc

            if (
                0 <= r < space_ROW
                and 0 <= c < space_COL
            ):
                bacterium = grid[r][c]

                if bacterium is not None:
                    neighbors.append(bacterium)

    return neighbors


toolbox.register("evaluate", evaluate)

pop = [
    toolbox.individual()
    for _ in range(initial_ROW * initial_COL)
]
Grid = [
    [None for _ in range(space_COL)]
    for _ in range(space_ROW)
]

positions = random.sample(
    [
        (r, c)
        for r in range(initial_ROW)
        for c in range(initial_COL)
    ],
    len(pop)
)

for individual, (row, col) in zip(pop, positions):

    individual.fitness.values = toolbox.evaluate(individual)

    Grid[row][col] = individual

collect_stats(Grid)
for g in range(NGEN):

    print(f"Generation: {g + 1}")

    # New grid for synchronous updating
    new_grid = [
        [None for _ in range(space_COL)]
        for _ in range(space_ROW)
    ]

    for r in range(space_ROW):
        for c in range(space_COL):

            bacterium = Grid[r][c]

            if bacterium is None:
                continue


            new_grid[r][c] = bacterium

            # Find free neighboring cells
            empty_nei_cells = []

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:

                    if dr == 0 and dc == 0:
                        continue

                    nr = r + dr
                    nc = c + dc

                    if (
                        0 <= nr < space_ROW
                        and 0 <= nc < space_COL
                        and Grid[nr][nc] is None
                        and new_grid[nr][nc] is None
                    ):
                        empty_nei_cells.append((nr, nc))

            # No free neighboring cells
            if not empty_nei_cells:
                continue

            # Reproduction probability
            if random.random() < repb:

                # Select free neighboring cell
                nr, nc = random.choice(empty_nei_cells)
                candidate = bacterium + get_neighbors(Grid, r, c)
                sel_ind= olbox.select(candidate, len(candidate))
                # Create offspring
                child = toolbox.clone(sel_ind)

                # Transduction from another neighboring bacterium
                donor_candidates = get_neighbors(Grid, r, c)

                if donor_candidates and random.random() < transProb:
                    donor = random.choice(donor_candidates)
                    toolbox.transduction(child, donor)
                # Mutation
                if random.random() < mutProb:

                    toolbox.mutate(child)

                # Calculate new fitness
                child.fitness.values = toolbox.evaluate(child)

                # Place offspring in new grid
                new_grid[nr][nc] = child


    # Replace old grid
    Grid = new_grid
    collect_stats(Grid)

    # Collect all bacteria
    pop = [
        Grid[r][c]
        for r in range(space_ROW)
        for c in range(space_COL)
        if Grid[r][c] is not None
    ]








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
