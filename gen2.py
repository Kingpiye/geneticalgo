import random
import time
import json
import matplotlib.pyplot as plt
from deap import base, creator, tools


# --------------------------------
# 1. PARAMETERS
# --------------------------------

print("Enter Environment: Enviroment1/Enviroment2/Enviroment3")
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
    print("No valid Environment selected")
temp_env = enviroment["temperature"]
ph_Env = enviroment["ph"]
optimal_nutrition_efficiency = enviroment["nutrition"]
salt_concentration = enviroment["salt"]


NGEN = 1000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

chromosome_length = 400

N_COL = 100
N_ROW = 100

INITIAL_POPULATION = 500

mutProb = 0.5
indpb = 0.01
transProb = 0.01
reproductionProb = 0.1

MAX_SEGMENT_LENGTH = 20


# --------------------------------
# 2. STATISTICS LISTS
# --------------------------------

populationHistory = []

bestTempFitness = []
meanTempFitness = []
worstTempFitness = []
stdTempFitness = []

bestPhFitness = []
meanPhFitness = []
worstPhFitness = []
stdPhFitness = []

bestNutFitness = []
meanNutFitness = []
worstNutFitness = []
stdNutFitness = []

bestSaltFitness = []
meanSaltFitness = []
worstSaltFitness = []
stdSaltFitness = []


# --------------------------------
# 3. DEAP SETUP
# --------------------------------

# Prevent errors when rerunning the script in the same Python session.
if not hasattr(creator, "FitnessMulti"):
    creator.create(
        "FitnessMulti",
        base.Fitness,
        weights=(-1.0, -1.0, -1.0, -1.0)
    )

if not hasattr(creator, "Individual"):
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

toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)


# --------------------------------
# 4. FITNESS FUNCTION
# --------------------------------

def evaluate(individual):
    temp_fitness = abs(
        temp_env - sum(individual[0:100])
    ) / 100

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

    return (
        temp_fitness,
        ph_fitness,
        nut_fitness,
        salt_fitness
    )


toolbox.register("evaluate", evaluate)


# --------------------------------
# 5. ABSTRACT TRANSDUCTION
# --------------------------------

def transduction(recipient, donor):
    start = random.randint(0, len(recipient) - 1)

    segment_length = random.randint(
        1,
        MAX_SEGMENT_LENGTH
    )

    end = min(start + segment_length, len(recipient))
    recipient[start:end] = donor[start:end]

    return recipient


toolbox.register("transduction", transduction)


# --------------------------------
# 6. LOCAL NEIGHBORHOOD
# --------------------------------

def get_neighbors(grid, row, col):
    neighbors = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue

            r = row + dr
            c = col + dc

            if (
                0 <= r < len(grid)
                and 0 <= c < len(grid[0])
            ):
                bacterium = grid[r][c]

                if bacterium is not None:
                    neighbors.append(bacterium)

    return neighbors


# --------------------------------
# 7. PARENT SELECTION
# --------------------------------

def select_parent(candidates):
    contestants = random.sample(
        candidates,
        min(3, len(candidates))
    )

    winner = contestants[0]

    for candidate in contestants[1:]:
        if candidate.fitness.dominates(winner.fitness):
            winner = candidate
        elif not winner.fitness.dominates(candidate.fitness):
            # Neither candidate dominates the other.
            if random.random() < 0.5:
                winner = candidate

    return winner


# --------------------------------
# 8. INITIAL POPULATION
# --------------------------------

pop = [
    [None for _ in range(N_COL)]
    for _ in range(N_ROW)
]

positions = random.sample(
    range(N_ROW * N_COL),
    INITIAL_POPULATION
)

for position in positions:
    r = position // N_COL
    c = position % N_COL

    bacterium = toolbox.individual()
    bacterium.fitness.values = toolbox.evaluate(bacterium)
    pop[r][c] = bacterium


# --------------------------------
# 9. STATISTICS FUNCTION
# --------------------------------

def record_statistics(grid, generation):
    living_bacteria = [
        bacterium
        for row in grid
        for bacterium in row
        if bacterium is not None
    ]

    population_count = len(living_bacteria)
    populationHistory.append(population_count)

    if population_count == 0:
        return

    # Each inner list contains one fitness objective for all living bacteria.
    fitness_values = [
        [b.fitness.values[i] for b in living_bacteria]
        for i in range(4)
    ]

    def add_statistics(values, best_list, mean_list, worst_list, std_list):
        mean_value = sum(values) / population_count

        best_list.append(min(values))
        mean_list.append(mean_value)
        worst_list.append(max(values))

        # Population standard deviation (division by N).
        variance = sum(
            (value - mean_value) ** 2 for value in values
        ) / population_count

        std_list.append(variance ** 0.5)

    # Temperature
    add_statistics(
        fitness_values[0],
        bestTempFitness, meanTempFitness,
        worstTempFitness, stdTempFitness
    )

    # pH
    add_statistics(
        fitness_values[1],
        bestPhFitness, meanPhFitness,
        worstPhFitness, stdPhFitness
    )

    # Nutrition
    add_statistics(
        fitness_values[2],
        bestNutFitness, meanNutFitness,
        worstNutFitness, stdNutFitness
    )

    # Salt
    add_statistics(
        fitness_values[3],
        bestSaltFitness, meanSaltFitness,
        worstSaltFitness, stdSaltFitness
    )

    # Print every statistic for this generation.
    print(f"\nGeneration: {generation}")
    print(f"Population: {population_count}")

    print(
        f"Temperature: Best={bestTempFitness[-1]:.4f}, "
        f"Mean={meanTempFitness[-1]:.4f}, "
        f"Worst={worstTempFitness[-1]:.4f}, "
        f"Std={stdTempFitness[-1]:.4f}"
    )

    print(
        f"pH: Best={bestPhFitness[-1]:.4f}, "
        f"Mean={meanPhFitness[-1]:.4f}, "
        f"Worst={worstPhFitness[-1]:.4f}, "
        f"Std={stdPhFitness[-1]:.4f}"
    )

    print(
        f"Nutrition: Best={bestNutFitness[-1]:.4f}, "
        f"Mean={meanNutFitness[-1]:.4f}, "
        f"Worst={worstNutFitness[-1]:.4f}, "
        f"Std={stdNutFitness[-1]:.4f}"
    )

    print(
        f"Salt: Best={bestSaltFitness[-1]:.4f}, "
        f"Mean={meanSaltFitness[-1]:.4f}, "
        f"Worst={worstSaltFitness[-1]:.4f}, "
        f"Std={stdSaltFitness[-1]:.4f}"
    )


# Record the initial population as generation 0.
record_statistics(pop, generation=0)


# --------------------------------
# 10. EVOLUTION
# --------------------------------

start_time = time.perf_counter()

for generation in range(1, NGEN + 1):

    # Stop if the grid is completely full.
    if populationHistory[-1] == N_ROW * N_COL:
        print("\nGrid is full. Stopping simulation.")
        break

    # Build the next grid separately so updates are synchronous.
    new_grid = [
        [None for _ in range(N_COL)]
        for _ in range(N_ROW)
    ]

    # Existing bacteria remain in their current positions.
    for r in range(N_ROW):
        for c in range(N_COL):
            if pop[r][c] is not None:
                new_grid[r][c] = pop[r][c]

    # Collect and shuffle all currently empty cells.
    empty_positions = [
        (r, c)
        for r in range(N_ROW)
        for c in range(N_COL)
        if pop[r][c] is None
    ]
    random.shuffle(empty_positions)

    # Try to fill empty cells with offspring.
    for r, c in empty_positions:
        neighbors = get_neighbors(pop, r, c)

        if not neighbors:
            continue

        if random.random() < reproductionProb:
            parent = select_parent(neighbors)

            # toolbox.clone is provided by DEAP's default Toolbox.
            child = toolbox.clone(parent)

            # Abstract gene-segment transfer from a local donor.
            possible_donors = [
                bacterium
                for bacterium in neighbors
                if bacterium is not parent
            ]

            if (
                possible_donors
                and random.random() < transProb
            ):
                donor = random.choice(possible_donors)
                toolbox.transduction(child, donor)
                del child.fitness.values

            # Mutation.
            if random.random() < mutProb:
                toolbox.mutate(child)
                del child.fitness.values

            # Recalculate fitness if the genome changed.
            if not child.fitness.valid:
                child.fitness.values = toolbox.evaluate(child)

            new_grid[r][c] = child

    # Replace the old grid and record the new generation.
    pop = new_grid
    record_statistics(pop, generation=generation)


end_time = time.perf_counter()
calculation_time = end_time - start_time

print("\n--------------------------------")
print(f"Calculation time: {calculation_time:.2f} seconds")
print(f"Final population: {populationHistory[-1]}")
print(f"Grid capacity: {N_ROW * N_COL}")


# --------------------------------
# 11. FITNESS STATISTIC PLOTS
# --------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_data = [
    (
        axes[0, 0], "Temperature Fitness",
        bestTempFitness, meanTempFitness,
        worstTempFitness, stdTempFitness
    ),
    (
        axes[0, 1], "pH Fitness",
        bestPhFitness, meanPhFitness,
        worstPhFitness, stdPhFitness
    ),
    (
        axes[1, 0], "Nutrition Fitness",
        bestNutFitness, meanNutFitness,
        worstNutFitness, stdNutFitness
    ),
    (
        axes[1, 1], "Salt Fitness",
        bestSaltFitness, meanSaltFitness,
        worstSaltFitness, stdSaltFitness
    )
]

generations_recorded = range(len(populationHistory))

for ax, title, best, mean, worst, std in plot_data:
    ax.plot(generations_recorded, best, label="Best (minimum)")
    ax.plot(generations_recorded, mean, label="Mean")
    ax.plot(generations_recorded, worst, label="Worst (maximum)")
    ax.plot(generations_recorded, std, label="Standard deviation")

    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (lower is better)")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


# --------------------------------
# 12. POPULATION GROWTH PLOT
# --------------------------------

plt.figure(figsize=(9, 5))
plt.plot(generations_recorded, populationHistory)

plt.xlabel("Generation")
plt.ylabel("Number of bacteria")
plt.title("Population Growth")
plt.grid(True)
plt.tight_layout()
plt.show()
