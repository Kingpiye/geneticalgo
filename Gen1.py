
import random
import time
import json

import matplotlib.pyplot as plt
import seaborn as sns

from deap import base, creator, tools
print("Enter Environment: Enviroment1/Enviroment2/Enviroment3")
Environment = input()
if Environment == "Enviroment1":
    with open("enviroment1.json", "r") as f:
        enviroment = json.load(f)
elif Environment == "Enviroment2":
    with open("enviroment2.json", "r") as f:
        enviroment = json.load(f)
elif Environment == "Enviroment3":
    with open("enviroment3.json", "r") as f:
        enviroment = json.load(f)
else:
    print("No valid Environment selected")
temp_env = enviroment["temperature"]
ph_Env = enviroment["ph"]
optimal_nutrition_efficiency = enviroment["nutrition"]
salt_concentration = enviroment["salt"]


NGEN = 50
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
chromosome_length = 400

popSize = 1000

mutProb = 0.5
indpb = 0.01
cxProb = 0.01

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


toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=popSize)
toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=3)
start_time = time.time()

pop = toolbox.population()


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


toolbox.register("evaluate", evaluate)


for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)

for g in range(NGEN):
    print(f"Generation: {g}")
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxProb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutProb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring


    # --------------------------------
    # STATISTICS
    # --------------------------------

    fitnessValues = [[ind.fitness.values[i] for ind in pop]for i in range(4)]

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
