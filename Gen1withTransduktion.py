
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
    print("Invalid Environment")

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

popSize = 5000
max_trans_laenge = 20
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

def transduction(recipient, donor):
    start = random.randint(0, len(recipient) - 1)
    segment_length = random.randint(1,max_trans_laenge)
    end = min(start + segment_length, len(recipient))
    recipient[start:end] = donor[start:end]
    return recipient
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=popSize)
toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
toolbox.register("select", tools.selTournament, tournsize=3)
start_time = time.time()

pop = toolbox.population()
toolbox.register("transduction", transduction)

def evaluate(individual):
    temp_fitness = abs(
        (temp_env - sum(individual[0:100])) / 100
    )

    ph_fitness = abs(
        (ph_Env - sum(individual[100:200])) / 100
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
    # Transduction
    if random.random() < transProb:
        donor = random.choice(offspring)
        recipient_candidates = [
            ind for ind in offspring
            if ind is not donor
        ]
        if recipient_candidates:
            recipient = random.choice(recipient_candidates)
            tran_recipient = toolbox.clone(recipient)
            toolbox.transduction(tran_recipient, donor)
            del tran_recipient.fitness.values
            offspring[offspring.index(recipient)] = tran_recipient
        for mutant in offspring:
            if random.random() < mutProb:
                toolbox.mutate(mutant)
                del mutant.fitness.values


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
