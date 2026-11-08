
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
temp_env = enviroment["temperature"] # angegebene Werte dürfen maximal eine Nachkommastelle haben, Fitnesskalkulation kann ansonsten nicht statfinden
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

bestFitness = []
meanFitness = []
EvaluatedFitnessStat = []
EvaluatedFitnessCount = 0
creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
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

def evaluate_details(individual):
    temp_fitness = abs(((temp_env + 273) / 573 ) - (sum(individual[0:100]) / 100))
    ph_fitness = abs((ph_Env / 14) - (sum(individual[100:200]) / 100))
    nut_fitness = abs(optimal_nutrition_efficiency - (sum(individual[200:300]) / 100))
    salt_fitness = abs(salt_concentration - (sum(individual[300:400]) / 100))
    return temp_fitness, ph_fitness, nut_fitness, salt_fitness


def evaluate(individual):
    details = evaluate_details(individual)
    general_fitness = sum(details)
    return (general_fitness,)

toolbox.register("evaluate", evaluate)


for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)
    EvaluatedFitnessCount += 1
EvaluatedFitnessStat.append(EvaluatedFitnessCount)
for g in range(NGEN):
    print(f"Generation: {g}")
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    # Transduction
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < transProb:

            if random.random() < 0.5:
                recipient, donor = child1, child2
            else:
                recipient, donor = child2, child1
            toolbox.transduction(recipient, donor)
            del recipient.fitness.values
    for mutant in offspring:
        if random.random() < mutProb:
            toolbox.mutate(mutant)
            del mutant.fitness.values


    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    EvaluatedFitnessCount += len(invalid_ind)
    EvaluatedFitnessStat.append(EvaluatedFitnessCount)
    # The population is entirely replaced by the offspring
    pop[:] = offspring


    # --------------------------------
    # STATISTICS
    # --------------------------------

    fitnessValues = [ind.fitness.values[0] for ind in pop]

    bestFitness.append(min(fitnessValues))
    meanFitness.append(sum(fitnessValues) / len(pop))

    details = [evaluate_details(ind) for ind in pop]

    temp_values = [v[0] for v in details]
    ph_values = [v[1] for v in details]
    nut_values = [v[2] for v in details]
    salt_values = [v[3] for v in details]

    bestTempFitness.append(min(temp_values))
    meanTempFitness.append(sum(temp_values) / len(pop))

    bestPhFitness.append(min(ph_values))
    meanPhFitness.append(sum(ph_values) / len(pop))

    bestNutFitness.append(min(nut_values))
    meanNutFitness.append(sum(nut_values) / len(pop))

    bestSaltFitness.append(min(salt_values))
    meanSaltFitness.append(sum(salt_values) / len(pop))
end_time = time.time()
calculation_time = end_time - start_time
print(f"Calculation time: {calculation_time:.4f} seconds")
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

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

axes[0, 2].plot(bestFitness, label="Best Fitness")
axes[0, 2].plot(meanFitness, label="Average Fitness")
axes[0, 2].set_title("General Fitness")

axes[1, 2].plot(range(NGEN + 1),EvaluatedFitnessStat,label="Cumulative Fitness Evaluations")
axes[1, 2].set_title("Cumulative Fitness Evaluations")

for ax in axes.flat:
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True)
axes[1, 2].set_ylabel("Number of Fitness Evaluations")
plt.tight_layout()
plt.show()
