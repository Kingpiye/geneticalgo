import random
import time
import json
import multiprocessing

import matplotlib.pyplot as plt
import seaborn as sns

from deap import base, creator, tools


NGEN = 10
RANDOM_SEED = 42

chromosome_length = 400
popSize = 1000000

mutProb = 0.5
indpb = 0.01
cxProb = 0.01

# Anzahl der Worker-Prozesse:
# Ein logischer CPU-Kern bleibt möglichst für das Betriebssystem frei.
workers = max(1, multiprocessing.cpu_count() - 1)

# Anzahl der Individuen, die pro Arbeitspaket verteilt werden
CHUNKSIZE = 500


# --------------------------------
# FITNESS-KLASSEN
# --------------------------------

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


# --------------------------------
# UMGEBUNGSPARAMETER
# --------------------------------

# Diese Variablen werden auch von den Worker-Prozessen verwendet.
temp_env = None
ph_Env = None
optimal_nutrition_efficiency = None
salt_concentration = None


def init_worker(environment):
    """
    Wird einmal beim Start jedes Worker-Prozesses aufgerufen.
    Überträgt die Umgebungswerte in den jeweiligen Prozess.
    """
    global temp_env
    global ph_Env
    global optimal_nutrition_efficiency
    global salt_concentration

    temp_env = environment["temperature"]
    ph_Env = environment["ph"]
    optimal_nutrition_efficiency = environment["nutrition"]
    salt_concentration = environment["salt"]


# --------------------------------
# FITNESS-FUNKTION
# --------------------------------

def evaluate(individual):

    temp_fitness = abs(
        temp_env - sum(individual[0:100]) / 100
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

    return (
        temp_fitness,
        ph_fitness,
        nut_fitness,
        salt_fitness
    )


# --------------------------------
# TOOLBOX
# --------------------------------

def create_toolbox():

    toolbox = base.Toolbox()

    toolbox.register("zeroOrOne", random.randint, 0, 1)

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.zeroOrOne,
        n=chromosome_length
    )

    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual,
        n=popSize
    )

    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    return toolbox


# --------------------------------
# HAUPTPROGRAMM
# --------------------------------

def main():

    random.seed(RANDOM_SEED)

    # Umgebung auswählen
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
        environment = json.load(f)

    # Umgebungswerte auch im Hauptprozess setzen
    init_worker(environment)

    # Toolbox erstellen
    toolbox = create_toolbox()

    # Statistiken
    bestTempFitness = []
    meanTempFitness = []

    bestPhFitness = []
    meanPhFitness = []

    bestNutFitness = []
    meanNutFitness = []

    bestSaltFitness = []
    meanSaltFitness = []

    start_time = time.perf_counter()

    # Population erstellen
    print("Creating population...")
    pop = toolbox.population()

    print(f"Starting multiprocessing with {workers} workers...")
    print(f"Population size: {popSize}")
    print(f"Chunksize: {CHUNKSIZE}")

    # --------------------------------
    # MULTIPROCESSING
    # --------------------------------

    with multiprocessing.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(environment,)
    ) as pool:

        # DEAP verwendet ab jetzt pool.map statt der normalen map-Funktion.
        toolbox.register(
            "map",
            pool.map,
            chunksize=CHUNKSIZE
        )

        # --------------------------------
        # ERSTE FITNESSBERECHNUNG
        # --------------------------------

        print("Evaluating initial population...")

        fitnesses = toolbox.map(toolbox.evaluate, pop)

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # --------------------------------
        # GENERATIONSSCHLEIFE
        # --------------------------------

        for g in range(NGEN):

            print(f"Generation: {g + 1}/{NGEN}")

            # Selektion
            offspring = toolbox.select(pop, len(pop))

            # Individuen kopieren
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(
                offspring[::2],
                offspring[1::2]
            ):

                if random.random() < cxProb:

                    toolbox.mate(child1, child2)

                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:

                if random.random() < mutProb:

                    toolbox.mutate(mutant)

                    del mutant.fitness.values

            # --------------------------------
            # FITNESS MIT MULTIPROCESSING
            # --------------------------------

            invalid_ind = [
                ind for ind in offspring
                if not ind.fitness.valid
            ]

            fitnesses = toolbox.map(
                toolbox.evaluate,
                invalid_ind
            )

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Population ersetzen
            pop[:] = offspring

            # --------------------------------
            # STATISTIKEN
            # --------------------------------

            fitnessValues = [
                [ind.fitness.values[i] for ind in pop]
                for i in range(4)
            ]

            # Temperatur
            bestTempFitness.append(min(fitnessValues[0]))
            meanTempFitness.append(
                sum(fitnessValues[0]) / len(fitnessValues[0])
            )

            # pH-Wert
            bestPhFitness.append(min(fitnessValues[1]))
            meanPhFitness.append(
                sum(fitnessValues[1]) / len(fitnessValues[1])
            )

            # Nährstoffeffizienz
            bestNutFitness.append(min(fitnessValues[2]))
            meanNutFitness.append(
                sum(fitnessValues[2]) / len(fitnessValues[2])
            )

            # Salzkonzentration
            bestSaltFitness.append(min(fitnessValues[3]))
            meanSaltFitness.append(
                sum(fitnessValues[3]) / len(fitnessValues[3])
            )

    # --------------------------------
    # LAUFZEIT
    # --------------------------------

    end_time = time.perf_counter()
    calculation_time = end_time - start_time

    print(f"Calculation time: {calculation_time:.4f} seconds")

    # --------------------------------
    # DIAGRAMME
    # --------------------------------

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


# --------------------------------
# PROGRAMMSTART
# --------------------------------

if __name__ == "__main__":

    multiprocessing.freeze_support()
    main()