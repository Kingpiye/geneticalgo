import random
import time
import json
import multiprocessing

import matplotlib.pyplot as plt
import seaborn as sns

from deap import base, creator, tools


NGEN = 50
RANDOM_SEED = 42

chromosome_length = 400
popSize = 100000

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

if not hasattr(creator, "FitnessMin"):
    creator.create(
        "FitnessMin",
        base.Fitness,
        weights=(-1.0,)
    )

if not hasattr(creator, "Individual"):
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMin
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

def evaluate_details(individual):

    temp_fitness = abs(
        (temp_env + 273) / 573
        - sum(individual[0:100]) / 100
    )

    ph_fitness = abs(
        ph_Env / 14
        - sum(individual[100:200]) / 100
    )

    nut_fitness = abs(
        optimal_nutrition_efficiency
        - sum(individual[200:300]) / 100
    )

    salt_fitness = abs(
        salt_concentration
        - sum(individual[300:400]) / 100
    )

    return (
        temp_fitness,
        ph_fitness,
        nut_fitness,
        salt_fitness
    )


def evaluate(individual):

    details = evaluate_details(individual)
    general_fitness = sum(details)
    return (general_fitness,)


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

    bestFitness = []
    meanFitness = []

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
                ind.fitness.values[0] for ind in pop
            ]

            bestFitness.append(min(fitnessValues))
            meanFitness.append(
                sum(fitnessValues) / len(pop)
            )

            # Einzelne Abweichungen getrennt auswerten
            details = [evaluate_details(ind) for ind in pop]

            temp_values = [v[0] for v in details]
            ph_values = [v[1] for v in details]
            nut_values = [v[2] for v in details]
            salt_values = [v[3] for v in details]

            # Temperatur
            bestTempFitness.append(min(temp_values))
            meanTempFitness.append(
                sum(temp_values) / len(pop)
            )

            # pH-Wert
            bestPhFitness.append(min(ph_values))
            meanPhFitness.append(
                sum(ph_values) / len(pop)
            )

            # Nährstoffeffizienz
            bestNutFitness.append(min(nut_values))
            meanNutFitness.append(
                sum(nut_values) / len(pop)
            )

            # Salzkonzentration
            bestSaltFitness.append(min(salt_values))
            meanSaltFitness.append(
                sum(salt_values) / len(pop)
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

    # Gesamtfitness
    axes[0, 2].plot(bestFitness, label="Best Fitness")
    axes[0, 2].plot(meanFitness, label="Average Fitness")
    axes[0, 2].set_title("General Fitness")

    # Sechstes Diagramm wird nicht benötigt
    axes[1, 2].set_visible(False)

    for ax in axes.flat:
        if ax.get_visible():
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