
import random
import time
import json
import math
import multiprocessing

import matplotlib.pyplot as plt
import seaborn as sns

from deap import base, creator, tools


# ============================================================
# KONFIGURATION
# ============================================================

NGEN = 1000

RANDOM_SEED = 42
chromosome_length = 400

initial_population_size = 100

space_COL = 1000
space_ROW = 1000

mutProb = 0.5
repr_mutProb = 0.5
indpb = 0.01

transProb = 0.01
max_trans_laenge = 20

repb = 0.2
death_rate = 0.1


# ============================================================
# DEAP-KLASSEN
# ============================================================

# Verhindert Fehler, wenn das Skript mehrfach
# in derselben Python-Umgebung ausgeführt wird.

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


# ============================================================
# FITNESSBERECHNUNG
# ============================================================

def evaluate_details(individual):
    temp_fitness = abs(((temp_env + 273) / 573 ) - (sum(individual[0:100]) / 100))
    ph_fitness = abs((ph_Env / 14) - (sum(individual[100:200]) / 100))
    nut_fitness = abs(optimal_nutrition_efficiency - (sum(individual[200:300]) / 100))
    salt_fitness = abs(salt_concentration - (sum(individual[300:400]) / 100))
    return temp_fitness, ph_fitness, nut_fitness, salt_fitness


def evaluate(individual, env):

    details = evaluate_details(individual, env)

    general_fitness = sum(details)

    return (general_fitness,)


# ============================================================
# TRANSDUKTION
# ============================================================

def transduction(recipient, donor):

    start = random.randint(0, len(recipient) - 1)

    segment_length = random.randint(
        1,
        max_trans_laenge
    )

    end = min(
        start + segment_length,
        len(recipient)
    )

    recipient[start:end] = donor[start:end]

    return recipient


# ============================================================
# NACHBARSCHAFTSSUCHE
# ============================================================

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


# ============================================================
# TODESSELEKTION
# ============================================================

def death(grid, deathRate):

    current_population = [
        (r, c, grid[r][c])
        for r in range(space_ROW)
        for c in range(space_COL)
        if grid[r][c] is not None
    ]

    dead_number = int(
        len(current_population) * deathRate
    )

    # Größere Fitnesswerte sind schlechter,
    # da alle Fitnesswerte minimiert werden.

    current_population.sort(
        key=lambda x: x[2].fitness.values[0],
        reverse=True
    )

    for r, c, bacterium in current_population[:dead_number]:
        grid[r][c] = None


# ============================================================
# STATISTIKEN
# ============================================================

def collect_stats(grid, env, stats):

    pop = [
        grid[r][c]
        for r in range(space_ROW)
        for c in range(space_COL)
        if grid[r][c] is not None
    ]

    if not pop:
        return

    # Gesamtfitness

    fitness_values = [
        ind.fitness.values[0]
        for ind in pop
    ]

    stats["bestFitness"].append(
        min(fitness_values)
    )

    stats["meanFitness"].append(
        sum(fitness_values) / len(pop)
    )

    # Einzelne Fitnesswerte

    details = [
        evaluate_details(ind, env)
        for ind in pop
    ]

    temp_values = [v[0] for v in details]
    ph_values = [v[1] for v in details]
    nut_values = [v[2] for v in details]
    salt_values = [v[3] for v in details]

    stats["bestTempFitness"].append(
        min(temp_values)
    )

    stats["meanTempFitness"].append(
        sum(temp_values) / len(pop)
    )

    stats["bestPhFitness"].append(
        min(ph_values)
    )

    stats["meanPhFitness"].append(
        sum(ph_values) / len(pop)
    )

    stats["bestNutFitness"].append(
        min(nut_values)
    )

    stats["meanNutFitness"].append(
        sum(nut_values) / len(pop)
    )

    stats["bestSaltFitness"].append(
        min(salt_values)
    )

    stats["meanSaltFitness"].append(
        sum(salt_values) / len(pop)
    )

    stats["popSize"].append(len(pop))


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # UMWELT AUSWÄHLEN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # STATISTIK-LISTEN
    # --------------------------------------------------------

    stats = {
        "bestTempFitness": [],
        "meanTempFitness": [],

        "bestPhFitness": [],
        "meanPhFitness": [],

        "bestNutFitness": [],
        "meanNutFitness": [],

        "bestSaltFitness": [],
        "meanSaltFitness": [],

        "bestFitness": [],
        "meanFitness": [],

        "popSize": []
    }

    # --------------------------------------------------------
    # DEAP TOOLBOX
    # --------------------------------------------------------

    toolbox = base.Toolbox()

    toolbox.register(
        "zeroOrOne",
        random.randint,
        0,
        1
    )

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.zeroOrOne,
        n=chromosome_length
    )

    toolbox.register(
        "mutate",
        tools.mutFlipBit,
        indpb=indpb
    )

    toolbox.register(
        "transduction",
        transduction
    )

    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=3
    )

    toolbox.register(
        "evaluate",
        evaluate,
        env=enviroment
    )

    # --------------------------------------------------------
    # STARTZEIT
    # --------------------------------------------------------

    start_time = time.time()

    # --------------------------------------------------------
    # MULTIPROCESSING
    # --------------------------------------------------------

    with multiprocessing.Pool(processes=20) as pool:

        # Fitnessberechnung auf mehrere Prozesse verteilen
        toolbox.register("map", pool.map)

        # ----------------------------------------------------
        # ANFANGSPOPULATION ERSTELLEN
        # ----------------------------------------------------

        pop = [
            toolbox.individual()
            for _ in range(initial_population_size)
        ]

        Grid = [
            [None for _ in range(space_COL)]
            for _ in range(space_ROW)
        ]

        # Zufällige, eindeutige Positionen auswählen.
        # Das spart Speicher gegenüber einer Liste
        # mit einer Million Koordinaten-Tupeln.

        positions = random.sample(
            range(space_ROW * space_COL),
            initial_population_size
        )

        # ----------------------------------------------------
        # ANFANGSPOPULATION PARALLEL BEWERTEN
        # ----------------------------------------------------

        fitness_values = toolbox.map(
            toolbox.evaluate,
            pop
        )

        for individual, fitness in zip(pop, fitness_values):
            individual.fitness.values = fitness

        # Bakterien im Grid platzieren

        for individual, position in zip(pop, positions):

            row, col = divmod(position, space_COL)

            Grid[row][col] = individual

        collect_stats(Grid, enviroment, stats)

        # ====================================================
        # EVOLUTIONSZYKLUS
        # ====================================================

        for g in range(NGEN):

            print(f"Generation: {g + 1}")

            # ------------------------------------------------
            # NEUES GRID FÜR SYNCHRONE AKTUALISIERUNG
            # ------------------------------------------------

            new_grid = [
                [None for _ in range(space_COL)]
                for _ in range(space_ROW)
            ]

            # Alle besetzten Positionen sammeln

            occupied_positions = [
                (r, c)
                for r in range(space_ROW)
                for c in range(space_COL)
                if Grid[r][c] is not None
            ]

            # Verarbeitungsreihenfolge zufällig mischen

            random.shuffle(occupied_positions)

            # ------------------------------------------------
            # BAKTERIEN DURCHLAUFEN
            # ------------------------------------------------

            for r, c in occupied_positions:

                bacterium = Grid[r][c]

                # Ursprüngliches Bakterium übernehmen

                new_grid[r][c] = bacterium

                # ============================================
                # ORIGINALMUTATION
                # ============================================

                if random.random() < mutProb:

                    mutant = toolbox.clone(
                        new_grid[r][c]
                    )

                    toolbox.mutate(mutant)

                    # Fitness ungültig machen
                    del mutant.fitness.values

                    new_grid[r][c] = mutant

                # ============================================
                # TRANSDUKTION
                # ============================================

                donor_candidates = get_neighbors(
                    Grid,
                    r,
                    c
                )

                if (
                    donor_candidates
                    and random.random() < transProb
                ):

                    donor = random.choice(
                        donor_candidates
                    )

                    # Auf dem möglicherweise bereits
                    # mutierten Bakterium aufbauen.

                    tran_recipient = toolbox.clone(
                        new_grid[r][c]
                    )

                    toolbox.transduction(
                        tran_recipient,
                        donor
                    )

                    del tran_recipient.fitness.values

                    new_grid[r][c] = tran_recipient

                # ============================================
                # FREIE NACHBARZELLEN SUCHEN
                # ============================================

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

                            empty_nei_cells.append(
                                (nr, nc)
                            )

                # Keine freien Nachbarzellen vorhanden

                if not empty_nei_cells:
                    continue

                # ============================================
                # REPRODUKTION
                # ============================================

                if random.random() < repb:

                    # Freie Nachbarzelle auswählen

                    nr, nc = random.choice(
                        empty_nei_cells
                    )

                    # Lokale Turnierselektion

                    candidate = (
                        [bacterium]
                        + get_neighbors(Grid, r, c)
                    )

                    sel_ind = toolbox.select(
                        candidate,
                        1
                    )[0]

                    # Nachkommen erzeugen

                    child = toolbox.clone(sel_ind)

                    # Reproduktionsmutation

                    if random.random() < repr_mutProb:

                        toolbox.mutate(child)

                        del child.fitness.values

                    # Kind im neuen Grid platzieren

                    new_grid[nr][nc] = child

            # =================================================
            # NEUES GRID ÜBERNEHMEN
            # =================================================

            Grid = new_grid

            # =================================================
            # UNGÜLTIGE FITNESSWERTE SAMMELN
            # =================================================

            invalid_individuals = [
                Grid[r][c]
                for r in range(space_ROW)
                for c in range(space_COL)
                if (
                    Grid[r][c] is not None
                    and not Grid[r][c].fitness.valid
                )
            ]

            # =================================================
            # FITNESS PARALLEL BERECHNEN
            # =================================================

            if invalid_individuals:

                fitness_values = toolbox.map(
                    toolbox.evaluate,
                    invalid_individuals
                )

                for individual, fitness in zip(
                    invalid_individuals,
                    fitness_values
                ):

                    individual.fitness.values = fitness

            # =================================================
            # TODESSELEKTION
            # =================================================

            death(Grid, death_rate)

            # =================================================
            # STATISTIKEN SAMMELN
            # =================================================

            collect_stats(
                Grid,
                enviroment,
                stats
            )

    # ========================================================
    # ENDE MULTIPROCESSING
    # ========================================================

    end_time = time.time()

    calculation_time = end_time - start_time

    print(
        f"Calculation time: {calculation_time:.4f} seconds"
    )

    # ========================================================
    # DIAGRAMME
    # ========================================================

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8)
    )

    # Temperatur

    axes[0, 0].plot(
        stats["bestTempFitness"],
        label="Best Fitness"
    )

    axes[0, 0].plot(
        stats["meanTempFitness"],
        label="Average Fitness"
    )

    axes[0, 0].set_title("Temperature Fitness")

    # pH-Wert

    axes[0, 1].plot(
        stats["bestPhFitness"],
        label="Best Fitness"
    )

    axes[0, 1].plot(
        stats["meanPhFitness"],
        label="Average Fitness"
    )

    axes[0, 1].set_title("pH Fitness")

    # Nährstoffe

    axes[1, 0].plot(
        stats["bestNutFitness"],
        label="Best Fitness"
    )

    axes[1, 0].plot(
        stats["meanNutFitness"],
        label="Average Fitness"
    )

    axes[1, 0].set_title("Nutrition Fitness")

    # Salz

    axes[1, 1].plot(
        stats["bestSaltFitness"],
        label="Best Fitness"
    )

    axes[1, 1].plot(
        stats["meanSaltFitness"],
        label="Average Fitness"
    )

    axes[1, 1].set_title("Salt Fitness")

    # Gesamtfitness

    axes[0, 2].plot(
        stats["bestFitness"],
        label="Best Fitness"
    )

    axes[0, 2].plot(
        stats["meanFitness"],
        label="Average Fitness"
    )

    axes[0, 2].set_title("General Fitness")

    # Populationsentwicklung

    axes[1, 2].plot(
        range(len(stats["popSize"])),
        stats["popSize"],
        label="Population Size"
    )

    axes[1, 2].set_title("Population Development")

    axes[1, 2].set_xlabel("Generation")
    axes[1, 2].set_ylabel("Number of Bacteria")

    axes[1, 2].legend()
    axes[1, 2].grid(True)

    # Achsenbeschriftungen

    for ax in [
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
        axes[0, 2]
    ]:

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")

        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# PROGRAMMSTART
# ============================================================

if __name__ == "__main__":
    main()
