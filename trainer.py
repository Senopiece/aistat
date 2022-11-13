#!/bin/python3
from multiprocessing import Pool
from functools import lru_cache
import random
import json
from typing import Callable
import mido
import math
from dataclasses import dataclass
import argparse
from copy import copy
import logging
import datetime
import traceback
import os
from VitalyMahonin import (
    AccompanimentFitnessChecker,
    Accompaniment,
    track_longing,
    Key,
    generate_random_population,
    progress,
    select,
    cross,
    mutate,
    compute_key_notes,
    filter_notes,
    detect_key,
    durations,
    Fitted,
)

logger = logging.getLogger("logger")

# Genetic stiff
FitnessCoefficients = tuple[float]


def generate_random_population_fc(
    population_size: int,
) -> set[FitnessCoefficients]:
    return set(
        tuple(random.uniform(0.0, 100.0) for _ in range(16))
        for _ in range(population_size)
    )


def select_fc(
    population: set[Fitted],
    fitness: Callable[[Fitted], float],
) -> set[Fitted]:
    return set(sorted(population, key=fitness, reverse=True)[: len(population) // 2])


def cross_fc(parents: tuple[FitnessCoefficients]) -> FitnessCoefficients:
    assert len(parents) > 1
    instance_len = len(parents[0])
    assert all(instance_len == len(parent) for parent in parents)
    return tuple(random.choice(parents)[i] for i in range(instance_len))


def mutate_fc(
    instance: FitnessCoefficients,
) -> FitnessCoefficients:
    return tuple(
        random.uniform(0.0, 100.0) if random.randint(0, 10) == 0 else e
        for e in instance
    )


def accompaniment_to_notes(accompaniment: Accompaniment) -> tuple[frozenset[int]]:
    return tuple(
        frozenset(note + e.octave*12 for note in e.notes) if e is not None else frozenset()
        for e in accompaniment
    )


def midi_track_to_notes(
    track: tuple[mido.Message],
    chord_duration: int,
) -> tuple[frozenset[int]]:
    notes_list = [[] for _ in range(math.ceil(track_longing(track) / chord_duration))]
    ticks = 0
    for msg in track:
        ticks += msg.time
        if isinstance(msg, mido.Message) and msg.type == "note_on":
            notes_list[ticks // chord_duration].append(msg.note)

    for i, s in enumerate(notes_list):
        notes_list[i] = frozenset(s)

    return tuple(notes_list)


@dataclass
class MelodyConfiguration:
    melody_key: Key
    key_notes: tuple[int]
    section_notes: tuple[frozenset[int]]
    pause_sensitivity: float

    def __init__(
        self,
        melody_key: Key,
        key_notes: tuple[int],
        section_notes: tuple[frozenset[int]],
        pause_sensitivity: float,
    ):
        self.melody_key = melody_key
        self.key_notes = key_notes
        self.section_notes = section_notes
        self.pause_sensitivity = pause_sensitivity

    def __hash__(self) -> int:
        return hash(
            (
                self.melody_key,
                self.key_notes,
                self.section_notes,
                self.pause_sensitivity,
            )
        )


@dataclass
class Solution:
    config: MelodyConfiguration  # melody context
    etalon: tuple[frozenset[int]]  # etalon accompaniment notes

    def __init__(
        self,
        config: MelodyConfiguration,
        etalon: tuple[frozenset[int]],
    ):
        self.config = config
        self.etalon = etalon

    def __hash__(self) -> int:
        return hash((self.config, self.etalon))


class AFCFitnessChecker:
    solutions: frozenset[Solution]

    def __init__(self, solutions: frozenset[Solution]):
        self.solutions = solutions

    @lru_cache(maxsize=120)
    def fitness(
        self,
        instance: FitnessCoefficients,
    ) -> float:
        score = 0

        # try to fit as much solutions
        for si, solution in enumerate(self.solutions):
            assert all(len(e) in {3, 0} for e in solution.etalon)

            fitness_checker = AccompanimentFitnessChecker(
                solution.config.melody_key,
                solution.config.key_notes,
                solution.config.section_notes,
                solution.config.pause_sensitivity,
                instance,
            )

            # inner evolution
            logger.info(
                f"{hash(instance)} evolute on melody {si+1}/{len(self.solutions)}"
            )
            population = generate_random_population(100, len(solution.config.key_notes))
            population = set(Fitted(e, fitness_checker.fitness(e)) for e in population)
            for i in range(1000):
                selected = select(population, lambda x: x.fitness)
                population = copy(selected)
                while len(population) < random.randint(90, 100):
                    parents = tuple(
                        random.choice(tuple(selected)).instance for _ in range(2)
                    )
                    individual = mutate(cross(parents))
                    population.add(
                        Fitted(individual, fitness_checker.fitness(individual))
                    )
            best = sorted(population, key=lambda x: x.fitness, reverse=True)[0].instance

            # transfer view
            best = accompaniment_to_notes(best)
            assert all(len(e) in {3, 0} for e in best)

            # the fitness function itself - compute etalon similarity
            score += sum(len(a & b) for a, b in zip(best, solution.etalon))

        logger.info(f"{hash(instance)}: fitness: {score}, instance: {instance}")
        return score


def f(x):
    e = x[0]
    fit = x[1]
    return (e, fit(e))


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "p",
        type=int,
        help="parallelization",
    )

    parser.add_argument(
        "-s",
        help="intialize from selected.json",
        action="store_true",
    )

    parser.add_argument("--e", type=str, help="working path", default="./")

    args = parser.parse_args()

    if not os.path.exists(args.e + "logs"):
        os.mkdir(args.e + "logs")
    file_handler = logging.FileHandler(
        f"{args.e}logs/trainer-{datetime.datetime.now()}.log"
    )
    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:  %(message)s", datefmt="%H:%M:%Ss"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    etalon_solutions = []
    for src, res in (
        ("input1.mid", "acc1.mid"),
        ("input2.mid", "acc2.mid"),
        ("input3.mid", "acc3.mid"),
    ):
        file = mido.MidiFile(args.e + res)
        track = file.tracks[1]  # accomponiment track

        sfile = mido.MidiFile(args.e + src)
        melody_track = tuple(filter_notes(sfile))
        chord_duration = (
            sfile.ticks_per_beat
        )  # NOTE: in assumption everything was taken at --cd = 1
        key_notes, section_notes = compute_key_notes(melody_track, chord_duration)

        # prepare things for evolution
        key = detect_key(durations(melody_track))

        etalon_solutions.append(
            Solution(
                MelodyConfiguration(
                    key,
                    key_notes,
                    section_notes,
                    50,  # NOTE: in assumption --ps = 50
                ),
                midi_track_to_notes(track, chord_duration),
            )
        )

    etalon_solutions = frozenset(etalon_solutions)

    checker = AFCFitnessChecker(etalon_solutions)

    print(checker.fitness((
            1,
            10,
            10,
            10,
            10,
            2,
            6,
            20,
            2,
            20,
            10,
            10,
            3,
            22,
            6,
            2,
        )))

    def load_selected():
        seledka = None
        with open(args.e + "selected.json", "r") as f:
            seledka = json.load(f)
        return set(map(lambda x: Fitted(tuple(x[0]), x[1]), seledka))

    def individ(selected):
        parents = tuple(random.choice(tuple(selected)).instance for _ in range(2))
        individual = mutate_fc(cross_fc(parents))
        return individual

    try:
        with Pool(args.p) as pool:
            population = None

            if args.s:
                selected = load_selected()
                population = copy(selected)
                tglen = random.randint(90, 100) - len(population)
                for e, fitness in progress(
                    pool.imap_unordered(
                        f,
                        map(
                            lambda _: (individ(selected), checker.fitness),
                            range(tglen),
                        ),
                    ),
                    total=tglen,
                    desc="Initializing from selected.json",
                ):
                    population.add(Fitted(e, fitness))
            else:
                population = generate_random_population_fc(100)
                tmp = set()
                for e, fitness in progress(
                    pool.imap_unordered(
                        f, map(lambda x: (x, checker.fitness), population)
                    ),
                    total=len(population),
                    desc="Initializing",
                ):
                    tmp.add(Fitted(e, fitness))
                population = tmp

            def dump_selected():  # emergency dump
                with open(args.e + "selected.json", "w") as f:
                    selected = select_fc(population, lambda x: x.fitness)
                    dumped = list([e.instance, e.fitness] for e in selected)
                    logger.info(f"Updated selected.json: {dumped}")
                    json.dump(dumped, f)

            def write():
                best = sorted(population, key=lambda x: x.fitness, reverse=True)[
                    0
                ].instance
                logger.info(f"Best: {best}")
                print(best)
                dump_selected()

            for i in progress(
                range(100),
                desc="Progress",
            ):
                logger.info(f">>> Generation {i+1}/100")
                selected = select_fc(population, lambda x: x.fitness)
                dump_selected()
                population = copy(selected)
                for e, fitness in pool.imap_unordered(
                    f,
                    map(
                        lambda _: (individ(selected), checker.fitness),
                        range(random.randint(90, 100) - len(population)),
                    ),
                ):
                    population.add(Fitted(e, fitness))
            write()
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e
