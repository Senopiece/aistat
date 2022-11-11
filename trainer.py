#!/bin/python3
from functools import lru_cache
import sys
import random
import signal
from typing import Callable, Any
import mido
import math
from dataclasses import dataclass
from copy import copy
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
)

# TODO: parallelization

# Genetic stiff
FitnessCoefficients = tuple[float]


def generate_random_population_fc(
    population_size: int,
) -> set[FitnessCoefficients]:
    return set(
        tuple(random.uniform(0.0, 60.0) for _ in range(16))
        for _ in range(population_size)
    )


def select_fc(
    population: set[FitnessCoefficients],
    fitness: Callable[[FitnessCoefficients], float],
) -> set[FitnessCoefficients]:
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
        random.uniform(0.0, 60.0) if random.randint(0, 10) == 0 else e for e in instance
    )


def accompaniment_to_notes(accompaniment: Accompaniment) -> tuple[frozenset[int]]:
    return tuple(
        frozenset(note + e.octave for note in e.notes) if e is not None else frozenset()
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
        for solution in self.solutions:
            assert all(len(e) in {3, 0} for e in solution.etalon)

            fitness_checker = AccompanimentFitnessChecker(
                solution.config.melody_key,
                solution.config.key_notes,
                solution.config.section_notes,
                solution.config.pause_sensitivity,
                instance,
            )

            # inner evolution
            population = generate_random_population(100, len(solution.config.key_notes))
            population = set(Fitted(e, fitness_checker.fitness(e)) for e in population)
            for _ in range(100):
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

        return score


@dataclass
class Fitted:
    instance: Any
    fitness: float

    def __init__(self, instance: Any, fitness: float):
        self.instance = instance
        self.fitness = fitness

    def __hash__(self) -> int:
        return hash((self.instance, self.fitness))


# Main

if __name__ == "__main__":
    etalon_solutions = []

    for src, res in (
        ("input1.mid", "acc1.mid"),
        ("input2.mid", "acc2.mid"),
        ("input3.mid", "acc3.mid"),
    ):
        file = mido.MidiFile(res)
        track = file.tracks[1]  # accomponiment track

        sfile = mido.MidiFile(src)
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

    # evolute
    population = generate_random_population_fc(100)
    population = set(Fitted(e, checker.fitness(e)) for e in population)

    def write():
        best = sorted(population, key=lambda x: x.fitness, reverse=True)[0].instance
        print(best)

    def sigint_handler(*_):
        # write so far best found solution even not all iterations are done
        write()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)  # handle ctrl+c

    try:
        for _ in progress(
            range(100),
            desc="Progress",
        ):
            selected = select_fc(population, lambda x: x.fitness)
            population = copy(selected)
            while len(population) < random.randint(90, 100):
                parents = tuple(
                    random.choice(tuple(selected)).instance for _ in range(2)
                )
                individual = mutate_fc(cross_fc(parents))
                population.add(Fitted(individual, checker.fitness(individual)))
    except Exception as e:
        write()
        raise e
    write()
