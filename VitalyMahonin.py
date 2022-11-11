#!/bin/python3
from functools import lru_cache
from copy import copy
from dataclasses import dataclass
import math
import random
import argparse
import signal
import sys
from typing import Callable, Iterator, Union, Any
import mido


try:
    from tqdm import tqdm

    def progress(arg, **kwargs):
        return tqdm(arg, **kwargs)

except ImportError:
    print("WARNING: no tqdm module found, no progress bar can be shown")
    print("         (run python3 -m pip install tqdm to solve this issue)")

    def progress(arg, **kwargs):
        return arg


## Helpers


@dataclass
class Fitted:
    instance: Any
    fitness: float

    def __init__(self, instance: Any, fitness: float):
        self.instance = instance
        self.fitness = fitness

    def __hash__(self) -> int:
        return hash((self.instance, self.fitness))


@dataclass
class Key:
    name: str
    tonic: int
    notes: frozenset[int]
    steps: tuple[
        int
    ]  # steps[note] = step_index or None if the note is not in Key (step_index starting from 0, not from 1 as in usual music theory)

    def __init__(self, name: str, tonic: int, scale: tuple[int]):
        """Create a key from a tonic and a scale.

        Args:
            name (str): the string representation of the key
            tonic (int): the MIDI note number of the tonic
            scale (tuple[int]): list of offsets from the tonic

            example of a possible scale:
                major scale: (0, 2, 4, 5, 7, 9, 11)
        """
        self.name = name
        self.tonic = tonic
        self.notes = frozenset((e + tonic) % 12 for e in scale)

        tmp = [None] * 12
        for i, e in enumerate(scale):
            tmp[(e + tonic) % 12] = i
        self.steps = tuple(tmp)

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash((self.name, self.tonic, self.notes, self.steps))


def durations(track: tuple[mido.Message]) -> tuple[float]:
    """Calculates the durations of the notes in the track

    Args:
        track (tuple[mido.Message]): the track to calculate the durations for

    Returns:
        tuple[float]: list of durations of the notes in the track
    """
    opened = [None] * 12
    res = [0] * 12
    time = 0
    for msg in track:
        time += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue
        note = msg.note % 12
        if msg.type == "note_on":
            opened[note] = time
        elif msg.type == "note_off":
            res[note] += time - opened[note]
            opened[note] = None
    return tuple(res)


def correlation(a: tuple[float], b: tuple[float]) -> float:
    assert len(a) == len(b)

    median_a = sum(a) / len(a)
    median_b = sum(b) / len(b)

    c = sum((x - median_a) * (y - median_b) for x, y in zip(a, b))
    d = math.sqrt(sum((x - median_a) ** 2 for x in a))
    e = math.sqrt(sum((y - median_b) ** 2 for y in b))

    return c / (d * e)


def cycle_offset(l: tuple, offset: int) -> tuple:
    """Returns a copy of the list with offset elements shifted to the beginning

    Args:
        l (tuple): source list
        offset (int): offset

    Returns:
        tuple: shifted list

    Examples:
        >>> cycle_offset((1, 2, 3, 4, 5), 2) = (3, 4, 5, 1, 2)
    """
    return l[offset % len(l) :] + l[: offset % len(l)]


def detect_key(note_durations: tuple[float]) -> Key:
    """Detects one of 24 keys.

    Args:
        note_durations (tuple[float]): list of durations of the notes in the track

    Returns:
        Key: the most appropriate key
    """
    # Krumhansl-Schmuckler key-finding algorithm:
    # https://rnhart.net/articles/key-finding/
    # with Simple profile from http://extras.humdrum.org/man/keycor/
    assert len(note_durations) == 12
    NOTENAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    SCALENAMES = ("", "m")
    SCALES = (
        (0, 2, 4, 5, 7, 9, 11),  # major
        (0, 2, 3, 5, 7, 8, 10),  # minor
    )
    PROFILES = (
        (2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1),  # major
        (2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0.5, 0.5),  # minor
    )

    best_score = -2
    best_scale = None
    best_tonic = None
    for scale in range(2):
        scale_profile = PROFILES[scale]
        for tonic in range(12):
            key_profile = cycle_offset(scale_profile, -tonic)
            score = correlation(key_profile, note_durations)
            if score > best_score:
                best_scale = scale
                best_tonic = tonic
                best_score = score

    return Key(
        f"{NOTENAMES[best_tonic]}{SCALENAMES[best_scale]}",
        best_tonic,
        SCALES[best_scale],
    )


def detect_octave(notes: Iterator[int]) -> Union[int, None]:
    """Detects the average octave of the notes

    Args:
        notes (Iterator[int]): List of used notes (midi numbers)

    Returns:
        Union[int, None]: the octave of the notes or None if the notes list in empty
    """
    s = 0
    n = 0
    for note in notes:
        s += note // 12
        n += 1
    return s // n if n != 0 else None


MINOR_CHORD = (0, 3, 7)
MAJOR_CHORD = (0, 4, 7)
DIM_CHORD = (0, 3, 6)
SUS2_CHORD = (0, 2, 7)
SUS4_CHORD = (0, 5, 7)


@dataclass
class Chord:
    octave: int
    root_note: int
    notes: frozenset[int]
    chord_type: tuple[int]

    def __init__(self, root_note: int, chord: tuple[int], octave: int):
        """Create a chord object

        Args:
            root_note (int): midi index of the root note (octave lost)
            chord (tuple[int]): triplets of offsets from the root note
            octave(int): octave of the chord
        """
        assert len(chord) == 3
        assert 0 <= root_note < 12
        self.root_note = root_note
        self.octave = octave
        self.chord_type = chord
        self.notes = frozenset((root_note + offset) % 12 for offset in chord)

    def __hash__(self) -> int:
        return hash((self.root_note, self.notes))


## Genetic stuff
# instances are tuple[Union[Chord, None]], where None means no chord occupying the section
Accompaniment = tuple[Union[Chord, None]]


def random_chord() -> Union[Chord, None]:
    return (
        None
        if random.randint(0, 1) == 0
        else Chord(
            random.randint(0, 11),
            random.choice(
                (
                    MINOR_CHORD,
                    MAJOR_CHORD,
                    DIM_CHORD,
                    SUS2_CHORD,
                    SUS4_CHORD,
                )
            ),
            random.randint(0, 8),
        )
    )


class AccompanimentFitnessChecker:
    melody_key: Key
    key_notes: tuple[int]  # may contain None's
    section_notes: tuple[set[int]]
    octaves: tuple[Union[int, None]]
    pause_sensitivity: float
    coefficients: tuple[float]

    def __init__(
        self,
        melody_key: Key,
        key_notes: tuple[int],
        section_notes: tuple[set[int]],
        pause_sensitivity: float,
        coefficients: tuple[float] = (
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
        ),
    ):
        """Creates the accompaniment fitness checker for a concrete melody

        Args:
            melody_key (Key): the key of the melody
            key_notes (tuple[int]): notes by each section (midi, lost octave) that need to be consonant with the accompaniment
            section_notes (tuple[set[int]]): all the notes in the section (midi number, octave present)
            pause_sensitivity (float): pause sensitivity
            coefficients (tuple[float]): list of coefficients for the fitness function
        """
        assert len(coefficients) == 16
        assert all(note is None or 0 <= note < 12 for note in key_notes)
        self.melody_key = melody_key
        self.key_notes = key_notes
        self.section_notes = tuple(
            frozenset(note % 12 for note in notes) for notes in section_notes
        )
        self.octaves = tuple(detect_octave(notes) for notes in section_notes)
        self.pause_sensitivity = pause_sensitivity
        self.coefficients = coefficients

    @lru_cache(maxsize=120)
    def fitness(self, accompaniment: Accompaniment) -> float:
        assert len(accompaniment) == len(self.key_notes)

        # GOOD_ACCESSORS[key.step_index(chord_root_note)] = set of good key indices for the next chords
        GOOD_ACCESSORS = (
            {0, 3, 4, 5},  # 0
            {1, 4},  # 1
            set(),  # 2
            {0, 3, 4},  # 3
            {0, 3, 4, 5},  # 4
            {1, 3, 5},  # 5
            set(),  # 6
        )

        prev_octave = None
        prev_key_step_index = None
        octave_s = 0
        score = 0

        for i, chord in enumerate(accompaniment):
            if chord is None:
                prev_key_step_index = None
                if self.key_notes[i] is None:
                    score += self.pause_sensitivity
                    if self.octaves[i] is None:
                        score += self.pause_sensitivity
            else:
                octave_s += chord.octave
                average_octave = octave_s / (i + 1)

                if chord.root_note in self.melody_key.notes:
                    score += self.coefficients[0]

                    # appropriate root_note rewards for major, minor and dim chords
                    forbidden = {2} if self.melody_key.name.endswith("m") else {7}
                    if self.melody_key.steps[
                        chord.root_note
                    ] not in forbidden and chord.chord_type in {
                        MAJOR_CHORD,
                        MINOR_CHORD,
                    }:
                        score += self.coefficients[1]
                    elif (
                        self.melody_key.steps[chord.root_note]
                        in forbidden  # witch was forbidden for major and minor is good for dim
                        and chord.chord_type == DIM_CHORD
                    ):
                        score += self.coefficients[2]

                    # appropriate root_note rewards for sus2 chord
                    forbidden = {2, 5} if self.melody_key.name.endswith("m") else {3, 7}
                    if (
                        self.melody_key.steps[chord.root_note] not in forbidden
                        and chord.chord_type == SUS2_CHORD
                    ):
                        score += self.coefficients[3]

                    # appropriate root_note rewards for sus4 chord
                    forbidden = {2, 6} if self.melody_key.name.endswith("m") else {4, 7}
                    if (
                        self.melody_key.steps[chord.root_note] not in forbidden
                        and chord.chord_type == SUS4_CHORD
                    ):
                        score += self.coefficients[4]

                    # check chord progression (self.melody_key.steps[chord.root_note] is guaranteed to be not None here because root_note in melody_key.notes)
                    if prev_key_step_index is not None and (
                        self.melody_key.steps[chord.root_note]
                        in GOOD_ACCESSORS[prev_key_step_index]
                    ):
                        score += self.coefficients[5]

                # note that for a chord that does not match the key with it's root self.melody_key.steps[chord.root_note] would return None
                prev_key_step_index = self.melody_key.steps[chord.root_note]

                # reward for key match
                score += self.coefficients[6] * len(self.melody_key.notes & chord.notes)

                # reward for more accompaniment chords are in the same octave
                score += (
                    (
                        (
                            self.coefficients[7]
                            - self.coefficients[8] * abs(prev_octave - chord.octave)
                            if abs(prev_octave - chord.octave) < 2
                            else 0
                        )
                        if self.octaves[i] is None
                        else (
                            self.coefficients[9] if prev_octave == chord.octave else 0
                        )
                    )
                    if prev_octave is not None
                    else 0
                )
                prev_octave = chord.octave

                # reward for chord hitting the major note of the melody on it's section
                if self.key_notes[i] in chord.notes:
                    score += self.coefficients[10]
                    if self.key_notes[i] == chord.root_note:
                        score += self.coefficients[11]

                # reward for more consonant sounds of chord with the melody
                score += self.coefficients[12] * len(
                    self.section_notes[i] & chord.notes
                )

                # target octave
                if self.octaves[i] is not None and chord.octave == self.octaves[i] - 1:
                    score += self.coefficients[13]

                # keep the average octave of the accomponent
                score += max(
                    0,
                    self.coefficients[14]
                    - self.coefficients[15] * abs(average_octave - chord.octave),
                )

        return score


def generate_random_population(
    population_size: int,
    instance_len: int,
) -> set[Accompaniment]:
    """Returns a random population of accompaniments

    Args:
        population_size (int): size of the resulting set
        instance_len (int): length of each instance of the population (number of chords in a accompaniment)

    Returns:
        set[Accompaniment]: the resulting set
    """
    res = set()
    for _ in range(population_size):
        res.add(
            tuple(random_chord() for _ in range(instance_len)),
        )
    return res


def select(
    population: set[Fitted],
    fitness: Callable[[Fitted], float],
) -> set[Fitted]:
    """Selects 50% of the best instances from the population according to the fitness function

    Args:
        population (set[Fitted]): parent population
        fitness (Callable[[Fitted], float]): fitness function ( higher is better )

    Returns:
        set[Fitted]: the set of selected instances
    """
    return set(sorted(population, key=fitness, reverse=True)[: len(population) // 2])


def cross(parents: tuple[Accompaniment]) -> Accompaniment:
    """Create a new instance crossing parents

    Args:
        parents (set[Accompaniment]): the parents

    Returns:
        Accompaniment: crossed instance
    """
    assert len(parents) > 1
    instance_len = len(parents[0])
    assert all(instance_len == len(parent) for parent in parents)
    return tuple(random.choice(parents)[i] for i in range(instance_len))


def mutate(
    instance: Accompaniment,
) -> Accompaniment:
    """Returns similar instance, but 10% of it's chords are modified

    Args:
        instance (Accompaniment): source instance

    Returns:
        Accompaniment: mutated instance
    """
    return tuple(
        random_chord() if random.randint(0, 10) == 0 else chord for chord in instance
    )


## Main


def track_longing(track: list[mido.Message]) -> int:
    """Track longing

    Args:
        track (list[mido.Message]): the track notes

    Returns:
        int: track longing in midi ticks
    """
    ticks = 0
    for msg in track:
        ticks += msg.time
    return ticks


def compute_key_notes(
    track: list[mido.Message], chord_duration: int
) -> list[tuple[int], tuple[frozenset[int]]]:
    """Divides the track into equal parts (call sections) by chord duration,
    and counts key and regular notes for each section

    Args:
        track (list[mido.Message]): the track notes
        chord_duration (int): duration of a chord in ticks

    Returns:
        list[tuple[int], tuple[frozenset[int]]]:
            [0] a tuple of key notes (midi number, lost octave)
            [1] a tuple of all notes in the section (midi number, octave present)
    """
    key_notes_list = [None] * (math.ceil(track_longing(track) / chord_duration))
    all_notes_list = [[] for _ in range(len(key_notes_list))]
    ticks = 0
    for msg in track:
        ticks += msg.time
        if msg.type == "note_on":
            if ticks % chord_duration == 0:
                key_notes_list[ticks // chord_duration] = msg.note % 12
            all_notes_list[ticks // chord_duration].append(msg.note)

    for i, e in enumerate(all_notes_list):
        all_notes_list[i] = frozenset(e)

    return tuple(key_notes_list), tuple(all_notes_list)


def filter_notes(file: mido.MidiFile) -> Iterator[mido.Message]:
    tempo = None
    for msg in file:
        if isinstance(msg, mido.MetaMessage) and msg.type == "set_tempo":
            tempo = msg.tempo
            continue

        if isinstance(msg, mido.Message):
            msg.time = int(
                mido.second2tick(msg.time, file.ticks_per_beat, tempo)
            )  # for some reason, the time is in seconds while iterating through raw file, so we need to convert it to ticks
            yield msg


def get_average_velocity(file: mido.MidiFile):
    sum_velocity = 0
    n = 0
    for msg in file:
        if isinstance(msg, mido.Message) and (msg.type == "note_on"):
            sum_velocity += msg.velocity
            n += 1
    return sum_velocity // n


def midi_file(string):
    try:
        res = mido.MidiFile(string)
        if res.type == 2:
            raise ValueError("type 2 is unsupported")
        return res
    except Exception as exc:
        raise argparse.ArgumentTypeError("Invalid MIDI file") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="accompaniment",
        description="Generate accompaniment for a melody",
    )

    parser.add_argument(
        "input",
        type=midi_file,
        help="the input MIDI file",
    )

    parser.add_argument(
        "--out",
        type=str,
        help='the output file name, default to "output"',
        default="output",
        metavar="filename",
    )

    parser.add_argument(
        "--iters",
        type=int,
        help="number of iterations, default to 1000",  # actually number of generations
        default=1000,
        metavar="int",
    )

    parser.add_argument(
        "--ps",
        type=float,
        help="pause sensitivity, default to 50",
        default=50,
        metavar="float",
    )

    parser.add_argument(
        "--cd",
        type=float,
        help="chord duration, default to 1",
        default=1,
        metavar="float",
    )

    parser.add_argument(
        "-k",
        help="add detected key to the output file name (like filename-C#m.mid)",
        action="store_true",
    )

    args = parser.parse_args()

    melody_track = tuple(filter_notes(args.input))

    # define chord duration and divide track into equal parts of chord durations
    chord_duration = int(args.input.ticks_per_beat * args.cd)
    key_notes, section_notes = compute_key_notes(melody_track, chord_duration)

    # prepare things for evolution
    key = detect_key(durations(melody_track))
    print("Detected key:", key)

    fitness_checker = AccompanimentFitnessChecker(
        key, key_notes, section_notes, args.ps
    )
    population = generate_random_population(100, len(key_notes))
    population = set(Fitted(e, fitness_checker.fitness(e)) for e in population)

    # prepare to write the best fit (and be able to write it even in the middle of the process when user presses ctrl+c)
    def write():
        best = sorted(population, key=lambda x: x.fitness, reverse=True)[0].instance
        velocity = int(get_average_velocity(args.input) * 0.7)

        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name="Accompaniment", time=0))
        start = 0

        for chord in best:
            if chord is None:
                start += chord_duration
            else:
                for i, note in enumerate(chord.notes):
                    track.append(
                        mido.Message(
                            "note_on",
                            note=note + chord.octave * 12,
                            velocity=velocity,
                            time=start if i == 0 else 0,
                        )
                    )
                for i, note in enumerate(chord.notes):
                    track.append(
                        mido.Message(
                            "note_off",
                            note=note + chord.octave * 12,
                            velocity=velocity,
                            time=chord_duration if i == 0 else 0,
                        )
                    )
                start = 0

        args.input.tracks.append(track)
        args.input.save(f"{args.out}{'-'+key.name if args.k else ''}.mid")

    def sigint_handler(*_):
        # write so far best found solution even not all iterations are done
        write()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)  # handle ctrl+c

    # run evolution on population (with iters number of generations)
    for _ in progress(
        range(args.iters),
        desc="Progress",
    ):
        selected = select(population, lambda x: x.fitness)
        population = copy(selected)
        while len(population) < random.randint(90, 100):
            parents = tuple(random.choice(tuple(selected)).instance for _ in range(2))
            individual = mutate(cross(parents))
            population.add(Fitted(individual, fitness_checker.fitness(individual)))

    # write result
    write()
