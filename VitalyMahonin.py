#!/bin/python3
from copy import deepcopy
from dataclasses import dataclass
import math
import random
import argparse
from typing import Callable, Iterator, Union
import mido


## Helpers


@dataclass
class Key:
    name: str
    tonic: int
    notes: frozenset[int]

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

    def __str__(self) -> str:
        return self.name


def list_diff(list1: list, list2: list) -> int:
    diff = 0
    for elem1, elem2 in zip(list1, list2):
        diff += abs(elem1 - elem2)
    return diff


def detect_key(notes: Iterator[int]) -> Key:
    """Detects one of 24 keys,
    but loses the octave information

    Args:
        notes (Iterator[int]): List of used notes (midi numbers)

    Returns:
        Key: the most appropriate key
    """
    _notes_names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    _scales = (
        (0, 2, 4, 5, 7, 9, 11),  # major
        (0, 2, 3, 5, 7, 8, 10),  # minor
    )
    _offsets_names = ("", "m")

    freq = [0] * 12  # freq[note] = number of occurrences of note
    for note in notes:
        freq[note % 12] += 1

    # select 7 most frequent notes
    most_freq = sorted(range(12), key=lambda x: freq[x], reverse=True)[:7]
    most_freq.sort()

    # determine scale by choosing the most appropriate match
    original_offset = [e - most_freq[0] for e in most_freq]
    scale_index = 0
    score = 10000  # minimizing the score
    for i, offset in enumerate(_scales):
        new_score = list_diff(original_offset, offset)
        if new_score < score:
            scale_index = i
            score = new_score

    # determine tonic
    tonic = most_freq[0]

    return Key(
        f"{_notes_names[tonic]}{_offsets_names[scale_index]}",
        tonic,
        _scales[scale_index],
    )


def detect_octave(notes: Iterator[int]) -> int:
    """Detects the average octave of the notes

    Args:
        notes (Iterator[int]): List of used notes (midi numbers)

    Returns:
        int: the octave of the notes
    """
    s = 0
    n = 0
    for note in notes:
        s += note // 12
        n += 1
    return s // n


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
        self.notes = frozenset((root_note + offset) % 12 for offset in chord)

    def __hash__(self) -> int:
        return hash((self.root_note, self.notes))


## Genetic stuff
# instances are tuple[Chord]


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
    key_notes: tuple[int]  # may contain None's TODO
    melody_octave: int

    def __init__(self, melody_key: Key, key_notes: tuple[int], melody_octave: int):
        """Creates the accompaniment fitness checker for a concrete melody

        Args:
            melody_key (Key): the key of the melody
            key_notes (tuple[int]): notes (midi, lost octave) that need to be consonant with the accompaniment
            melody_octave (int): midi octave of the melody
        """
        assert all(note is None or 0 <= note < 12 for note in key_notes)
        self.melody_key = melody_key
        self.key_notes = key_notes
        self.melody_octave = melody_octave

    def fitness(self, accompaniment: tuple[Chord]) -> float:
        assert len(accompaniment) == len(self.key_notes)
        octave = None
        score = 0
        for i, chord in enumerate(accompaniment):
            if chord is not None:
                if chord.notes < self.melody_key.notes:
                    score += 1
                if octave is None:
                    octave = chord.octave
                if chord.octave == self.melody_octave - 1:
                    score += 1
                if chord.octave == self.melody_octave - 2:
                    score += 0.7
                if chord.octave == self.melody_octave - 3:
                    score += 0.5
                if octave == chord.octave:
                    score += 2
                if chord.root_note == self.key_notes[i]:
                    score += 10
            elif self.key_notes[i] is None:
                score += 10
            # TODO: add circle of fifths or chord progression or the thing described before chord progression
            # TODO: not always skip chord if no simultaneous melody note, try to search another notes in this section and play with it (maybe except full break), also maybe the todo above may help to find candidates
            # TODO: increase score for a chord that have consonans with other melody notes in his section (so it's a possible implementation for the todo above if the cosonans score can overdo the disable score)
        return score


def generate_random_population(
    population_size: int,
    instance_len: int,
) -> set[tuple[Chord]]:
    """Returns a random population of accompaniments

    Args:
        population_size (int): size of the resulting set
        instance_len (int): length of each instance of the population

    Returns:
        set[tuple[Chord]]: the resulting set
    """
    res = set()
    for _ in range(population_size):
        res.add(
            tuple(random_chord() for _ in range(instance_len)),
        )
    return res


def select(
    population: set[tuple[Chord]],
    fitness: Callable[[tuple[Chord]], int],
) -> set[tuple[Chord]]:
    """Selects 50% of the best instances from the population according to the fitness function

    Args:
        population (set[tuple[Chord]]): parent population
        fitness (Callable[[tuple[Chord]], int]): fitness function ( higher is better )

    Returns:
        set[tuple[Chord]]: the set of selected instances
    """
    return set(sorted(population, key=fitness, reverse=True)[: len(population) // 2])


def cross(parents: tuple[tuple[Chord]]) -> tuple[Chord]:
    """Create a new instance crossing parents

    Args:
        parents (set[tuple[Chord]]): the parents

    Returns:
        tuple[Chord]: crossed instance
    """
    assert len(parents) > 1
    instance_len = len(parents[0])
    assert all(instance_len == len(parent) for parent in parents)
    return tuple(random.choice(parents)[i] for i in range(instance_len))


def mutate(
    instance: tuple[Chord],
) -> tuple[Chord]:
    """Returns similar instance, but 10% of it's chords are modified

    Args:
        instance (tuple[Chord]): source instance

    Returns:
        tuple[Chord]: mutated instance
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


def compute_key_notes(track: list[mido.Message], chord_duration: int) -> tuple[int]:
    """Returns notes that need to be consonant with the accompaniment

    Args:
        track (list[mido.Message]): the track notes
        chord_duration (int): duration of a chord in ticks

    Returns:
        tuple[int]: a tuple of midi notes (lost octave)
    """
    notes_list = [None] * (math.ceil(track_longing(track) / chord_duration))
    ticks = 0
    for msg in track:
        ticks += msg.time
        if ticks % chord_duration == 0 and msg.type == "note_on":
            notes_list[ticks // chord_duration] = msg.note % 12
    return tuple(notes_list)


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
        return mido.MidiFile(string)
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
        "-k",
        help="add detected key to the output file name (like filename-C#m.mid)",
        action="store_true",
    )

    args = parser.parse_args()
    assert args.input.type != 2

    melody_notes = tuple(
        msg.note
        for msg in args.input
        if isinstance(msg, mido.Message) and msg.type == "note_on"
    )

    # define chord duration and divide track into equal parts of chord durations
    chord_duration = args.input.ticks_per_beat * 2
    key_notes = compute_key_notes(
        list(filter_notes(args.input)),
        chord_duration,
    )

    # run genetic to find a good accompaniment
    key = detect_key(melody_notes)
    fitness_checker = AccompanimentFitnessChecker(
        key,
        key_notes,
        detect_octave(melody_notes),
    )

    population = generate_random_population(100, len(key_notes))
    for _ in range(1000):  # 1000 generations
        selected = select(population, fitness_checker.fitness)
        population = deepcopy(selected)
        while len(population) < random.randint(90, 100):
            parents = tuple(random.choice(tuple(selected)) for _ in range(2))
            population.add(mutate(cross(parents)))
    best = sorted(population, key=fitness_checker.fitness, reverse=True)[0]

    # write result (as a new track)
    velocity = int(get_average_velocity(args.input) * 0.9)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Accompaniment", time=0))
    start = 0
    for chord in best:
        if chord is None:
            start = chord_duration
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
    args.input.save(f"{args.out}{'-'+str(key) if args.k else ''}.mid")
