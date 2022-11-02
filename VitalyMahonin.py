#!/bin/python3
from copy import deepcopy
from dataclasses import dataclass
import math
import random
import argparse
import signal
import sys
from typing import Callable, Iterator, Union
import mido

# TODO: output3 is broken

try:
    from tqdm import tqdm

    def progress(arg, **kwargs):
        return tqdm(arg, **kwargs)

except ImportError:
    print("WARNING: no tqdm module found, no progress bar can be showed")

    def progress(arg, **kwargs):
        return arg


## Helpers


@dataclass
class Key:
    name: str
    tonic: int
    notes: frozenset[int]
    steps: tuple[int]  # steps[note] = step_index or None if the note is not in Key

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
            tmp[(e + tonic) % 12] = i + 1
        self.steps = tuple(tmp)

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
    key_notes: tuple[int]  # may contain None's
    section_notes: tuple[set[int]]
    melody_octave: int
    pause_rate: float

    def __init__(
        self,
        melody_key: Key,
        key_notes: tuple[int],
        section_notes: tuple[set[int]],
        melody_octave: int,
        pause_rate: float,
    ):
        """Creates the accompaniment fitness checker for a concrete melody

        Args:
            melody_key (Key): the key of the melody
            key_notes (tuple[int]): notes by each section (midi, lost octave) that need to be consonant with the accompaniment
            section_notes (tuple[set[int]]): all the notes in the section (midi number, lost octave)
            melody_octave (int): midi octave of the melody
            pause_rate (float): whether you want more pause sensitivity
        """
        assert all(note is None or 0 <= note < 12 for note in key_notes)
        self.melody_key = melody_key
        self.key_notes = key_notes
        self.section_notes = section_notes
        self.pause_rate = pause_rate
        self.melody_octave = melody_octave

    def fitness(self, accompaniment: tuple[Chord]) -> float:
        assert len(accompaniment) == len(self.key_notes)

        octave = None
        score = 0

        for i, chord in enumerate(accompaniment):
            if chord is None and self.key_notes[i] is None:
                score += self.pause_rate
            elif chord is not None:
                if chord.notes < self.melody_key.notes:
                    score += 1

                    # appropriate root_note rewards for major, minor and dim chords
                    forbidden = {2} if self.melody_key.name.endswith("m") else {7}
                    if self.melody_key.steps[
                        chord.root_note
                    ] not in forbidden and chord.chord_type in {
                        MAJOR_CHORD,
                        MINOR_CHORD,
                    }:
                        score += 1
                    elif (
                        self.melody_key.steps[chord.root_note]
                        in forbidden  # witch was forbidden for major and minor is good for dim
                        and chord.chord_type == DIM_CHORD
                    ):
                        score += 1

                    # appropriate root_note rewards for sus2 chord
                    forbidden = {2, 5} if self.melody_key.name.endswith("m") else {3, 7}
                    if (
                        self.melody_key.steps[chord.root_note] not in forbidden
                        and chord.chord_type == SUS2_CHORD
                    ):
                        score += 1

                    # appropriate root_note rewards for sus4 chord
                    forbidden = {2, 6} if self.melody_key.name.endswith("m") else {4, 7}
                    if (
                        self.melody_key.steps[chord.root_note] not in forbidden
                        and chord.chord_type == SUS4_CHORD
                    ):
                        score += 1

                # reward for more accompaniment chords are in the same octave
                if octave is None:
                    octave = chord.octave
                if octave == chord.octave:
                    score += 2

                # reward for chord hitting the major note of the melody on it's section
                if chord.root_note == self.key_notes[i]:
                    score += 10

                # reward for more consonant sounds of chord with the melody
                score += 3 * len(self.section_notes[i] & chord.notes)

        # target octave
        if octave == self.melody_octave - 1:
            score += 20

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


def compute_key_notes(
    track: list[mido.Message], chord_duration: int
) -> list[tuple[int], tuple[set[int]]]:
    """Divides the track into equal parts (call sections) by chord duration,
    and counts key and regular notes for each section

    Args:
        track (list[mido.Message]): the track notes
        chord_duration (int): duration of a chord in ticks

    Returns:
        list[tuple[int], tuple[int]]:
            [0] a tuple of key notes (midi number, lost octave)
            [1] a tuple of all notes in the section (midi number, lost octave)
    """
    key_notes_list = [None] * (math.ceil(track_longing(track) / chord_duration))
    all_notes_list = [set() for _ in range(len(key_notes_list))]
    ticks = 0
    for msg in track:
        ticks += msg.time
        if msg.type == "note_on":
            if ticks % chord_duration == 0:
                key_notes_list[ticks // chord_duration] = msg.note % 12
            all_notes_list[ticks // chord_duration].add(msg.note % 12)
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
        "--iters",
        type=int,
        help="number of iterations, default to 1000",  # actually number of generations
        default=1000,
        metavar="int",
    )

    parser.add_argument(
        "--pr",
        type=float,
        help="pause rate, whether you want more pause sensitivity, default to 10",  # actually number of generations
        default=10,
        metavar="float",
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
    key_notes, section_notes = compute_key_notes(
        list(filter_notes(args.input)),
        chord_duration,
    )

    # run genetic to find a good accompaniment
    key = detect_key(melody_notes)
    fitness_checker = AccompanimentFitnessChecker(
        key, key_notes, section_notes, detect_octave(melody_notes), args.pr
    )

    population = generate_random_population(100, len(key_notes))

    def write():
        best = sorted(population, key=fitness_checker.fitness, reverse=True)[0]
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
        args.input.save(f"{args.out}{'-'+key.name if args.k else ''}.mid")

    def signal_handler(sig, frame):
        # write so far best found solution even not all iterations are done
        write()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)  # handle ctrl+c

    for _ in progress(
        range(args.iters),
        desc="Progress",
    ):
        selected = select(population, fitness_checker.fitness)
        population = deepcopy(selected)
        while len(population) < random.randint(90, 100):
            parents = tuple(random.choice(tuple(selected)) for _ in range(2))
            population.add(mutate(cross(parents)))

    # write result
    write()
