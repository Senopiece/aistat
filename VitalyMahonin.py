#!/bin/python3
from dataclasses import dataclass
import mido
import argparse


## Helpers


@dataclass
class Key:
    name: str
    tonic: int
    notes: set[int]

    def __init__(self, name: str, tonic: int, scale: tuple[int]):
        """Create a key from a tonic and a scale.

        Args:
            name (str): the string representation of the key
            tonic (int): the MIDI note number of the tonic
            scale (tuple[int]): list of offsets from the tonic

            example of a possible scale:
                major scale: (0, 2, 4, 5, 7, 9, 11)
        """
        assert len(scale) == 7
        self.name = name
        self.tonic = tonic
        self.notes = set([(e + tonic) % 12 for e in scale])

    def __str__(self) -> str:
        return self.name


def list_diff(list1: list, list2: list) -> int:
    diff = 0
    for elem1, elem2 in zip(list1, list2):
        diff += abs(elem1 - elem2)
    return diff


def detect_key(notes: list[int]) -> Key:
    """Detects one of 24 keys,
    but loses the octave information

    Args:
        notes (list[int]): List of used notes (midi numbers)

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

    # determine scale
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


def detect_octave(notes: list[int]) -> int:
    """Detects the average octave of the notes

    Args:
        notes (list[int]): List of used notes (midi numbers)

    Returns:
        int: the octave of the notes
    """
    s = 0
    for note in notes:
        s += note // 12
    return s // len(notes)


MINOR_CHORD = (0, 3, 7)
MAJOR_CHORD = (0, 4, 7)
DIM_CHORD = (0, 3, 6)


@dataclass
class Chord:
    root_note: int
    notes: set[int]

    def __init__(self, root_note: int, chord: tuple[int]):
        """Create a chord object

        Args:
            root_note (int): midi index of the root note
            chord (list[int]): triplets of offsets from the root note
        """
        assert len(chord) == 3
        self.root_note = root_note
        self.notes = set([(root_note + offset) % 12 for offset in chord])


## Main


def midi_file(string):
    try:
        return mido.MidiFile(string)
    except:
        raise argparse.ArgumentTypeError("Invalid MIDI file")


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
        help='the output file name, default to "accompaniment"',
        default="accompaniment",
        metavar="filename",  # TODO
    )

    parser.add_argument(
        "-k",
        help="add detected key to the output file name (like filename-C#m.mid)",
        action="store_true",  # TODO
    )

    args = parser.parse_args()
    args.input.type = 1

    melody_notes = [
        msg.note
        for msg in args.input
        if isinstance(msg, mido.Message) and msg.type == "note_on"
    ]

    print(detect_key(melody_notes))
    print(detect_octave(melody_notes))
