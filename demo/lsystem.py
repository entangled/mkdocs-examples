# ~/~ begin <<docs/l-systems.md#demo/lsystem.py>>[init]
from __future__ import annotations
from typing import Optional, Iterable, Iterator
from dataclasses import dataclass
from .turtle import \
    Turtle, Point, EndMarker, Command, walk, turn, collect, \
    pen_down, pen_up, save, restore, identity


@dataclass
class LSystem:
    axiom: str
    rules: dict[str, str]
    commands: dict[str, Command[Turtle]]

    # ~/~ begin <<docs/l-systems.md#lsystem-methods>>[init]
    def expand(self, gen: int, inp: Optional[str] = None) -> Iterator[str]:
        inp = inp or self.axiom

        if gen == 0:
            yield from inp
            return

        for c in inp:
            yield from self.expand(gen - 1, self.rules.get(c, c))
    # ~/~ end
    # ~/~ begin <<docs/l-systems.md#lsystem-methods>>[1]
    def run(self, gen: int) -> Iterable[Command[Turtle]]:
        return map(lambda x: self.commands.get(x, identity), self.expand(gen))
    # ~/~ end
    # ~/~ begin <<docs/l-systems.md#lsystem-methods>>[2]
    def to_gnuplot(self, gen: int):
        for p in collect(Turtle(), self.run(gen)):
            match p:
                case Point(x, y):
                    print(x, y)
                case EndMarker:
                    print()
    # ~/~ end


# ~/~ begin <<docs/l-systems.md#l-systems>>[init]
sierspinsky = LSystem(
    axiom = "F-G-G",
    rules = {
        "F": "F-G+F+G-F",
        "G": "GG"
    },
    # ~/~ begin <<docs/l-systems.md#sierspinsky-commands>>[init]
    commands = {
        "F": walk(1),
        "G": walk(1),
        "+": turn(-120),
        "-": turn(120)
    }
    # ~/~ end
)
# ~/~ end
# ~/~ begin <<docs/l-systems.md#l-systems>>[1]
dragon = LSystem(
    "F",
    {"F": "F+G", "G": "F-G"},
    {
        "F": walk(0.7),
        "+": turn(45) >> walk(0.35) >> turn(45),
        "-": turn(-45) >> walk(0.35) >> turn(-45),
        "G": walk(0.7),
    },
)
# ~/~ end
# ~/~ begin <<docs/l-systems.md#l-systems>>[2]
barnsley_fern = LSystem(
    "X",
    {
        "X": "F+[[X]-X]-F[-FX]+X",
        "F": "FF"
    },
    {
        "F": walk(1.0),
        "+": turn(25),
        "-": turn(-25),
        "[": save,
        "]": pen_up >> restore >> pen_down
    }
)
# ~/~ end
# ~/~ begin <<docs/l-systems.md#l-systems>>[3]
def koch(angle):
    return LSystem(
        axiom = "F",
        rules = {
            "F": "F+F--F+F"
        },
        commands = {
            "F": walk(1),
            "+": turn(angle),
            "-": turn(-angle)
        })
# ~/~ end
# ~/~ end