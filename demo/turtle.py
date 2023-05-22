# ~/~ begin <<docs/l-systems.md#demo/turtle.py>>[init]
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Generator, Callable, TypeVar, Union, Iterator, Generic, Type
# ~/~ begin <<docs/l-systems.md#turtle-imports>>[init]
from math import sin, cos, pi
from copy import copy
# ~/~ end

# ~/~ begin <<docs/l-systems.md#turtle-point>>[init]
@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y))
# ~/~ end
# ~/~ begin <<docs/l-systems.md#turtle-end-marker>>[init]
class EndMarker:
    pass
# ~/~ end
# ~/~ begin <<docs/l-systems.md#turtle-state>>[init]
@dataclass
class TurtleState:
    pos: Point = Point(0.0, 0.0)
    direction: Point = Point(1.0, 0.0)
    pen_down: bool = True
# ~/~ end
# ~/~ begin <<docs/l-systems.md#turtle-turtle>>[init]
@dataclass
class Turtle:
    stack: list[TurtleState] = field(default_factory=lambda: [TurtleState()])

    @property
    def current(self):
        return self.stack[-1]

    def push(self):
        self.stack.append(copy(self.current))

    def pop(self):
        self.stack.pop()
# ~/~ end

# ~/~ begin <<docs/l-systems.md#turtle-composable>>[init]
T = TypeVar("T")
Output = Union[Point, Type[EndMarker]]
Command = Callable[[T], Generator[Output, None, T]]
# ~/~ end
# ~/~ begin <<docs/l-systems.md#turtle-composable>>[1]
class composable(Generic[T]):
    def __init__(self, f: Command[T]):
        self.f = f

    def __call__(self, obj: T) -> Generator[Output, None, T]:
        return (yield from self.f(obj))

    def __rshift__(self, other: composable[T]) -> composable[T]:
        def composed(obj: T) -> Generator[Output, None, T]:
            obj = yield from self.f(obj)
            obj = yield from other.f(obj)
            return obj

        return composable(composed)


def collect(t: Turtle, cmds: Iterable[Command]) -> Iterator[Output]:
    yield t.current.pos
    for c in cmds:
        t = yield from c(t)
    return t
# ~/~ end
# ~/~ begin <<docs/l-systems.md#turtle-commands>>[init]
def turn(angle: float) -> composable:
    u = cos(angle * pi / 180.0)
    v = sin(angle * pi / 180.0)

    @composable
    def _turn(t: Turtle) -> Generator[Output, None, Turtle]:
        dx, dy = t.current.direction
        t.current.direction = Point(u * dx - v * dy, v * dx + u * dy)
        yield from ()
        return t

    return _turn


@composable
def pen_down(t: Turtle) -> Generator[Output, None, Turtle]:
    t.current.pen_down = True
    yield t.current.pos
    return t


@composable
def pen_up(t: Turtle) -> Generator[Output, None, Turtle]:
    t.current.pen_down = False
    yield EndMarker
    return t


def walk(dist: float) -> composable:
    @composable
    def _walk(t: Turtle) -> Generator[Output, None, Turtle]:
        x, y = t.current.pos
        dx, dy = t.current.direction
        t.current.pos = Point(x + dist * dx, y + dist * dy)
        if t.current.pen_down:
            yield t.current.pos
        return t

    return _walk


@composable
def save(t: Turtle) -> Generator[Output, None, Turtle]:
    t.push()
    yield from ()
    return t


@composable
def restore(t: Turtle) -> Generator[Output, None, Turtle]:
    t.pop()
    yield from ()
    return t


@composable
def identity(t: Turtle) -> Generator[Output, None, Turtle]:
    yield from ()
    return t
# ~/~ end
# ~/~ end