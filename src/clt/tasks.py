from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import argparse
import ast
import json
import operator
import random
from pathlib import Path
from typing import Callable, Iterable


TaskName = str


@dataclass(frozen=True)
class Example:
    prompt: str
    trace: str
    answer: str
    metadata: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def list_tasks() -> list[str]:
    return [
        "graph_reachability",
        "pointer_chasing",
        "shortest_path",
        "maze_planning",
        "symbolic_arithmetic",
    ]


def generate_example(task: TaskName, seed: int, split: str = "train", difficulty: str = "standard") -> Example:
    rng = random.Random(seed)
    ood = split == "ood_test"
    if task == "graph_reachability":
        if difficulty == "easy":
            return _generate_easy_graph_reachability(rng, seed, split, ood)
        if difficulty == "easy_ladder":
            return _generate_easy_ladder_graph_reachability(rng, seed, split, ood)
        return _generate_graph_reachability(rng, seed, split, ood)
    if task == "pointer_chasing":
        return _generate_pointer_chasing(rng, seed, split, ood, simple=difficulty == "simple")
    if task == "shortest_path":
        return _generate_shortest_path(rng, seed, split, ood)
    if task == "maze_planning":
        return _generate_maze_planning(rng, seed, split, ood)
    if task == "symbolic_arithmetic":
        return _generate_symbolic_arithmetic(rng, seed, split, ood)
    raise ValueError(f"Unknown task: {task}")


def _validate_difficulty(task: TaskName, difficulty: str) -> None:
    if difficulty == "standard":
        return
    if task == "graph_reachability" and difficulty in {"easy", "easy_ladder"}:
        return
    if task == "pointer_chasing" and difficulty == "simple":
        return
    raise ValueError(f"Unsupported difficulty={difficulty!r} for task={task!r}")


def verify_answer(example: Example, candidate: str) -> bool:
    expected = _normalize_answer(example.answer)
    actual = _normalize_answer(candidate)
    return expected == actual


def write_jsonl(examples: Iterable[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(example.to_json())
            f.write("\n")


def _normalize_answer(text: str) -> str:
    text = text.strip()
    if text.lower().startswith("answer:"):
        text = text.split(":", 1)[1].strip()
    return " ".join(text.split()).upper()


def _node_name(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"N{index}"


def _format_nodes(n: int) -> str:
    return ", ".join(_node_name(i) for i in range(n))


def _format_edges(edges: set[tuple[int, int]]) -> str:
    return ", ".join(f"{_node_name(a)}->{_node_name(b)}" for a, b in sorted(edges))


def _bfs_path(n: int, edges: set[tuple[int, int]], source: int, target: int) -> tuple[list[int] | None, list[int]]:
    graph = [[] for _ in range(n)]
    for a, b in sorted(edges):
        graph[a].append(b)

    parent = {source: None}
    visit_order = [source]
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for neighbor in graph[node]:
            if neighbor in parent:
                continue
            parent[neighbor] = node
            visit_order.append(neighbor)
            queue.append(neighbor)

    if target not in parent:
        return None, visit_order

    path = []
    node: int | None = target
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path)), visit_order


def _bfs_distance(n: int, edges: set[tuple[int, int]], source: int, target: int) -> tuple[int | None, dict[int, int]]:
    graph = [[] for _ in range(n)]
    for a, b in sorted(edges):
        graph[a].append(b)

    dist = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor in dist:
                continue
            dist[neighbor] = dist[node] + 1
            queue.append(neighbor)
    return dist.get(target), dist


def _generate_graph_reachability(rng: random.Random, seed: int, split: str, ood: bool) -> Example:
    n = rng.randint(12, 18) if ood else rng.randint(6, 10)
    source, target = 0, n - 1
    reachable = rng.random() < 0.5
    max_path = min(n - 1, 8 if ood else 4)
    min_path = min(5 if ood else 1, max_path)
    desired_len = rng.randint(min_path, max_path)

    if reachable:
        middle = rng.sample(range(1, n - 1), max(0, desired_len - 1))
        path = [source, *middle, target]
        edges = set(zip(path, path[1:]))
        attempts = n * 3 if ood else n * 2
        for _ in range(attempts):
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                edges.add((a, b))
        solved_path, _ = _bfs_path(n, edges, source, target)
        answer = "YES" if solved_path else "NO"
    else:
        reachable_side_size = rng.randint(1, n - 2)
        reachable_side = {source, *rng.sample(range(1, n - 1), reachable_side_size - 1)}
        blocked_side = set(range(n)) - reachable_side
        edges = set()
        attempts = n * 4 if ood else n * 3
        for _ in range(attempts):
            pool = reachable_side if rng.random() < 0.65 else blocked_side
            if len(pool) < 2:
                continue
            a, b = rng.sample(sorted(pool), 2)
            edges.add((a, b))
        solved_path, _ = _bfs_path(n, edges, source, target)
        answer = "YES" if solved_path else "NO"

    solved_path, visit_order = _bfs_path(n, edges, source, target)
    answer = "YES" if solved_path else "NO"
    trace = _graph_trace(source, target, solved_path, visit_order)
    prompt = (
        "You are given a directed graph.\n"
        f"Nodes: {_format_nodes(n)}\n"
        f"Edges: {_format_edges(edges)}\n"
        f"Question: Is there a path from {_node_name(source)} to {_node_name(target)}?\n"
        "Return YES or NO."
    )
    return Example(
        prompt=prompt,
        trace=trace,
        answer=answer,
        metadata={
            "task": "graph_reachability",
            "difficulty": "standard",
            "split": split,
            "seed": seed,
            "num_nodes": n,
            "num_edges": len(edges),
            "reachable": answer == "YES",
            "path_length": None if solved_path is None else len(solved_path) - 1,
        },
    )


def _generate_easy_graph_reachability(rng: random.Random, seed: int, split: str, ood: bool) -> Example:
    n = 6 if ood else 4
    return _generate_easy_graph_reachability_fixed_n(rng, seed, split, n, distractor_edges=2 if ood else 1)


def _generate_easy_ladder_graph_reachability(rng: random.Random, seed: int, split: str, ood: bool) -> Example:
    node_choices = [7, 8] if ood else [4, 5, 6]
    n = node_choices[seed % len(node_choices)]
    reachable = (seed // len(node_choices)) % 2 == 0
    return _generate_easy_graph_reachability_fixed_n(
        rng,
        seed,
        split,
        n,
        distractor_edges=2,
        reachable=reachable,
        difficulty="easy_ladder",
    )


def generate_easy_graph_reachability_fixed_nodes(
    seed: int,
    n: int,
    split: str = "diagnostic",
    reachable: bool | None = None,
    difficulty: str = "easy",
) -> Example:
    if n < 4:
        raise ValueError("easy graph reachability diagnostics require n >= 4")
    rng = random.Random(seed)
    return _generate_easy_graph_reachability_fixed_n(
        rng,
        seed,
        split,
        n,
        distractor_edges=2,
        reachable=reachable,
        difficulty=difficulty,
    )


def _generate_easy_graph_reachability_fixed_n(
    rng: random.Random,
    seed: int,
    split: str,
    n: int,
    distractor_edges: int,
    reachable: bool | None = None,
    difficulty: str = "easy",
) -> Example:
    source, target = 0, n - 1
    reachable = rng.random() < 0.5 if reachable is None else reachable
    edges: set[tuple[int, int]] = set()

    if reachable:
        if rng.random() < 0.5 or n == 4:
            edges.add((source, target))
        else:
            middle = rng.randrange(1, n - 1)
            edges.add((source, middle))
            edges.add((middle, target))
    else:
        # Keep source disconnected from target so the label is learnable for a small model.
        for node in range(1, n - 1):
            if rng.random() < 0.6:
                edges.add((node, target))
        if rng.random() < 0.5:
            edges.add((1, 2 if n > 4 else target))
        edges = {edge for edge in edges if edge[0] != source}

    # Add a small number of distractor edges that do not change reachability.
    for _ in range(distractor_edges):
        a = rng.randrange(1, n)
        b = rng.randrange(1, n)
        if a != b:
            edges.add((a, b))

    solved_path, visit_order = _bfs_path(n, edges, source, target)
    answer = "YES" if solved_path else "NO"
    trace = _graph_trace(source, target, solved_path, visit_order)
    prompt = (
        "You are given a directed graph.\n"
        f"Nodes: {_format_nodes(n)}\n"
        f"Edges: {_format_edges(edges)}\n"
        f"Question: Is there a path from {_node_name(source)} to {_node_name(target)}?\n"
        "Return YES or NO."
    )
    return Example(
        prompt=prompt,
        trace=trace,
        answer=answer,
        metadata={
            "task": "graph_reachability",
            "difficulty": difficulty,
            "split": split,
            "seed": seed,
            "num_nodes": n,
            "num_edges": len(edges),
            "reachable": answer == "YES",
            "path_length": None if solved_path is None else len(solved_path) - 1,
        },
    )


def _graph_trace(source: int, target: int, path: list[int] | None, visit_order: list[int]) -> str:
    if path:
        hops = ". ".join(
            f"Visit {_node_name(path[i + 1])} from {_node_name(path[i])}" for i in range(len(path) - 1)
        )
        return f"Start at {_node_name(source)}. {hops}. {_node_name(target)} is reached."
    visited = ", ".join(_node_name(node) for node in visit_order)
    return f"Start at {_node_name(source)}. Visited nodes: {visited}. {_node_name(target)} is not reached."


def _generate_pointer_chasing(rng: random.Random, seed: int, split: str, ood: bool, simple: bool = False) -> Example:
    if simple:
        depth_choices = [2, 3] if ood else [1]
        n = 4
    else:
        depth_choices = [5, 6, 7, 8] if ood else [2, 3, 4]
        n = 12
    depth = depth_choices[seed % len(depth_choices)]
    answer_is_yes = (seed // len(depth_choices)) % 2 == 0

    start = rng.randrange(n)
    path = [start]
    unused = [node for node in range(n) if node != start]
    for _ in range(depth):
        next_node = rng.choice(unused)
        unused.remove(next_node)
        path.append(next_node)

    transitions: dict[int, int] = {}
    for src, dst in zip(path, path[1:]):
        transitions[src] = dst
    for node in range(n):
        if node not in transitions:
            transitions[node] = rng.randrange(n)

    final_state = path[-1]
    if answer_is_yes:
        target = final_state
    else:
        candidates = [node for node in range(n) if node != final_state]
        target = rng.choice(candidates)

    rules = ", ".join(f"{_node_name(src)}->{_node_name(transitions[src])}" for src in range(n))
    answer = "YES" if target == final_state else "NO"
    trace_steps = ". ".join(
        f"Step {i}: {_node_name(path[i - 1])}->{_node_name(path[i])}" for i in range(1, len(path))
    )
    relation = "equals" if answer == "YES" else "does not equal"
    trace = (
        f"Start at {_node_name(start)}. {trace_steps}. "
        f"After {depth} steps the state is {_node_name(final_state)}, which {relation} target {_node_name(target)}."
    )
    if simple:
        prompt = (
            f"Rules: {rules}\n"
            f"Start: {_node_name(start)}\n"
            f"Next: {_node_name(path[1])}\n"
            f"Steps: {depth}\n"
            f"Target: {_node_name(target)}\n"
            f"Compare: {_node_name(path[1])}={_node_name(target)}\n"
            "Return YES or NO."
        )
    else:
        prompt = (
            "You are given deterministic state transition rules.\n"
            f"States: {_format_nodes(n)}\n"
            f"Rules: {rules}\n"
            f"Start: {_node_name(start)}\n"
            f"Steps: {depth}\n"
            f"Question: after exactly {depth} transitions, are you at {_node_name(target)}?\n"
            "Return YES or NO."
        )
    return Example(
        prompt=prompt,
        trace=trace,
        answer=answer,
        metadata={
            "task": "pointer_chasing",
            "difficulty": "simple" if simple else "standard",
            "split": split,
            "seed": seed,
            "num_states": n,
            "depth": depth,
            "start": _node_name(start),
            "target": _node_name(target),
            "final_state": _node_name(final_state),
            "reachable_target": answer == "YES",
        },
    )


def _generate_shortest_path(rng: random.Random, seed: int, split: str, ood: bool) -> Example:
    n = rng.randint(12, 18) if ood else rng.randint(6, 10)
    source, target = 0, n - 1
    reachable = rng.random() < 0.8
    min_dist = 6 if ood else 2
    max_dist = min(n - 1, 10 if ood else 5)
    if min_dist > max_dist:
        min_dist = max_dist
    desired_dist = rng.randint(min_dist, max_dist)

    if reachable:
        edges = _sample_graph_with_exact_distance(rng, n, source, target, desired_dist, ood)
        distance, dist_map = _bfs_distance(n, edges, source, target)
        answer = str(distance)
    else:
        edges = _sample_unreachable_graph(rng, n, source, target, ood)
        distance, dist_map = _bfs_distance(n, edges, source, target)
        answer = "INF" if distance is None else str(distance)

    trace = _distance_trace(source, target, distance, dist_map)
    prompt = (
        "You are given an unweighted directed graph.\n"
        f"Nodes: {_format_nodes(n)}\n"
        f"Edges: {_format_edges(edges)}\n"
        f"Question: What is the shortest path distance from {_node_name(source)} to {_node_name(target)}?\n"
        "Return an integer, or INF if unreachable."
    )
    return Example(
        prompt=prompt,
        trace=trace,
        answer=answer,
        metadata={
            "task": "shortest_path",
            "split": split,
            "seed": seed,
            "num_nodes": n,
            "num_edges": len(edges),
            "distance": distance,
        },
    )


def _sample_graph_with_exact_distance(
    rng: random.Random, n: int, source: int, target: int, desired_dist: int, ood: bool
) -> set[tuple[int, int]]:
    for _ in range(500):
        middle = rng.sample(range(1, n - 1), desired_dist - 1)
        path = [source, *middle, target]
        edges = set(zip(path, path[1:]))
        attempts = n * 4 if ood else n * 3
        for _ in range(attempts):
            a, b = rng.randrange(n), rng.randrange(n)
            if a == b:
                continue
            candidate = set(edges)
            candidate.add((a, b))
            distance, _ = _bfs_distance(n, candidate, source, target)
            if distance == desired_dist:
                edges = candidate
        distance, _ = _bfs_distance(n, edges, source, target)
        if distance == desired_dist:
            return edges
    raise RuntimeError("Failed to generate graph with exact shortest path distance")


def _sample_unreachable_graph(rng: random.Random, n: int, source: int, target: int, ood: bool) -> set[tuple[int, int]]:
    reachable_side_size = rng.randint(1, n - 2)
    reachable_side = {source, *rng.sample(range(1, n - 1), reachable_side_size - 1)}
    blocked_side = set(range(n)) - reachable_side
    edges = set()
    attempts = n * 4 if ood else n * 3
    for _ in range(attempts):
        pool = reachable_side if rng.random() < 0.65 else blocked_side
        if len(pool) < 2:
            continue
        a, b = rng.sample(sorted(pool), 2)
        edges.add((a, b))
    return edges


def _distance_trace(source: int, target: int, distance: int | None, dist_map: dict[int, int]) -> str:
    items = ", ".join(f"{_node_name(node)}={dist}" for node, dist in sorted(dist_map.items(), key=lambda x: x[1]))
    if distance is None:
        return f"Run BFS from {_node_name(source)}. Distances found: {items}. {_node_name(target)} is unreachable."
    return (
        f"Run BFS from {_node_name(source)}. Distances found: {items}. "
        f"The shortest distance to {_node_name(target)} is {distance}."
    )


def _generate_maze_planning(rng: random.Random, seed: int, split: str, ood: bool) -> Example:
    if ood:
        height = rng.randint(10, 14)
        width = rng.randint(10, 14)
        min_len, max_len = 14, 28
        wall_density = rng.uniform(0.20, 0.35)
    else:
        height = rng.randint(5, 8)
        width = rng.randint(5, 8)
        min_len, max_len = 4, 12
        wall_density = rng.uniform(0.15, 0.30)

    solvable = rng.random() < 0.8
    grid, distance = _sample_maze(rng, height, width, wall_density, min_len, max_len, solvable)
    answer = str(distance) if distance is not None else "INF"
    trace = (
        f"Run BFS from S. G is reached at distance {distance}."
        if distance is not None
        else "Run BFS from S. The search exhausts reachable cells without reaching G."
    )
    prompt = (
        "Find the shortest path from S to G in the grid.\n"
        + "\n".join("".join(row) for row in grid)
        + "\nReturn the shortest path length, or INF if no path exists."
    )
    return Example(
        prompt=prompt,
        trace=trace,
        answer=answer,
        metadata={
            "task": "maze_planning",
            "split": split,
            "seed": seed,
            "height": height,
            "width": width,
            "distance": distance,
            "solvable": distance is not None,
        },
    )


def _sample_maze(
    rng: random.Random,
    height: int,
    width: int,
    wall_density: float,
    min_len: int,
    max_len: int,
    solvable: bool,
) -> tuple[list[list[str]], int | None]:
    for _ in range(1000):
        grid = [["#" if rng.random() < wall_density else "." for _ in range(width)] for _ in range(height)]
        grid[0][0] = "S"
        grid[height - 1][width - 1] = "G"
        distance = _maze_distance(grid)
        if solvable and distance is not None and min_len <= distance <= max_len:
            return grid, distance
        if not solvable and distance is None:
            return grid, None

    grid = [["." for _ in range(width)] for _ in range(height)]
    grid[0][0] = "S"
    grid[height - 1][width - 1] = "G"
    if not solvable:
        if height > 1:
            grid[height - 2][width - 1] = "#"
        if width > 1:
            grid[height - 1][width - 2] = "#"
        return grid, None
    return grid, _maze_distance(grid)


def _maze_distance(grid: list[list[str]]) -> int | None:
    height, width = len(grid), len(grid[0])
    queue = deque([(0, 0)])
    dist = {(0, 0): 0}
    while queue:
        row, col = queue.popleft()
        if (row, col) == (height - 1, width - 1):
            return dist[(row, col)]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if grid[nr][nc] == "#":
                continue
            if (nr, nc) in dist:
                continue
            dist[(nr, nc)] = dist[(row, col)] + 1
            queue.append((nr, nc))
    return None


@dataclass(frozen=True)
class Expr:
    value: int | None = None
    op: str | None = None
    left: "Expr | None" = None
    right: "Expr | None" = None

    def render(self) -> str:
        if self.value is not None:
            return str(self.value)
        assert self.left is not None and self.right is not None and self.op is not None
        return f"({self.left.render()} {self.op} {self.right.render()})"

    def evaluate(self) -> tuple[int, list[str]]:
        if self.value is not None:
            return self.value, []
        assert self.left is not None and self.right is not None and self.op is not None
        left_value, left_steps = self.left.evaluate()
        right_value, right_steps = self.right.evaluate()
        op_fn: dict[str, Callable[[int, int], int]] = {"+": operator.add, "-": operator.sub}
        result = op_fn[self.op](left_value, right_value)
        return result, [*left_steps, *right_steps, f"{left_value} {self.op} {right_value} = {result}"]


def _generate_symbolic_arithmetic(rng: random.Random, seed: int, split: str, ood: bool) -> Example:
    max_depth = rng.randint(5, 8) if ood else rng.randint(2, 4)
    max_int = 50 if ood else 20
    expr = _sample_expr(rng, max_depth, max_int)
    answer_value, steps = expr.evaluate()
    rendered = expr.render()
    trace = ". ".join(steps) + f". The result is {answer_value}."
    prompt = f"Evaluate the expression:\n{rendered}\nReturn the integer result."
    return Example(
        prompt=prompt,
        trace=trace,
        answer=str(answer_value),
        metadata={
            "task": "symbolic_arithmetic",
            "split": split,
            "seed": seed,
            "max_depth": max_depth,
            "answer": answer_value,
        },
    )


def _sample_expr(rng: random.Random, depth: int, max_int: int) -> Expr:
    if depth <= 0:
        return Expr(value=rng.randint(0, max_int))
    if depth < 3 and rng.random() < 0.25:
        return Expr(value=rng.randint(0, max_int))
    return Expr(
        op=rng.choice(["+", "-"]),
        left=_sample_expr(rng, depth - 1, max_int),
        right=_sample_expr(rng, depth - 1, max_int),
    )


def parse_arithmetic_answer(candidate: str) -> int | None:
    normalized = _normalize_answer(candidate)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return None
    if not _is_safe_arithmetic_ast(tree):
        return None
    return int(eval(compile(tree, "<answer>", "eval"), {"__builtins__": {}}, {}))


def _is_safe_arithmetic_ast(node: ast.AST) -> bool:
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )
    if not isinstance(node, allowed):
        return False
    return all(_is_safe_arithmetic_ast(child) for child in ast.iter_child_nodes(node))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FDT synthetic task data as JSONL.")
    parser.add_argument("--task", choices=[*list_tasks(), "all"], default="all")
    parser.add_argument("--split", choices=["train", "dev", "id_test", "ood_test"], default="train")
    parser.add_argument("--difficulty", choices=["standard", "easy", "easy_ladder", "simple"], default="standard")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("data/debug"))
    args = parser.parse_args()

    tasks = list_tasks() if args.task == "all" else [args.task]
    for task in tasks:
        _validate_difficulty(task, args.difficulty)
        examples = [
            generate_example(task, args.seed_start + i, args.split, difficulty=args.difficulty)
            for i in range(args.num_examples)
        ]
        for example in examples:
            if not verify_answer(example, example.answer):
                raise RuntimeError(f"Self verification failed for {task} seed={example.metadata['seed']}")
        path = args.out_dir / args.split / f"{task}.jsonl"
        write_jsonl(examples, path)
        print(f"Wrote {len(examples)} examples to {path}")


if __name__ == "__main__":
    main()
