from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Sequence

from .visualization import plot_fitness_evolution

if TYPE_CHECKING:
    from .evolution import EvolutionStatistics


INPUT_FOLDER = Path("input")

DEFAULT_INPUT = "input"
DEFAULT_OUTPUT = "output"
DEFAULT_GENERATIONS = 40
DEFAULT_POPULATION_SIZE = 120
DEFAULT_CROSSOVER = 50
DEFAULT_WORKING_RESOLUTION = 512
DEFAULT_OUTPUT_SIZE = 512
DEFAULT_SUBDIVISION_LEVEL = 3
DEFAULT_COLOR_GENERATIONS = 24
DEFAULT_RANDOM_SEED = 0
DEFAULT_STATISTICS = False
TOP_COUNT = 5


GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class SpacedDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings:
            return super()._format_action_invocation(action)

        if action.nargs == 0:
            return ", ".join(f"{GREEN}{option}{RESET}" for option in action.option_strings)

        default_metavar = self._get_default_metavar_for_optional(action)
        argument_string = self._format_args(action, default_metavar)
        colored_argument = f"{YELLOW}{argument_string}{RESET}"

        return ", ".join(
            f"{GREEN}{option}{RESET} {colored_argument}" for option in action.option_strings
        )

    def _format_action(self, action: argparse.Action) -> str:
        formatted = super()._format_action(action)
        if action.option_strings:
            return f"{formatted}\n"
        return formatted


@dataclass(frozen=True, slots=True)
class RunData:
    resize: int
    outsize: int
    quantization: int
    seed: int
    runtime: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "resize": self.resize,
            "outsize": self.outsize,
            "quantization": self.quantization,
            "seed": self.seed,
            "runtime": self.runtime,
        }


def _nearest_resampling() -> int:
    image_module = importlib.import_module("PIL.Image")

    if hasattr(image_module, "Resampling"):
        return image_module.Resampling.NEAREST
    return image_module.NEAREST


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("Boolean value must be 'true' or 'false'.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stylized images with a quadtree-based genetic algorithm that "
            "evolves toward an input image."
        ),
        formatter_class=SpacedDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="input filename")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="output folder")
    parser.add_argument(
        "-or",
        "--output-resolution",
        type=int,
        default=DEFAULT_OUTPUT_SIZE,
        help="final output resolution",
    )
    parser.add_argument("-g", "--generations", type=int, default=DEFAULT_GENERATIONS, help="generation count")
    parser.add_argument(
        "-ps",
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help="population size per generation",
    )
    parser.add_argument("-c", "--crossover", type=int, default=DEFAULT_CROSSOVER, help="crossover size")
    parser.add_argument(
        "-cg",
        "--color-generations",
        type=int,
        default=DEFAULT_COLOR_GENERATIONS,
        help="color mutation attempts",
    )
    parser.add_argument(
        "-wr",
        "--working-resolution",
        type=int,
        default=DEFAULT_WORKING_RESOLUTION,
        help="internal working resolution",
    )
    parser.add_argument(
        "-sl",
        "--subdivision-level",
        type=int,
        default=DEFAULT_SUBDIVISION_LEVEL,
        help="tile subdivision level (2^level per axis)",
    )
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_RANDOM_SEED, help="random seed")
    parser.add_argument(
        "-st",
        "--statistics",
        type=_parse_bool,
        default=DEFAULT_STATISTICS,
        help="enable statistics output",
    )

    return parser


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    normalized = list(argv)
    if normalized and normalized[0] == "--":
        return normalized[1:]
    return normalized


def _resolve_input_path(input_value: str) -> Path:
    raw = Path(input_value)
    if raw.suffix:
        if raw.parent == Path("."):
            return INPUT_FOLDER / raw
        return raw

    with_jpg = raw.with_suffix(".jpg")
    if raw.parent == Path("."):
        return INPUT_FOLDER / with_jpg
    return with_jpg


def _delete_previous_files(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    for child in output_folder.iterdir():
        if child.is_file():
            child.unlink()


def _load_image(input_path: Path, resize: int) -> object:
    numpy_module = importlib.import_module("numpy")
    image_module = importlib.import_module("PIL.Image")

    if not input_path.exists():
        raise FileNotFoundError(f"Input image was not found: {input_path}")

    with image_module.open(input_path) as source_image:
        source_rgb = source_image.convert("RGB")
        resized = source_rgb.resize((resize, resize), _nearest_resampling())
        return numpy_module.asarray(resized, dtype=numpy_module.uint8).copy()


def _print_current_configuration(input_path: Path, parsed: argparse.Namespace) -> None:
    print(f"Input file: {input_path.name}")
    print(f"Output resolution: {parsed.output_resolution}x{parsed.output_resolution}")
    print(f"Working resolution: {parsed.working_resolution}x{parsed.working_resolution}")
    print(f"Subdivision level: {parsed.subdivision_level}")
    print(f"Generations: {parsed.generations}")
    print(f"Population Size: {parsed.population_size}")
    print(f"Crossover size: {parsed.crossover}")
    print(f"Color Generations: {parsed.color_generations}")
    print(f"Random seed: {parsed.seed}")


def _save_results(results: list[object], output_folder: Path, output_size: int) -> None:
    image_module = importlib.import_module("PIL.Image")

    for result_id, result in enumerate(results):
        image = image_module.fromarray(result, mode="RGB")
        resized = image.resize((output_size, output_size), _nearest_resampling())
        output_path = output_folder / f"output_{result_id}.jpg"
        resized.save(output_path, format="JPEG")
        resized.close()
        image.close()


def _serialize_statistics(
    statistics: list[list["EvolutionStatistics"]],
) -> list[list[dict[str, int | float]]]:
    return [[record.to_dict() for record in tile_row] for tile_row in statistics]


def _format_runtime(seconds: float) -> str:
    total_ticks = int(seconds * 10_000_000)
    hours, remainder = divmod(total_ticks, 36_000_000_000)
    minutes, remainder = divmod(remainder, 600_000_000)
    secs, fraction = divmod(remainder, 10_000_000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{fraction:07}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = _normalize_argv(sys.argv[1:] if argv is None else argv)

    parsed = parser.parse_args(arguments)

    try:
        try:
            from .evolution import Evolution, EvolutionData
        except ModuleNotFoundError as exc:
            missing_module = getattr(exc, "name", "dependency")
            print(
                f"Error: Missing dependency '{missing_module}'. "
                "Install requirements in PythonRewrite before running."
            )
            return 1

        output_folder = Path(parsed.output)
        _delete_previous_files(output_folder)

        input_path = _resolve_input_path(parsed.input)
        original = _load_image(input_path, parsed.working_resolution)

        evolution_data = EvolutionData(
            generations=parsed.generations,
            generation_size=parsed.population_size,
            color_generations=parsed.color_generations,
            crossover=parsed.crossover,
        )
        evolution = Evolution(original, evolution_data, parsed.seed)

        _print_current_configuration(input_path, parsed)
        print("\n=== Evolution started ===\n")

        start_time = time.perf_counter()
        results = evolution.evolve_async(parsed.subdivision_level, TOP_COUNT)
        elapsed_seconds = time.perf_counter() - start_time

        print("\n=== Evolution finished ===\n")

        if parsed.statistics:
            for tile_id, tile_stats in enumerate(evolution.statistics):
                print(f"=== Last Fitness for Quad {tile_id} ===")
                start_generation = max(0, len(tile_stats) - 3)
                for generation_index in range(start_generation, len(tile_stats)):
                    stat = tile_stats[generation_index]
                    print(
                        f"Generation: {stat.generation} | Average Fitness {stat.average_fitness}"
                    )

            graph_path = output_folder / "graph.png"
            plot_fitness_evolution(evolution.statistics, str(graph_path))

        _save_results(results, output_folder, parsed.output_resolution)

        run_data = RunData(
            resize=parsed.working_resolution,
            outsize=parsed.output_resolution,
            quantization=parsed.subdivision_level,
            seed=parsed.seed,
            runtime=_format_runtime(elapsed_seconds),
        )

        (output_folder / "run_data.txt").write_text(
            json.dumps(run_data.to_dict(), separators=(",", ":")),
            encoding="utf-8",
        )
        (output_folder / "evo_data.txt").write_text(
            json.dumps(evolution_data.to_dict(), separators=(",", ":")),
            encoding="utf-8",
        )
        (output_folder / "stat_data.txt").write_text(
            json.dumps(_serialize_statistics(evolution.statistics), separators=(",", ":")),
            encoding="utf-8",
        )

        print()
        print("Results saved as output_n.jpg")
        print(f"Elapsed time: {run_data.runtime}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
