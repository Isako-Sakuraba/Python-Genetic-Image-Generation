from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evolution import EvolutionStatistics


def plot_fitness_evolution(
    statistics: list[list["EvolutionStatistics"]],
    output_path: str = "fitness_evolution.png",
    width: int = 1200,
    height: int = 800,
) -> None:
    import matplotlib.pyplot as plt

    tile_count = len(statistics)
    generation_count = len(statistics[0]) if tile_count > 0 else 0

    figure, axis = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)

    for tile_id in range(tile_count):
        generations = [statistics[tile_id][generation].generation for generation in range(generation_count)]
        fitness = [statistics[tile_id][generation].average_fitness for generation in range(generation_count)]
        axis.plot(generations, fitness, linewidth=1, color="gray", alpha=0.15)

    mean_generations = list(range(generation_count))
    mean_fitness: list[float] = []
    min_fitness: list[float] = []
    max_fitness: list[float] = []

    for generation in range(generation_count):
        generation_fitness = [statistics[tile_id][generation].average_fitness for tile_id in range(tile_count)]
        if not generation_fitness:
            mean_fitness.append(0.0)
            min_fitness.append(0.0)
            max_fitness.append(0.0)
            continue

        mean_fitness.append(sum(generation_fitness) / len(generation_fitness))
        min_fitness.append(min(generation_fitness))
        max_fitness.append(max(generation_fitness))

    axis.plot(mean_generations, min_fitness, linewidth=2, color="green", label="Min Fitness")
    axis.plot(mean_generations, max_fitness, linewidth=2, color="red", label="Max Fitness")
    axis.plot(mean_generations, mean_fitness, linewidth=3, color="blue", label="Mean Fitness")

    axis.set_title("Fitness Evolution Across Tiles")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Average Fitness")
    axis.legend(loc="upper right")

    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)

    print(f"Plot saved to: {output}")
