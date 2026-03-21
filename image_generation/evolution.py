from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
import os
import random
import threading
from typing import Callable

import numpy as np

from .color_filters import adjust_saturation
from .extensions import Color, next_color, remap
from .quadtree import QuadTreeImage, Rectangle

try:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
except ImportError:  # pragma: no cover - optional dependency fallback
    Progress = None


WHEAT: Color = (245, 222, 179)


@dataclass(slots=True)
class EvolutionStatistics:
    id: int
    generation: int
    average_fitness: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "id": self.id,
            "generation": self.generation,
            "averageFitness": self.average_fitness,
        }


@dataclass(slots=True)
class EvolutionData:
    generations: int = 40
    generation_size: int = 100
    color_generations: int = 90
    crossover: int = 40

    def to_dict(self) -> dict[str, int]:
        return {
            "generations": self.generations,
            "generationSize": self.generation_size,
            "colorGenerations": self.color_generations,
            "crossover": self.crossover,
        }


@dataclass(slots=True)
class MutationData:
    fitness: float
    rectangle_index: int
    base_color: Color
    secondary_color: Color


class Evolution:
    def __init__(self, original: np.ndarray, data: EvolutionData, seed: int = 0, id: int = 0) -> None:
        self.id = id
        self.original = original
        self.data = data
        self.seed = seed
        self.random = random.Random(seed)

        height, width = original.shape[:2]
        self.elite_individual = QuadTreeImage.blank(width, height)
        self.population = [
            MutationData(float("inf"), 0, WHEAT, WHEAT)
            for _ in range(self.data.generation_size)
        ]
        self.statistics = [
            [
                EvolutionStatistics(0, generation_index, 0.0)
                for generation_index in range(self.data.generations)
            ]
        ]
        self.on_generation_complete: Callable[[int], None] = lambda _generation: None
        self._original_int16 = self.original.astype(np.int16, copy=False)

    def fitness_function(self, quad_tree: QuadTreeImage) -> float:
        difference = np.abs(self._original_int16 - quad_tree.image.astype(np.int16))
        return float(np.mean(difference))

    def create_new_population(self, generation_number: int) -> None:
        average_fitness = 0.0
        for individual_index in range(self.data.generation_size):
            average_fitness += self.mutate_quad(individual_index)

        if self.data.generation_size > 0:
            average_fitness /= self.data.generation_size

        self.statistics[0][generation_number] = EvolutionStatistics(
            self.id,
            generation_number,
            average_fitness,
        )

    def mutate_quad(self, individual_index: int) -> float:
        current_tree_image = QuadTreeImage.from_quadtree(self.elite_individual, clone_image=True)
        random_rect_index = self.random.randrange(0, len(current_tree_image.quads))

        main_color = WHEAT
        secondary_color = WHEAT
        best_fitness = float("inf")

        for _ in range(self.data.color_generations):
            random_color = next_color(self.random)
            adjusted_color = adjust_saturation(
                random_color,
                remap(self.random.random(), 0.4, 2.4),
            )

            current_tree_image.draw(random_rect_index, random_color, adjusted_color)

            current_fitness = self.fitness_function(current_tree_image)
            if current_fitness < best_fitness:
                main_color = random_color
                secondary_color = adjusted_color
                best_fitness = current_fitness

        self.population[individual_index] = MutationData(
            best_fitness,
            random_rect_index,
            main_color,
            secondary_color,
        )
        return best_fitness

    def create_next_elite_individual(self) -> QuadTreeImage:
        self.population.sort(key=lambda mutation: mutation.fitness)

        quad_tree = QuadTreeImage.from_quadtree(self.elite_individual, clone_image=False)
        used_indexes: set[int] = set()
        chunks = [0, 0, 0, 0]

        max_crossover = min(self.data.crossover, len(self.population))
        for crossover_index in range(max_crossover):
            mutation = self.population[crossover_index]

            if mutation.rectangle_index in used_indexes:
                continue

            used_indexes.add(mutation.rectangle_index)

            if quad_tree.can_split(mutation.rectangle_index):
                quad_tree.draw(
                    mutation.rectangle_index,
                    mutation.base_color,
                    mutation.secondary_color,
                )
                quad_tree.try_split_non_alloc(mutation.rectangle_index, chunks)

        return quad_tree

    def evolve(self, top_count: int = 5) -> list[np.ndarray]:
        for generation_index in range(self.data.generations):
            self.create_new_population(generation_index)
            best = self.create_next_elite_individual()
            self.elite_individual = best
            self.on_generation_complete(generation_index + 1)

        return self.create_top_individuals(top_count)

    def create_top_individuals(self, count: int) -> list[np.ndarray]:
        self.population.sort(key=lambda mutation: mutation.fitness)

        actual_count = min(count, len(self.population))
        images: list[np.ndarray] = []

        for top_index in range(actual_count):
            quad_tree = QuadTreeImage.from_quadtree(self.elite_individual, clone_image=False)
            used_indexes: set[int] = set()
            chunks = [0, 0, 0, 0]

            max_mutation_index = min(top_index + self.data.crossover, len(self.population) - 1)
            for mutation_index in range(max_mutation_index + 1):
                mutation = self.population[mutation_index]

                if mutation.rectangle_index in used_indexes:
                    continue

                used_indexes.add(mutation.rectangle_index)

                if quad_tree.can_split(mutation.rectangle_index):
                    quad_tree.draw(
                        mutation.rectangle_index,
                        mutation.base_color,
                        mutation.secondary_color,
                    )
                    quad_tree.try_split_non_alloc(mutation.rectangle_index, chunks)

            images.append(quad_tree.image.copy())

        return images

    def evolve_async(self, quantization: int, top_count: int = 5, show_progress: bool = True) -> list[np.ndarray]:
        tiles_x = int(math.pow(2, quantization))
        tiles_y = tiles_x
        tile_width = self.original.shape[1] // tiles_x
        tile_height = self.original.shape[0] // tiles_y

        tile_infos: list[tuple[int, Rectangle, np.ndarray]] = []
        self.statistics = [
            [
                EvolutionStatistics(tile_id, generation_index, 0.0)
                for generation_index in range(self.data.generations)
            ]
            for tile_id in range(tiles_x * tiles_y)
        ]

        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                tile_id = tile_y * tiles_x + tile_x
                rectangle = Rectangle(tile_x * tile_width, tile_y * tile_height, tile_width, tile_height)
                tile = self.original[
                    rectangle.y : rectangle.y + rectangle.height,
                    rectangle.x : rectangle.x + rectangle.width,
                ].copy()
                tile_infos.append((tile_id, rectangle, tile))

        progress_state: tuple[Progress, dict[int, int], threading.Lock] | None = None
        results: dict[int, tuple[Rectangle, list[np.ndarray]]]

        if Progress is not None and show_progress:
            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                SpinnerColumn(),
                transient=False,
            ) as progress:
                task_ids = {
                    tile_id: progress.add_task(f"Tile {tile_id + 1}", total=self.data.generations)
                    for tile_id, _, _ in tile_infos
                }
                progress_state = (progress, task_ids, threading.Lock())
                results = self._run_tile_evolutions(tile_infos, top_count, progress_state)
        else:
            results = self._run_tile_evolutions(tile_infos, top_count, progress_state)

        outputs: list[np.ndarray] = []
        for image_index in range(top_count):
            output = np.zeros_like(self.original)
            for tile_id in range(len(tile_infos)):
                rectangle, evolved_array = results[tile_id]
                if image_index >= len(evolved_array):
                    continue

                output[
                    rectangle.y : rectangle.y + rectangle.height,
                    rectangle.x : rectangle.x + rectangle.width,
                ] = evolved_array[image_index]

            outputs.append(output)

        return outputs

    def _run_tile_evolutions(
        self,
        tile_infos: list[tuple[int, Rectangle, np.ndarray]],
        top_count: int,
        progress_state: tuple[Progress, dict[int, int], threading.Lock] | None,
    ) -> dict[int, tuple[Rectangle, list[np.ndarray]]]:
        results: dict[int, tuple[Rectangle, list[np.ndarray]]] = {}

        max_workers = max(1, min(len(tile_infos), os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: list[Future[tuple[int, Rectangle, list[np.ndarray], list[EvolutionStatistics]]]] = []

            for tile_id, rectangle, tile in tile_infos:
                callback = self._build_generation_callback(progress_state, tile_id)
                futures.append(
                    executor.submit(self._evolve_tile, tile_id, rectangle, tile, top_count, callback)
                )

            for future in as_completed(futures):
                tile_id, rectangle, evolved, tile_statistics = future.result()
                self.statistics[tile_id] = tile_statistics
                results[tile_id] = (rectangle, evolved)

        return results

    def _build_generation_callback(
        self,
        progress_state: tuple[Progress, dict[int, int], threading.Lock] | None,
        tile_id: int,
    ) -> Callable[[int], None]:
        if progress_state is None:
            return lambda _generation: None

        progress, task_ids, progress_lock = progress_state

        def on_generation(_generation: int) -> None:
            with progress_lock:
                progress.advance(task_ids[tile_id], 1)

        return on_generation

    def _evolve_tile(
        self,
        tile_id: int,
        rectangle: Rectangle,
        tile: np.ndarray,
        top_count: int,
        callback: Callable[[int], None],
    ) -> tuple[int, Rectangle, list[np.ndarray], list[EvolutionStatistics]]:
        evolution = Evolution(tile, self.data, self.seed, tile_id)
        evolution.on_generation_complete = callback
        evolved = evolution.evolve(top_count)
        return tile_id, rectangle, evolved, list(evolution.statistics[0])
