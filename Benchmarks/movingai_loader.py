"""
MovingAI Benchmark Loader
=========================
Load standard MovingAI .map and .scen files, converting them into
the RMFS system's MapState representation for benchmark experiments.
"""

from typing import List, Tuple

from Config.config_loader import MapConfig
from WorldState.map_state import MapState


class MovingAILoader:
    """Load .map and .scen files from the MovingAI benchmark suite."""

    def load_map(self, map_path: str) -> MapState:
        """Parse a MovingAI .map file into a MapState.

        Format: 4-line header (type / height / width / 'map') followed by
        *height* lines of grid characters.

        Character mapping:
            '.', 'G', 'S'       -> FREE
            '@', 'O', 'T'       -> OBSTACLE
            anything else       -> FREE
        """
        with open(map_path) as f:
            lines = f.readlines()

        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])

        obstacles: List[Tuple[int, int]] = []
        for r, line in enumerate(lines[4:4 + height]):
            for c, ch in enumerate(line.strip()):
                if ch in ('@', 'O', 'T'):
                    obstacles.append((r, c))

        map_config = MapConfig(
            rows=height,
            cols=width,
            obstacles=obstacles,
            stations=[],
            pod_zones=[],
        )
        return MapState(map_config)

    def load_scenario(self, scen_path: str) -> List[dict]:
        """Parse a MovingAI .scen file into a list of start/goal pairs.

        Each entry: {"start": (row, col), "goal": (row, col)}

        .scen column order (tab-separated):
            bucket, map, width, height, start_col, start_row, goal_col, goal_row, optimal_length
        """
        scenarios: List[dict] = []
        with open(scen_path) as f:
            for line in f:
                if line.startswith("version"):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 9:
                    sc, sr = int(parts[4]), int(parts[5])
                    gc, gr = int(parts[6]), int(parts[7])
                    scenarios.append({
                        "start": (sr, sc),
                        "goal": (gr, gc),
                    })
        return scenarios
