# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
"""
Cython-accelerated Space-Time A* for multi-agent pathfinding.

用 Cython 加速的空间-时间 A* 多智能体路径规划核心。
将 Python tuple/set/dict 操作替换为 C 级整数编码和高效数据结构。

编译:
    python setup_cython.py build_ext --inplace
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset

# We use Python set for reservation tables (shared with Python code)
# but encode keys as int64 for faster hashing.

import heapq
from collections import deque


def spatial_bfs(
    int start_r, int start_c,
    int goal_r, int goal_c,
    int rows, int cols,
    const unsigned char[:, :] walkable,
    set static_blocked_set,
):
    """
    Cython-accelerated BFS for spatial reachability check.

    Returns shortest path length (>= 1) if reachable, -1 if not.
    Ignores time and reservation tables — only checks static obstacles.
    """
    if start_r == goal_r and start_c == goal_c:
        return 0

    cdef set static_blocked = set()
    cdef int r, c
    cdef long long key

    for (r, c) in static_blocked_set:
        static_blocked.add(<long long>r * cols + c)

    # BFS with C-typed variables
    cdef int nr, nc, dist, d
    cdef int[4] dr = [-1, 1, 0, 0]
    cdef int[4] dc = [0, 0, -1, 1]

    cdef set visited = set()
    cdef long long start_key = <long long>start_r * cols + start_c
    visited.add(start_key)

    queue = deque()
    queue.append((start_r, start_c, 0))

    while queue:
        r, c, dist = queue.popleft()
        for d in range(4):
            nr = r + dr[d]
            nc = c + dc[d]
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if walkable[nr, nc] == 0:
                continue
            if nr == goal_r and nc == goal_c:
                return dist + 1
            key = <long long>nr * cols + nc
            if key not in visited and key not in static_blocked:
                visited.add(key)
                queue.append((nr, nc, dist + 1))

    return -1


# ---------------------------------------------------------------------------
# State encoding: pack (row, col, time) into a single 64-bit integer
# Layout: time * (rows * cols) + row * cols + col
# This avoids creating Python tuples for every state.
# ---------------------------------------------------------------------------

cdef inline long long encode3(int r, int c, int t, int cols, int rows_cols) nogil:
    """Encode (row, col, timestep) as a single int64."""
    return <long long>t * rows_cols + <long long>r * cols + c

cdef inline long long encode2(int r, int c, int cols) nogil:
    """Encode (row, col) as a single int64."""
    return <long long>r * cols + c

cdef inline long long encode5(int r1, int c1, int r2, int c2, int t,
                               int cols, int rows_cols) nogil:
    """Encode edge (r1,c1,r2,c2,t) for edge reservation lookup."""
    # Use two separate 3D encodings packed into upper/lower 32 bits
    return ((<long long>t * rows_cols + <long long>r1 * cols + c1) * rows_cols
            + <long long>r2 * cols + c2)

cdef inline int manhattan(int r1, int c1, int r2, int c2) nogil:
    """Manhattan distance."""
    cdef int dr = r1 - r2
    cdef int dc = c1 - c2
    if dr < 0:
        dr = -dr
    if dc < 0:
        dc = -dc
    return dr + dc


def space_time_astar(
    int start_r, int start_c,
    int goal_r, int goal_c,
    int rows, int cols,
    const unsigned char[:, :] walkable,
    set vertex_res_set,
    set edge_res_set,
    set static_blocked_set,
    int max_horizon,
    int goal_reserve,
    int max_expansions,
):
    """
    Cython-accelerated space-time A* search.

    Parameters
    ----------
    start_r, start_c : int
        Start position.
    goal_r, goal_c : int
        Goal position.
    rows, cols : int
        Grid dimensions.
    walkable : unsigned char[:, :]
        2D array, 1 = walkable, 0 = blocked.
    vertex_res_set : set of (int, int, int)
        Vertex reservations (row, col, timestep).
    edge_res_set : set of (int, int, int, int, int)
        Edge reservations (r1, c1, r2, c2, timestep).
    static_blocked_set : set of (int, int)
        Static blocked positions.
    max_horizon, goal_reserve, max_expansions : int
        Search limits.

    Returns
    -------
    list of (int, int) or empty list.
    """
    # Convert Python sets to encoded int64 sets for faster lookup
    cdef set vertex_res = set()
    cdef set edge_res = set()
    cdef set static_blocked = set()
    cdef int rows_cols = rows * cols

    cdef int r, c, t
    cdef int r1, c1, r2, c2
    cdef long long key

    for (r, c, t) in vertex_res_set:
        vertex_res.add(encode3(r, c, t, cols, rows_cols))

    for (r1, c1, r2, c2, t) in edge_res_set:
        edge_res.add(encode5(r1, c1, r2, c2, t, cols, rows_cols))

    for (r, c) in static_blocked_set:
        static_blocked.add(encode2(r, c, cols))

    # A* search
    cdef int counter = 0
    cdef long long start_key = encode3(start_r, start_c, 0, cols, rows_cols)
    cdef int h0 = manhattan(start_r, start_c, goal_r, goal_c)

    # open_set: (f, counter, r, c, t)
    open_set = [(h0, counter, start_r, start_c, 0)]
    counter += 1

    # g_score and came_from using encoded keys
    cdef dict g_score = {start_key: 0}
    cdef dict came_from = {}

    cdef int expansions = 0
    cdef int nr, nc, next_t, tentative_g, f_val
    cdef long long state_key, curr_key, neighbor_key, edge_key
    cdef int goal_blocked
    cdef int ft
    cdef long long goal_ft_key

    # Direction offsets: up, down, left, right
    cdef int[4] dr = [-1, 1, 0, 0]
    cdef int[4] dc = [0, 0, -1, 1]
    cdef int d

    while open_set:
        _, _, r, c, t = heapq.heappop(open_set)
        expansions += 1
        if expansions > max_expansions:
            break

        curr_key = encode3(r, c, t, cols, rows_cols)

        if r == goal_r and c == goal_c:
            # Reconstruct path
            path = []
            node_key = curr_key
            node_r, node_c, node_t = r, c, t
            while not (node_r == start_r and node_c == start_c and node_t == 0):
                path.append((node_r, node_c))
                parent_key = came_from[node_key]
                # Decode parent key
                node_t = <int>(parent_key // rows_cols)
                rem = <int>(parent_key % rows_cols)
                node_r = rem // cols
                node_c = rem % cols
                node_key = parent_key
            path.reverse()
            return path

        if t >= max_horizon:
            continue

        next_t = t + 1
        tentative_g = g_score.get(curr_key, 2147483647) + 1

        # Try 4 cardinal directions + wait-in-place (5 candidates)
        for d in range(5):
            if d < 4:
                nr = r + dr[d]
                nc = c + dc[d]
                # Bounds check
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                # Walkable check
                if walkable[nr, nc] == 0:
                    continue
            else:
                nr = r
                nc = c

            # Static blocked check
            neighbor_key = encode2(nr, nc, cols)
            if neighbor_key in static_blocked:
                continue

            # Vertex reservation check
            state_key = encode3(nr, nc, next_t, cols, rows_cols)
            if state_key in vertex_res:
                continue

            # Goal occupancy window check
            if nr == goal_r and nc == goal_c:
                goal_blocked = 0
                for ft in range(next_t + 1, next_t + 1 + goal_reserve):
                    goal_ft_key = encode3(nr, nc, ft, cols, rows_cols)
                    if goal_ft_key in vertex_res:
                        goal_blocked = 1
                        break
                if goal_blocked:
                    continue

            # Edge reservation check (swap conflict)
            edge_key = encode5(r, c, nr, nc, next_t, cols, rows_cols)
            if edge_key in edge_res:
                continue

            # Check if this is a better path
            if tentative_g < g_score.get(state_key, 2147483647):
                came_from[state_key] = curr_key
                g_score[state_key] = tentative_g
                f_val = tentative_g + manhattan(nr, nc, goal_r, goal_c)
                heapq.heappush(open_set, (f_val, counter, nr, nc, next_t))
                counter += 1

    return []
