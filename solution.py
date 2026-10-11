"""Student entry point. See README.md for the API and scoring rules."""

import time
from collections import deque

from sokoban import sokoban_goal_state

def solve(initial_state, timebound=120):
    """Return a goal SokobanState with a valid parent chain, or False."""
    
    start_time = time.monotonic()

    if timebound < 0:
        return False

    if sokoban_goal_state(initial_state):
        return initial_state

    queue = deque([initial_state])
    visited = {initial_state.hashable_state()}

    while queue:

        if time.monotonic() - start_time > timebound:
            return False

        current = queue.popleft()
        for nxt in current.successors():
            key = nxt.hashable_state()

            if key in visited:
                continue

            if sokoban_goal_state(nxt):
                return nxt

            visited.add(key)
            queue.append(nxt)

    return False
    

    raise NotImplementedError("Implement your Sokoban solver")

