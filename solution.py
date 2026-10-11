"""Student entry point. See README.md for the API and scoring rules."""

import time
import heapq 

from sokoban import sokoban_goal_state

def heuristic(state):

    total = 0 
    

    for box in state.boxes:
        bx = box[0]
        by = box[1]

        min_dist = 999999

        for storage in state.storage:
            sx = storage[0]
            sy = storage[1]

            dist = abs(bx - sx) + abs(by - sy)

            if (dist < min_dist):
                min_dist = dist

        total = total + min_dist        

    return total



def solve(initial_state, timebound=120):
    """Return a goal SokobanState with a valid parent chain, or False."""

    start_time = time.monotonic()

    if timebound <= 0:
        return False

    best_g = {}
    best_g[initial_state.hashable_state()] = 0

    g0 = 0 
    h0 = heuristic(initial_state)
    f0 = g0 + h0

    heap = []
    count = 0

    heapq.heappush( heap, (f0, h0, count, initial_state) )

    while len(heap) > 0:
        
        if time.monotonic() - start_time > timebound :
            return False

        item = heapq.heappop(heap)
        current = item[3]

        key = current.hashable_state()

        if current.gval > best_g[key]:
            continue

        if sokoban_goal_state(current):
            return current

        for nxt in current.successors():
            key = nxt.hashable_state()
            g = nxt.gval

            if key in best_g:
                if best_g[key] <= g:
                    continue

            best_g[key] = g

            h = heuristic(nxt)
            f = g + h

            count = count + 1
            heapq.heappush(heap, (f, h, count, nxt))

    return False


