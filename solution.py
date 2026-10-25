"""Student entry point. See README.md for the API and scoring rules."""

import time
import heapq 
from collections import deque

from sokoban import sokoban_goal_state

UNREACHABLE = 999999999                              # đánh đấu những cặp thùng-đích không thể ghép với nhau
WEIGHTS = [3, 2, 1.5, 1]
REFINE_SECONDS = 10
MAX_STATES = 1500000

def reverse_bfs(goal, free_cells):                   # hàm BFS ngược: tính số lần đẩy tối thiểu từ mỗi vị trí của thùng đến một đích cụ thể.

    DIRECTIONS = [(0,1), (0,-1), (-1,0), (1,0)]      # lên, xuống, trái, phải
    distance = {}
    distance[goal] = 0
    queue = deque([goal])

    while len(queue) > 0:
        current = queue.popleft()
        x = current[0]
        y = current[1]

        for direction in DIRECTIONS:
            dx = direction[0]
            dy = direction[1]
            box_cell = (x - dx, y - dy)             # box tại vị trí trước khi đến đích
            robot_cell = (x - 2*dx, y - 2*dy)       # robot ở vị trí để đẩy thùng vào đích

            if box_cell in free_cells and robot_cell in free_cells:
                if box_cell not in distance:
                    distance[box_cell] = distance[current] + 1
                    queue.append(box_cell)

    return distance

def best_matching(cost_matrix, box_index, used_goals):

    if box_index >= len(cost_matrix):
        return 0
    
    min_cost = UNREACHABLE

    for goal_index in range(len(cost_matrix[box_index])):

        if (not used_goals[goal_index]) and cost_matrix[box_index][goal_index] < UNREACHABLE:
            used_goals[goal_index] = True
            total_cost = cost_matrix[box_index][goal_index] + best_matching(cost_matrix, box_index + 1, used_goals)
            used_goals[goal_index] = False

            if total_cost < min_cost: 
                min_cost = total_cost
                
    return min_cost

class heuristic:

    def __init__(self, state):
        self.free_cells = set()     # dùng set để tối ưu: kiểm tra mất O(1) thay vì dùng list duyệt mất O(n)
        for x in range (state.width):
            for y in range (state.height):
                if (x, y) not in state.obstacles:       # nếu tọa độ kh nằm trong tập các ô tường
                    self.free_cells.add((x, y))
        
        self.goals = list(state.storage)        # danh sách goals thuận tiện khi truy cập theo chỉ số và giữ thứ tự tương ứng với self.tables.
        self.tables = []
        for goal in self.goals:
            self.tables.append(reverse_bfs(goal, self.free_cells))
        self.alive_cells = set()
        for table in self.tables:
            for cell in table:
                self.alive_cells.add(cell)
        self.cache = {}         # tạo 1 dictionary để lưu lại kết quả heuristics đã tính

    def __call__(self, state):
        boxes = state.boxes
        if boxes in self.cache:
            return self.cache[boxes]
        for box in boxes:
            if box not in self.alive_cells: 
                self.cache[boxes] = UNREACHABLE
                return UNREACHABLE
        # cost[i][j]: chi phí đưa thùng i tới đích j
        cost = []
        for box in boxes:
            row = []
            for table in self.tables:
                if box in table:
                    row.append(table[box])
                else:
                    row.append(UNREACHABLE)
            cost.append(row)

        used = [False] * len(self.goals)        # đánh dấu đích. nếu đã sử dụng -> true
        result = best_matching(cost, 0, used) 

        self.cache[boxes] = result
        return result  


def state_key(state) :
    return (tuple(sorted(state.robots)), state.boxes)


def weighted_search(initial_state, h_calc, weight, deadline, max_states, deadlock_data) :

    wall, dead, goals = deadlock_data
    g0 = 0
    h0 = h_calc(initial_state)
    f0 = g0 + h0 * weight

    best_g = { state_key(initial_state) : 0 }

    count = 0 

    heap = []
    heapq.heappush(heap, (f0, h0, count, initial_state))

    while len(heap) > 0 :

        if time.monotonic() > deadline:
            return None

        item = heapq.heappop(heap)
        curr_state = item[3]

        if curr_state.gval > best_g[state_key(curr_state)]:
            continue

        if sokoban_goal_state(curr_state):
            return curr_state 

        for nxt in curr_state.successors() :
            h = h_calc(nxt)

            if h >= UNREACHABLE:
                continue

            pushed = nxt.boxes - curr_state.boxes
            if pushed :
                moved  = next(iter(pushed))

                if deadlock(moved, nxt.boxes, nxt.robots, wall, dead, goals):
                    continue

            key = state_key(nxt)
            g = nxt.gval

            if ( key in best_g ) and ( best_g[key] <= g ) :
                continue
            best_g[key] = g

            if len(best_g) > max_states:
                return None
            
            count = count + 1

            heapq.heappush( heap, (g + weight * h, h, count, nxt) )

    return None


def solve(initial_state, timebound = 120) :

    start_time = time.monotonic()

    if timebound <= 0:
        return False

    deadline = start_time + timebound * 0.95

    h_calc = heuristic(initial_state)

    if h_calc(initial_state) >= UNREACHABLE:
        return False

    deadlock_data = prepare_deadlock(initial_state.width, initial_state.height,
                                     initial_state.obstacles, initial_state.storage)
    
    wall, dead, goals = deadlock_data

    if deadlock_full(initial_state.boxes, initial_state.robots, wall, dead, goals):
        return False                      

    best = None
    limit = deadline


    for weight in WEIGHTS :
        
        if time.monotonic() > limit:
            break

        found = weighted_search( initial_state, h_calc, weight, deadline, MAX_STATES, deadlock_data )

        if found is None:
            break

        if best is None:
            limit = min(deadline, time.monotonic() + REFINE_SECONDS)

        if ( best is None ) or ( found.gval < best.gval ) :
            best = found 

        if weight == 1:
            break
        
    if best is None:
        return False

    return best    

DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))

def prepare_deadlock(width, height, obtacles, storage):

    wall = set(obtacles)

    for x in range(-1, width + 1) :
        
        wall.add( (x, -1) )
        wall.add( (x, height) )

    for y in range(-1, height + 1) :
        wall.add( (-1, y) )
        wall.add( (width, y) )

    alive = set(storage)
    stack = list(storage)

    while stack:
        x,y = stack.pop()

        for dx, dy in DIRS :
            pre_box = (x - dx, y - dy)
            pre_robot = (x - 2*dx, y - 2*dy)

            if (( pre_box in alive ) or
                ( pre_box in wall ) or
                ( pre_robot in wall ) ) :
                continue
            alive.add(pre_box)
            stack.append(pre_box)

    dead = set()

    for x in range(width):
        for y in range(height):
            pos = (x, y)

            if (pos not in wall) and (pos not in alive):
                dead.add(pos)

    return wall, dead, set(storage)


def axis_blocked( left_cell, right_cell,boxes, checked, wall, dead ) :

    if ( left_cell in wall ) or ( right_cell in wall ):
        return True

    if ( left_cell in dead ) and ( right_cell in dead ):
        return True

    if left_cell in boxes:
        if (( left_cell in checked ) or
            is_frozen(left_cell, boxes, checked, wall, dead)) :
                return True

    if right_cell in boxes:
        if (( right_cell in checked ) or 
            is_frozen(right_cell, boxes, checked, wall, dead)) :
                return True

    return False


def is_frozen(box, boxes, checked, wall, dead) :

    checked = checked | { box }

    x,y = box

    return ( axis_blocked( (x - 1, y), (x + 1, y), boxes, checked, wall, dead ) and
             axis_blocked( (x, y - 1), (x, y + 1), boxes, checked, wall, dead ) )


def corral_deadlock( boxes, robots, frozen_boxes, wall, goals ) :

    reachable_pos = set(robots)
    stack = list(robots)

    while stack :

        x,y = stack.pop()

        for dx, dy in DIRS:
            nxt = ( x + dx, y + dy )

            if ((nxt in wall) or 
                (nxt in frozen_boxes) or
                (nxt in reachable_pos) ) :
                    continue
            reachable_pos.add(nxt)
            stack.append(nxt)

    for box in boxes :
        if box not in goals and box not in reachable_pos:
            return True
    
    return False


def deadlock( moved, boxes, robots, wall, dead, goals ) :

    if ( moved in dead ) and ( moved not in goals ) : 
        return True

    if not is_frozen( moved, boxes, frozenset(), wall, dead ) :
        return False

    frozen_on_goal = set()

    for box in boxes:
        if ( box not in goals ) and ( box in dead ) :
            return True

        if is_frozen( box, boxes, frozenset(), wall, dead ) :
            if box not in goals:
                return True
            frozen_on_goal.add((box))

    return corral_deadlock(boxes, robots, frozen_on_goal, wall, goals)


def deadlock_full(boxes, robots, wall, dead, goals):

    frozen_on_goal = set()

    for box in boxes:
        if ( box not in goals ) and ( box in dead ) :
            return True

        if is_frozen( box, boxes, frozenset(), wall, dead ) :
            if ( box not in goals ) :
                return True 
            frozen_on_goal.add(box)

    return corral_deadlock(boxes, robots, frozen_on_goal, wall, goals)


