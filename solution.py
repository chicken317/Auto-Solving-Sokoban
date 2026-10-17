"""Student entry point. See README.md for the API and scoring rules."""

import time
import heapq 

from sokoban import sokoban_goal_state

UNREACHABLE = 999999999 

def reverse_bfs(goal, free_cells):                   # hàm BFS ngược: tính số lần đẩy tối thiểu từ mỗi vị trí của thùng đến một đích cụ thể.

    DIRECTIONS = [(0,1), (0,-1), (-1,0), (1,0)]      # lên, xuống, trái, phải
    distance = {}
    distance[goal] = 0
    queue = [goal]

    while len(queue) > 0:
        current = queue.pop(0)
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


def solve(initial_state, timebound=120):
    """Return a goal SokobanState with a valid parent chain, or False."""

    start_time = time.monotonic()

    if timebound <= 0:
        return False

    best_g = {}
    best_g[initial_state.hashable_state()] = 0

    g0 = 0 
    h_calc = heuristic(initial_state)
    h0 = h_calc(initial_state)
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

            h = h_calc(nxt)
            f = g + 2*h

            count = count + 1
            heapq.heappush(heap, (f, h, count, nxt))

    return False


