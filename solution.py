"""Student entry point. See README.md for the API and scoring rules."""

import time
import heapq 
from collections import deque

from sokoban import sokoban_goal_state

UNREACHABLE = 999999999 
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


def hungarian_matching(cost):
    m = len(cost)       # số hàng -> số thùng
    if m == 0:
        return 0
    n = len(cost[0])    # số cột -> số đích
    if m > n: 
        return UNREACHABLE
    
    BIG = 1             # xử lí những  nơi mà không thể ghép thùng với đích. 
                        # sử dụng BIG để không bị nhầm với UNREACHABLE(đánh dấu những đường không đi được) 
    for row in cost:
        max_cost = 0    # khởi tạo max_cost = 0 để tính toán
        for c in row:
            if c < UNREACHABLE and c > max_cost:
                max_cost = c
        BIG += max_cost
    # tạo ma trận a, thay những chỗ UNREACHABLE bằng BIG 
    a = []
    for row in cost:
        new_row = []
        for c in row:
            if c >= UNREACHABLE:
                new_row.append(BIG)
            else:
                new_row.append(c)
        a.append(new_row)       # a gồm các list là các hàng của ma trận cũ nhưng thay UNREACHABLE = BIG để xử lí

    # thuật toán Hungarian (phát triển từ Hungarian cổ điển)
    INF = 10**18           
    u = [0] * (m + 1)        # thế vị của hàng (thùng). khởi tạo đến m + 1 do ở code chỉ số bắt đầu từ 0, nhưng ở ma trận chỉ số bắt đầu từ 1. 
    v = [0] * (n + 1)        # thế vị của cột (đích). tương tự
    p = [0] * (n + 1)           # p[j] = i: thùng i ghép với đích j (hàng i ghép với cột j)
    way = [0] * (n + 1)         # lưu trữ đường đi để tìm cột trước đó. way[j] = cột đứng trước j khi truy vết -> sử dụng để đổi phép ghép cặp

    for i in range (1, m + 1):
        p[0] = i                # bắt đầu tìm kiếm xem thùng i gán với đích nào. cột 0 là cột giả để thể hiện bắt đầu thuật toán
        j0 = 0                  # j0 = 0: cột giả 
        minv = [INF] * (n + 1)              # khởi tạo chi phí nhỏ nhất khi đi tới cột cụ thể, bắt đầu bằng INF
        visited = [False] * (n + 1)         # khởi tạo visited, cho tất cả bằng false đánh dấu các ô chưa được thăm
        # tìm kiếm

        while True:
            visited[j0] = True              # thăm j0
            i0 = p[j0]                      # cho i0 là thùng nối với đích j0
            delta = INF                     # khởi tạo delta là chi phí nhỏ nhất trong các cột chưa thăm, là min của minv
            j1 = 0                          # khởi tạo cột tiếp theo được chọn, ban đầu chưa chọn cột nào. j1 là cột tiếp theo được chọn (trong vòng tìm kiếm)
            
            for j in range (1, n + 1):
                if not visited[j]:          # nếu chưa thăm j
                    cur = a[i0 - 1][j - 1]  - u[i0] - v[j]           # tính chi phí giảm khi ghép hàng i0 với cột j
                    
                    if cur < minv[j]: 
                        minv[j] = cur
                        way[j] = j0                                  # lưu cột đứng trước j để sau này truy vết nếu cần
                    
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j                                       # lưu cột j là cột chọn tiếp theo
            
            for j in range (n + 1):
                if visited[j]: 
                    u[p[j]] += delta                                 # tăng thế vị hàng đã thăm để điều chỉnh chi phí giảm
                    v[j] -= delta                                    # giảm thế vị cột đã thăm để bù trừ, giữ nguyên chi phí giảm giữa các hàng và cột đã thăm, tránh làm mất số 0 đã tạo được trong hungarian
                else:
                    minv[j] -= delta                                    # trừ delta ở minv để đưa ít nhất một minv về 0, giúp thuật toán tiếp tục tìm đường ghép

            j0 = j1                                                  # cập nhật cột hiện tại thành cột có minv nhỏ nhất để tiếp tục tìm đường ghép
            if p[j0] == 0:                                           # nếu đích hiện tại không ghép thùng nào
                break

        #truy vết
        while j0 != 0:                                               # cột hiện tại khác 0
            j1 = way[j0]                                             # j1 là cột đứng trước j0 (trong vòng truy vết)
            p[j0] = p[j1]                                            # Chuyển hàng đang ghép với cột trước sang cột hiện tại
            j0 = j1                                                  # Quay về cột trước đó để tiếp tục lần ngược đường ghép
    
    total = 0
    for j in range (1, n + 1):
        if p[j] != 0: 
            total = total + a[p[j] - 1][j - 1]
    if total >= BIG: 
        return UNREACHABLE
    return total


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

        result = hungarian_matching(cost)       # ghép cặp tối ưu bằng Hungarian: O(m^2 * n) thay cho quay lui

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

        found = weighted_search( initial_state, h_calc, weight, limit, MAX_STATES, deadlock_data )

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