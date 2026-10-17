import time
from sokoban import PROBLEMS
import solution

print("=" * 55)
print(f"{'MAP':<8} | {'KẾT QUẢ':<10} | {'SỐ BƯỚC':<10} | {'THỜI GIAN':<10}")
print("=" * 55)

# Chạy thử 20 map (hoặc đổi thành range(5) để chạy thử 5 map đầu)
for i in range(20):
    map_state = PROBLEMS[i]
    t0 = time.time()
    
    # Cho mỗi map tối đa 5 giây (bạn có thể tăng/giảm số này)
    res = solution.solve(map_state, timebound=120)
    t = time.time() - t0

    if res:
        print(f"Map {i:<4} | {'PASS':<10} | {res.gval:<10} | {t:.2f}s")
    else:
        print(f"Map {i:<4} | {'TIMEOUT':<10} | {'-':<10} | {t:.2f}s")

print("=" * 55)