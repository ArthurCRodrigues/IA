import time
import heapq
import random
import math

# ==========================================
# 1. CORE GAME LOGIC (The 8-Puzzle)
# ==========================================

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)  # 0 represents the empty tile


class Node:
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g  # Cost from start (depth)
        self.h = h  # Heuristic cost
        self.f = g + h  # Total cost

    # Comparison for Priority Queue (needed for A* and Greedy)
    def __lt__(self, other):
        return self.f < other.f


def get_neighbors(state):
    """Generates valid moves (Up, Down, Left, Right)"""
    neighbors = []
    # Convert tuple to list to manipulate
    grid = list(state)
    zero_idx = grid.index(0)
    row, col = divmod(zero_idx, 3)

    # Possible moves: (row_change, col_change, direction_name)
    moves = [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]

    for r_move, c_move, action in moves:
        new_row, new_col = row + r_move, col + c_move

        # Check boundaries
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            # Swap
            new_grid = grid[:]
            new_grid[zero_idx], new_grid[new_idx] = new_grid[new_idx], new_grid[zero_idx]
            neighbors.append((tuple(new_grid), action))

    return neighbors


def print_board(state):
    """Helper to pretty print the board"""
    for i in range(0, 9, 3):
        print(f" {state[i]} | {state[i + 1]} | {state[i + 2]} ")
    print("-----------")


# ==========================================
# 2. HEURISTICS (For A* and Greedy)
# ==========================================

def h_misplaced(state):
    """Count tiles in the wrong position (excluding the blank space)"""
    count = 0
    for i in range(9):
        if state[i] != 0 and state[i] != GOAL_STATE[i]:
            count += 1
    return count


def h_manhattan(state):
    """Sum of distances of tiles to their goal positions"""
    distance = 0
    for i in range(9):
        tile = state[i]
        if tile != 0:
            # Current (row, col)
            current_row, current_col = divmod(i, 3)
            # Goal (row, col) - value 1 is at index 0, value 2 at index 1...
            target_idx = tile - 1
            target_row, target_col = divmod(target_idx, 3)
            distance += abs(current_row - target_row) + abs(current_col - target_col)
    return distance


def h_euclidean(state):
    """Straight-line distance (Hypotenuse)"""
    distance = 0
    for i in range(9):
        tile = state[i]
        if tile != 0:
            current_row, current_col = divmod(i, 3)
            target_idx = tile - 1
            target_row, target_col = divmod(target_idx, 3)
            # sqrt((x2-x1)^2 + (y2-y1)^2)
            distance += math.sqrt((current_row - target_row) ** 2 + (current_col - target_col) ** 2)
    return distance


# ==========================================
# 3. SEARCH ALGORITHMS
# ==========================================

def solve(start_state, method, heuristic_func=None):
    """
    Generic solver that handles BFS, Greedy, and A*.
    method: 'BFS', 'GREEDY', 'ASTAR'
    """
    start_time = time.time()

    # Setup initial node
    start_node = Node(start_state, g=0, h=0)
    if method in ['ASTAR', 'GREEDY'] and heuristic_func:
        start_node.h = heuristic_func(start_state)
        # For Greedy, f = h. For A*, f = g + h
        start_node.f = start_node.h if method == 'GREEDY' else (start_node.g + start_node.h)

    # Data structures
    frontier = []  # Priority Queue (Heap) or List
    explored = set()
    nodes_generated = 0

    if method == 'BFS':
        frontier.append(start_node)  # Standard list acting as Queue
    else:
        heapq.heappush(frontier, start_node)  # Priority Queue

    explored.add(start_state)

    while frontier:
        # Pop next node
        if method == 'BFS':
            current_node = frontier.pop(0)  # FIFO
        else:
            current_node = heapq.heappop(frontier)

        # Check Goal
        if current_node.state == GOAL_STATE:
            end_time = time.time()
            return reconstruct_path(current_node, nodes_generated, end_time - start_time)

        # Expand Neighbors
        for neighbor_state, action in get_neighbors(current_node.state):
            if neighbor_state not in explored:
                nodes_generated += 1
                new_g = current_node.g + 1
                new_h = 0

                # Calculate Heuristics if needed
                if method != 'BFS' and heuristic_func:
                    new_h = heuristic_func(neighbor_state)

                # Create new node
                new_node = Node(neighbor_state, parent=current_node, action=action, g=new_g, h=new_h)

                # Calculate Priority (f)
                if method == 'GREEDY':
                    new_node.f = new_h
                elif method == 'ASTAR':
                    new_node.f = new_g + new_h

                explored.add(neighbor_state)

                if method == 'BFS':
                    frontier.append(new_node)
                else:
                    heapq.heappush(frontier, new_node)

    return None  # No solution found


def reconstruct_path(node, nodes_count, time_taken):
    path = []
    while node.parent:
        path.append(node.action)
        node = node.parent
    path.reverse()
    return {
        "path": path,
        "depth": len(path),
        "nodes_visited": nodes_count,
        "time": time_taken
    }


# ==========================================
# 4. UTILITIES (Setup & Validation)
# ==========================================

def generate_random_solvable_board(steps=20):
    """Start from goal and shuffle backwards to guarantee solvability."""
    current = GOAL_STATE
    for _ in range(steps):
        neighbors = get_neighbors(current)
        current, _ = random.choice(neighbors)
    return current


def get_user_board():
    print("\nEnter the board row by row (use 0 for empty space).")
    print("Example: 1 2 3 (enter), 4 5 6 (enter), 7 8 0 (enter)")
    board = []
    try:
        for i in range(3):
            row = list(map(int, input(f"Row {i + 1}: ").split()))
            board.extend(row)
        return tuple(board)
    except:
        print("Invalid input. Using random board.")
        return generate_random_solvable_board()


# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================

def main():
    print("=== 8-Puzzle Solver (BFS, A*, Greedy) ===")

    # 1. Select State
    choice = input("\n1. Generate Random Board\n2. Enter Board Manually\nChoose (1/2): ")
    if choice == '2':
        start_state = get_user_board()
    else:
        difficulty = int(input("Enter shuffle depth (suggested 15-25 for speed): "))
        start_state = generate_random_solvable_board(difficulty)

    print("\nInitial State:")
    print_board(start_state)

    while True:
        print("\n--- Select Algorithm ---")
        print("1. Breadth-First Search (BFS)")
        print("2. Greedy Search (Heuristic: Misplaced Tiles)")
        print("3. A* (Heuristic: Misplaced Tiles)")
        print("4. A* (Heuristic: Manhattan Distance)")
        print("5. A* (Heuristic: Euclidean Distance)")
        print("0. Exit")

        alg = input("Choice: ")

        result = None
        method_name = ""

        if alg == '0':
            break

        elif alg == '1':
            method_name = "BFS"
            print("Running BFS... (this might take a while for deep solutions)")
            result = solve(start_state, 'BFS')

        elif alg == '2':
            method_name = "Greedy (Misplaced)"
            result = solve(start_state, 'GREEDY', h_misplaced)

        elif alg == '3':
            method_name = "A* (Misplaced)"
            result = solve(start_state, 'ASTAR', h_misplaced)

        elif alg == '4':
            method_name = "A* (Manhattan)"
            result = solve(start_state, 'ASTAR', h_manhattan)

        elif alg == '5':
            method_name = "A* (Euclidean)"
            result = solve(start_state, 'ASTAR', h_euclidean)

        # Output Results
        if result:
            print(f"\n--- Results for {method_name} ---")
            print(f"Time Taken: {result['time']:.4f} seconds")
            print(f"Nodes Visited: {result['nodes_visited']}")
            print(f"Solution Depth: {result['depth']}")
            print(f"Path: {result['path']}")
        else:
            print("No solution found (or timeout).")


if __name__ == "__main__":
    main()