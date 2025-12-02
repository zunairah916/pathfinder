import pygame
import heapq
import math
from collections import deque
import sys
import time

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
GRID_ROWS = 30
GRID_COLS = 40
FPS = 60
CELL_MARGIN = 1

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
WALL_COLOR = (40, 40, 40)
START_COLOR = (60, 180, 75)
GOAL_COLOR = (230, 50, 50)
OPEN_COLOR = (135, 206, 250)
CLOSED_COLOR = (70, 130, 180)
PATH_COLOR = (255, 223, 87)
GRID_LINE = (200, 200, 200)
TEXT_COLOR = (10, 10, 10)
BG_COLOR = (242, 242, 247)
CURRENT_COLOR = (255, 160, 122)

pygame.init()
FONT = pygame.font.SysFont("Arial", 14)
SMALL_FONT = pygame.font.SysFont("Arial", 12)
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Grid Pathfinding Visualizer — BFS / DFS / A*")
clock = pygame.time.Clock()

class Cell:
    def __init__(self, r, c):
        self.r = r
        self.c = c
        self.walkable = True
        self.is_start = False
        self.is_goal = False

    def pos(self):
        return (self.r, self.c)

def make_grid(rows, cols):
    return [[Cell(r, c) for c in range(cols)] for r in range(rows)]

grid = make_grid(GRID_ROWS, GRID_COLS)

start = grid[0][0]
start.is_start = True
goal = grid[GRID_ROWS - 1][GRID_COLS - 1]
goal.is_goal = True

usable_width = WINDOW_WIDTH - 240  
usable_height = WINDOW_HEIGHT - 20
cell_w = usable_width // GRID_COLS
cell_h = usable_height // GRID_ROWS
CELL_SIZE = min(cell_w, cell_h)
GRID_ORIGIN = (10, 10)

diagonals_allowed = False
speed_delay = 0.01  
running_algo = None  
last_algo_name = "None"
show_scores = False  

def in_bounds(r, c):
    return 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS

def neighbors(cell):
    r, c = cell.r, cell.c
    dirs_4 = [(1,0),(-1,0),(0,1),(0,-1)]
    dirs_diag = [(1,1),(1,-1),(-1,1),(-1,-1)]
    out = []
    for dr, dc in dirs_4:
        nr, nc = r+dr, c+dc
        if in_bounds(nr, nc) and grid[nr][nc].walkable:
            out.append((grid[nr][nc], 1.0))
    if diagonals_allowed:
        for dr, dc in dirs_diag:
            nr, nc = r+dr, c+dc
            if in_bounds(nr, nc) and grid[nr][nc].walkable:
                out.append((grid[nr][nc], math.sqrt(2)))
    return out

def manhattan(a, b):
    return abs(a.r - b.r) + abs(a.c - b.c)

def euclidean(a, b):
    return math.hypot(a.r - b.r, a.c - b.c)

def heuristic(a, b):
    return manhattan(a, b) if not diagonals_allowed else euclidean(a, b)

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def bfs_generator(start_cell, goal_cell):
    """Yields (open_set, closed_set, came_from, current, g_scores, extra) for visualization."""
    frontier = deque([start_cell])
    came_from = {}
    visited = {start_cell}
    g = {start_cell: 0}
    while frontier:
        current = frontier.popleft()
        yield set(frontier), set(visited), dict(came_from), current, dict(g), None
        if current is goal_cell:
            break
        for neigh, cost in neighbors(current):
            if neigh not in visited:
                visited.add(neigh)
                came_from[neigh] = current
                g[neigh] = g[current] + cost
                frontier.append(neigh)
                yield set(frontier), set(visited), dict(came_from), current, dict(g), None
    path = reconstruct_path(came_from, goal_cell) if goal_cell in came_from else []
    yield set(), set(visited), dict(came_from), None, dict(g), path

def dfs_generator(start_cell, goal_cell):
    stack = [start_cell]
    came_from = {}
    visited = set()
    g = {start_cell: 0}
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        yield set(stack), set(visited), dict(came_from), current, dict(g), None
        if current is goal_cell:
            break
        for neigh, cost in neighbors(current):
            if neigh not in visited:
                came_from[neigh] = current
                g[neigh] = g[current] + cost
                stack.append(neigh)
                yield set(stack), set(visited), dict(came_from), current, dict(g), None
    path = reconstruct_path(came_from, goal_cell) if goal_cell in came_from else []
    yield set(), set(visited), dict(came_from), None, dict(g), path

def a_star_generator(start_cell, goal_cell):
    open_heap = []
    count = 0
    g_score = {start_cell: 0}
    f_score = {start_cell: heuristic(start_cell, goal_cell)}
    heapq.heappush(open_heap, (f_score[start_cell], count, start_cell))
    open_set = {start_cell}
    came_from = {}
    closed_set = set()
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current not in open_set:
            continue
        open_set.discard(current)
        closed_set.add(current)
        yield set(open_set), set(closed_set), dict(came_from), current, dict(g_score), None
        if current is goal_cell:
            break
        for neigh, cost in neighbors(current):
            tentative_g = g_score.get(current, math.inf) + cost
            if tentative_g < g_score.get(neigh, math.inf):
                came_from[neigh] = current
                g_score[neigh] = tentative_g
                f = tentative_g + heuristic(neigh, goal_cell)
                if neigh not in open_set:
                    count += 1
                    heapq.heappush(open_heap, (f, count, neigh))
                    open_set.add(neigh)
                yield set(open_set), set(closed_set), dict(came_from), current, dict(g_score), None
    path = reconstruct_path(came_from, goal_cell) if goal_cell in came_from else []
    yield set(), set(closed_set), dict(came_from), None, dict(g_score), path

def cell_rect(cell):
    ox, oy = GRID_ORIGIN
    x = ox + cell.c * CELL_SIZE
    y = oy + cell.r * CELL_SIZE
    return pygame.Rect(x, y, CELL_SIZE - CELL_MARGIN, CELL_SIZE - CELL_MARGIN)

def draw_grid(open_set=None, closed_set=None, came_from=None, current=None, g_scores=None, final_path=None):
    screen.fill(BG_COLOR)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = grid[r][c]
            rect = cell_rect(cell)
            color = WHITE
            if not cell.walkable:
                color = WALL_COLOR
            elif cell.is_start:
                color = START_COLOR
            elif cell.is_goal:
                color = GOAL_COLOR
            pygame.draw.rect(screen, color, rect)

    if closed_set:
        for cell in closed_set:
            if cell.is_start or cell.is_goal:
                continue
            rect = cell_rect(cell)
            pygame.draw.rect(screen, CLOSED_COLOR, rect)
    if open_set:
        for cell in open_set:
            if cell.is_start or cell.is_goal:
                continue
            rect = cell_rect(cell)
            pygame.draw.rect(screen, OPEN_COLOR, rect)
    if current:
        rect = cell_rect(current)
        pygame.draw.rect(screen, CURRENT_COLOR, rect)

    if final_path:
        for cell in final_path:
            if cell.is_start or cell.is_goal:
                continue
            rect = cell_rect(cell)
            pygame.draw.rect(screen, PATH_COLOR, rect)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = grid[r][c]
            rect = cell_rect(cell)
            pygame.draw.rect(screen, GRID_LINE, rect, 1)
            if show_scores and g_scores and (cell in g_scores):
                s = f"{g_scores[cell]:.1f}"
                text = SMALL_FONT.render(s, True, TEXT_COLOR)
                screen.blit(text, (rect.x + 2, rect.y + 2))
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = grid[r][c]
            rect = cell_rect(cell)
            if cell.is_start:
                text = FONT.render("S", True, BLACK)
                screen.blit(text, (rect.centerx - 6, rect.centery - 9))
            elif cell.is_goal:
                text = FONT.render("G", True, BLACK)
                screen.blit(text, (rect.centerx - 6, rect.centery - 9))


    panel_x = GRID_ORIGIN[0] + GRID_COLS * CELL_SIZE + 10
    panel_rect = pygame.Rect(panel_x, 10, WINDOW_WIDTH - panel_x - 10, WINDOW_HEIGHT - 20)
    pygame.draw.rect(screen, WHITE, panel_rect)

    start_y = 20
    draw_text("Controls & Info", panel_x + 10, start_y, 18)
    start_y += 30
    draw_text(f"Algorithm: {last_algo_name}", panel_x + 10, start_y)
    start_y += 22
    draw_text("Mouse:", panel_x + 10, start_y)
    start_y += 18
    draw_text("Left-drag: Draw walls", panel_x + 14, start_y, size=12)
    start_y += 16
    draw_text("Right-click: Erase wall", panel_x + 14, start_y, size=12)
    start_y += 16
    draw_text("1: Set Start (click cell)", panel_x + 10, start_y, size=12)
    start_y += 16
    draw_text("2: Set Goal (click cell)", panel_x + 10, start_y, size=12)
    start_y += 18
    draw_text("Keys:", panel_x + 10, start_y)
    start_y += 18
    draw_text("B: BFS   D: DFS   A: A*   SPACE: Start", panel_x + 14, start_y, size=12)
    start_y += 16
    draw_text("C: Clear walls   R: Reset visited/path", panel_x + 14, start_y, size=12)
    start_y += 16
    draw_text("G: Toggle diagonals   +/- : Speed", panel_x + 14, start_y, size=12)
    start_y += 16
    draw_text("H: Toggle scores (g) display", panel_x + 14, start_y, size=12)
    start_y += 22
    draw_text(f"Diagonals: {'ON' if diagonals_allowed else 'OFF'}", panel_x + 10, start_y)
    start_y += 18
    draw_text(f"Delay: {speed_delay:.3f}s", panel_x + 10, start_y)
    start_y += 24
    draw_text("Legend:", panel_x + 10, start_y)
    start_y += 18
    draw_legend(panel_x + 14, start_y)
   
    pygame.display.flip()

def draw_text(text, x, y, size=14):
    f = pygame.font.SysFont("Arial", size)
    txt = f.render(text, True, TEXT_COLOR)
    screen.blit(txt, (x, y))

def draw_legend(x, y):
    items = [
        (START_COLOR, "Start"),
        (GOAL_COLOR, "Goal"),
        (WALL_COLOR, "Wall"),
        (OPEN_COLOR, "Open set"),
        (CLOSED_COLOR, "Closed set"),
        (CURRENT_COLOR, "Current node"),
        (PATH_COLOR, "Path"),
    ]
    yy = y
    for color, label in items:
        pygame.draw.rect(screen, color, pygame.Rect(x, yy, 18, 14))
        screen.blit(SMALL_FONT.render(label, True, TEXT_COLOR), (x + 24, yy))
        yy += 20


placing_mode = None  
mouse_down = False
last_mouse_btn = None

def cell_at_pixel(pos):
    ox, oy = GRID_ORIGIN
    x, y = pos
    x -= ox
    y -= oy
    if x < 0 or y < 0:
        return None
    c = x // CELL_SIZE
    r = y // CELL_SIZE
    if in_bounds(r, c):
        return grid[r][c]
    return None

def clear_walls():
    for row in grid:
        for cell in row:
            cell.walkable = True

def reset_visits():
    pass  
open_set_vis = set()
closed_set_vis = set()
came_from_vis = {}
current_vis = None
g_scores_vis = {}
path_vis = None

def start_algorithm(name):
    global running_algo, last_algo_name
    last_algo_name = name
    if name == "BFS":
        return bfs_generator(start, goal)
    if name == "DFS":
        return dfs_generator(start, goal)
    if name == "A*":
        return a_star_generator(start, goal)
    return None

def handle_events():
    global mouse_down, last_mouse_btn, placing_mode, start, goal, running_algo
    global diagonals_allowed, speed_delay, show_scores, running_algo, path_vis
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
            last_mouse_btn = event.button
            pos = pygame.mouse.get_pos()
            cell = cell_at_pixel(pos)
            if cell:
                if placing_mode == 'start':
                    start.is_start = False
                    cell.is_start = True
                    start = cell
                    placing_mode = None
                    running_algo = None
                    path_vis = None
                elif placing_mode == 'goal':
                    goal.is_goal = False
                    cell.is_goal = True
                    goal = cell
                    placing_mode = None
                    running_algo = None
                    path_vis = None
                else:
                    if event.button == 1:
                        if cell is not start and cell is not goal:
                            cell.walkable = False
                            running_algo = None
                            path_vis = None
                    elif event.button == 3:
                       
                        cell.walkable = True
                        running_algo = None
                        path_vis = None
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
            last_mouse_btn = None
        elif event.type == pygame.MOUSEMOTION and mouse_down:
            pos = pygame.mouse.get_pos()
            cell = cell_at_pixel(pos)
            if cell and placing_mode is None:
                if last_mouse_btn == 1:
                    if cell is not start and cell is not goal:
                        cell.walkable = False
                        running_algo = None
                        path_vis = None
                elif last_mouse_btn == 3:
                    cell.walkable = True
                    running_algo = None
                    path_vis = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                placing_mode = 'start'
            elif event.key == pygame.K_2:
                placing_mode = 'goal'
            elif event.key == pygame.K_c:
                clear_walls()
                running_algo = None
                path_vis = None
            elif event.key == pygame.K_r:
             
                running_algo = None
                path_vis = None
            elif event.key == pygame.K_b:
                running_algo = start_algorithm("BFS")
            elif event.key == pygame.K_d:
                running_algo = start_algorithm("DFS")
            elif event.key == pygame.K_a:
                running_algo = start_algorithm("A*")
            elif event.key == pygame.K_SPACE:
                if running_algo is None:
                    running_algo = start_algorithm(last_algo_name) if last_algo_name in ("BFS","DFS","A*") else None
            elif event.key == pygame.K_g:

                toggle_diag()
            elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                speed_up()
            elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                speed_down()
            elif event.key == pygame.K_h:
                toggle_scores()

def toggle_diag():
    global diagonals_allowed, running_algo, path_vis
    diagonals_allowed = not diagonals_allowed
    running_algo = None
    path_vis = None

def speed_up():
    global speed_delay
    speed_delay = max(0.0, speed_delay - 0.005)

def speed_down():
    global speed_delay
    speed_delay += 0.01

def toggle_scores():
    global show_scores
    show_scores = not show_scores


def main_loop():
    global running_algo, open_set_vis, closed_set_vis, came_from_vis, current_vis, g_scores_vis, path_vis
    step_time = 0
    while True:
        dt = clock.tick(FPS) / 1000.0
        handle_events()
        if running_algo is not None:
            try:
            
                result = next(running_algo)
                open_set_vis, closed_set_vis, came_from_vis, current_vis, g_scores_vis, maybe_path = result
                if maybe_path is not None:
                    path_vis = maybe_path
                    running_algo = None
                if speed_delay > 0:
                    time.sleep(speed_delay)
            except StopIteration:
                running_algo = None
        draw_grid(open_set_vis, closed_set_vis, came_from_vis, current_vis, g_scores_vis, path_vis)

if __name__ == "__main__":
    main_loop()
