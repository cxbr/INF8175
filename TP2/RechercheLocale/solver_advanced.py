from schedule import Schedule
import random
import time
import math


TIME_LIMIT = 297  # Allowing 3 seconds buffer
CONVERGENCE_THRESHOLD = 10


def solve(schedule: Schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    loops_without_improvement = 0
    start_time = time.time()
    total_conflicts_map = get_map_of_conflicts(schedule)
    initial_state = generate_initial_solution(schedule)
    current_state, best_solution, best_score = initial_state, None, math.inf

    while not is_finished(loops_without_improvement, start_time):
        conflicts_map = get_conflicts_per_course_map(
            current_state, total_conflicts_map)
        neighbors, course = get_state_neighbors(current_state, conflicts_map)
        valid_neighbors = get_valid_neighbors(
            neighbors, course, total_conflicts_map, current_state)
        next_state = get_random_neighbor(valid_neighbors)

        if (conflicts := get_total_nb_conflicts(next_state, total_conflicts_map)) == 0:
            new_score = get_state_score(next_state)
            if new_score < best_score:
                best_solution, best_score = next_state, new_score
                loops_without_improvement = 0
            else:
                loops_without_improvement += 1
            current_state = generate_initial_solution(schedule)
            continue

        current_state = next_state

    return best_solution


def generate_initial_solution(schedule: Schedule) -> dict:
    """
    Initializes state with all courses assigned to the same time slot.
    """
    return {course: 1 for course in schedule.course_list}


def is_finished(loops_without_improvement: int, start_time: float) -> bool:
    """
    Checks if the process should stop based on time and improvement loops.
    """
    return (time.time() - start_time) >= TIME_LIMIT or loops_without_improvement > CONVERGENCE_THRESHOLD


def get_map_of_conflicts(schedule: Schedule) -> dict:
    """
    Creates a conflict map for courses to avoid re-evaluation.
    """
    conflicts_map = {}
    for course1, course2 in schedule.conflict_list():
        conflicts_map.setdefault(course1, set()).add(course2)
        conflicts_map.setdefault(course2, set()).add(course1)
    return conflicts_map


def get_nb_conflicts_of_course(course: str, state: dict, conflicts_map: dict) -> int:
    """
    Counts conflicts involving a specific course in a state.
    """
    return sum(1 for conflict in conflicts_map[course] if state.get(conflict) == state[course])


def get_total_nb_conflicts(state: dict, conflicts_map: dict) -> int:
    """
    Calculates total conflicts for a given state.
    """
    return sum(1 for course, time in state.items() if course in conflicts_map
               for conflict in conflicts_map[course] if state.get(conflict) == time)


def get_conflicts_per_course_map(state: dict, conflicts_map: dict) -> dict:
    """
    Maps each course to its number of conflicts.
    """
    return {course: get_nb_conflicts_of_course(course, state, conflicts_map) for course in state}


def get_state_neighbors(state: dict, conflicts_map: dict) -> tuple:
    """
    Generates neighbors by moving the course with max conflicts to different slots.
    """
    max_conflicts = max(conflicts_map.values())
    courses_with_max_conflicts = [
        course for course, conflicts in conflicts_map.items() if conflicts == max_conflicts]
    selected_course = random.choice(courses_with_max_conflicts)
    time_slots = set(state.values())
    neighbors = [dict(state, **{selected_course: time})
                 for time in time_slots if time != state[selected_course]]
    return neighbors, selected_course


def get_valid_neighbors(all_neighbors: list, course: str, conflicts_map: dict, state: dict) -> list:
    """
    Filters neighbors to those that reduce conflicts or adds a new time slot if none do.
    """
    valid_neighbors = [neighbor for neighbor in all_neighbors
                       if get_nb_conflicts_of_course(course, neighbor, conflicts_map) == 0]
    if not valid_neighbors:
        neighbor = dict(state, **{course: max(state.values()) + 1})
        valid_neighbors.append(neighbor)
    return valid_neighbors


def get_random_neighbor(valid_neighbors: list) -> dict:
    """
    Chooses a random neighbor from valid neighbors.
    """
    return random.choice(valid_neighbors)


def get_state_score(state: dict) -> int:
    """
    Counts unique time slots used in a state; lower is better.
    """
    return len(set(state.values()))
