import cv2
import numpy as np
import heapq
from PIL import Image


def image_to_maze(image_path, threshold=250):
    """
    Converts an image to a binary maze where white pixels are passable and black pixels are obstacles.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_array = np.array(image)
    maze = (image_array > threshold).astype(int)  # Convert to binary maze based on threshold
    cv2.imshow('Graph with Best Path (A*)', image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return maze

def image_to_graph(image_path, threshold=170):
    """
    Converts an image to a graph representation where nodes are connected to their neighboring nodes with corresponding edge weights.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_array = np.array(image)
    maze = (image_array > threshold).astype(int)  # Convert to binary maze based on threshold
    # cv2.imshow('Graph with Best Path (A*)', image_array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    graph = {}
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 1:
                neighbors = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    x, y = i + dx, j + dy
                    if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] == 1:
                        neighbors.append(((x, y), 1))  # Assuming each edge has weight 1
                graph[(i, j)] = dict(neighbors)

    return graph

def dijkstra(graph, start, end):
    """Returns a dictionary containing the shortest distance from the start node to each node in the graph."""

    # Initialize distances dictionary with infinity for all nodes except start node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Priority queue to store nodes to visit next
    pq = [(0, start)]

    while pq:
        # Pop the node with the smallest distance from the priority queue
        current_distance, current_node = heapq.heappop(pq)

        # Skip if we've already found a shorter path to this node
        if current_distance > distances[current_node]:
            continue

        # If the current node is the end node, we can stop
        if current_node == end:
            break

        # Update distances to neighbors through current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            # If we find a shorter path to a neighbor, update its distance and add it to the priority queue
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

def astar(graph, start, end, heuristic):
    """Returns a dictionary containing the shortest distance from the start node to each node in the graph."""

    # Initialize distances dictionary with infinity for all nodes except start node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Priority queue to store nodes to visit next
    pq = [(0 + heuristic(start, end), start)]  # Initial priority queue with start node and its f_score

    while pq:
        # Pop the node with the smallest f_score from the priority queue
        current_f_score, current_node = heapq.heappop(pq)

        # If the current node is the end node, we can stop
        if current_node == end:
            break

        # Update distances to neighbors through current node
        for neighbor, weight in graph[current_node].items():
            distance = distances[current_node] + weight
            # If we find a shorter path to a neighbor, update its distance and add it to the priority queue
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance + heuristic(neighbor, end), neighbor))

    return distances

def manhattan_distance(node, goal):
    """Manhattan distance heuristic."""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def draw_path(image, path, color):
    """
    Draws the given path on the image.
    """
    for i in range(len(path) - 1):
        cv2.line(image, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), color, 2)

def draw_graph_with_path_(graph, shape, start, end, path, shortest_distances):
    """
    Draws the graph with the best path highlighted on an image with white background and green lines.
    """
    # Create an image with white background
    image = np.ones(shape, dtype=np.uint8)

    # Draw green lines for edges
    for node, neighbors in graph.items():
        x, y = node
        for neighbor, _ in neighbors.items():
            nx, ny = neighbor
            cv2.line(image, (y, x), (ny, nx), (0, 255, 0), 1)  # Draw green edge between nodes

    # Draw the best path in blue
    draw_path(image, path, (255, 0, 0))

    # Draw start and end points
    cv2.circle(image, (start[1], start[0]), 5, (0, 0, 255), -1)  # Draw start point in red
    cv2.circle(image, (end[1], end[0]), 5, (0, 255, 0), -1)  # Draw end point in green

    # Draw text showing the cost of the path
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'Cost: {shortest_distances[end]:.2f}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image

def draw_graph_with_path(graph, shape, start, end, path, shortest_distances):
    """
    Draws the graph with the best path highlighted on an image with black background and white walkable path.
    """
    # Create an image with black background
    image = np.zeros((*shape, 3), dtype=np.uint8)  # Black background

    # Draw white lines for walkable paths
    for node, neighbors in graph.items():
        x, y = node
        for neighbor, _ in neighbors.items():
            nx, ny = neighbor
            cv2.line(image, (y, x), (ny, nx), (255, 255, 255), 1)  # Draw white edge between nodes for walkable paths

    # Draw the best path in green
    for i in range(len(path) - 1):
        cv2.line(image, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), (0, 0, 255), 2)  # Draw green path

    # Draw start and end points
    cv2.circle(image, (start[1], start[0]), 5, (0, 255, 0), -1)  # Draw start point in red
    cv2.circle(image, (end[1], end[0]), 5, (255, 0, 0), -1)  # Draw end point in green

    # Draw text showing the cost of the path
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'Cost: {shortest_distances[end]:.2f}', (550, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image

# Example usage:
# image_path = r"C:\Users\agran\Pictures\Screenshots\Newfolder\555.png"
image_path = r"555.png"
image = cv2.imread(image_path)
height, width = image.shape[:2]

graph = image_to_graph(image_path)
#         Up/Down  Right/Left
start_node = (110, 70)
end_node = (440,500)

# Run Dijkstra's algorithm to find the best path
shortest_distances_dijkstra = dijkstra(graph, start_node, end_node)
best_path_dijkstra = [] if shortest_distances_dijkstra[end_node] == float('inf') else [end_node]
while best_path_dijkstra[-1] != start_node:
    current_node = best_path_dijkstra[-1]
    for neighbor, _ in graph[current_node].items():
        if shortest_distances_dijkstra[neighbor] + graph[current_node][neighbor] == shortest_distances_dijkstra[current_node]:
            best_path_dijkstra.append(neighbor)
            break
    else:
        break
best_path_dijkstra.reverse()
print("Cost of Dijkstra: ",shortest_distances_dijkstra[end_node])

# Run A* algorithm to find the best path
shortest_distances_astar = astar(graph, start_node, end_node, manhattan_distance)
best_path_astar = [] if shortest_distances_astar[end_node] == float('inf') else [end_node]
while best_path_astar[-1] != start_node:
    current_node = best_path_astar[-1]
    for neighbor, _ in graph[current_node].items():
        if shortest_distances_astar[neighbor] + graph[current_node][neighbor] == shortest_distances_astar[current_node]:
            best_path_astar.append(neighbor)
            break
    else:
        break

best_path_astar.reverse()
print("Cost of A*: ", shortest_distances_astar[end_node])
# Draw the graph with the best path highlighted for Dijkstra's algorithm
graph_image_with_path_dijkstra = draw_graph_with_path(graph, (height, width), start_node, end_node, best_path_dijkstra, shortest_distances_dijkstra)
cv2.imshow('Graph with Best Path (Dijkstra)', graph_image_with_path_dijkstra)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw the graph with the best path highlighted for A* algorithm
graph_image_with_path_astar = draw_graph_with_path(graph, (height, width), start_node, end_node, best_path_astar, shortest_distances_astar)
cv2.imshow('Graph with Best Path (A*)', graph_image_with_path_astar)
cv2.waitKey(0)
cv2.destroyAllWindows()

