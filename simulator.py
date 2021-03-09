"""
Self Driving Car Simulator using PyGame module and NEAT Algorithm
"""
import sys
import math
import pygame
import neat

pygame.init()
pygame.display.set_caption("SELF DRIVING CAR SIMULATOR")
WINDOW_SIZE = 1280, 720
SCREEN = pygame.display.set_mode(WINDOW_SIZE, pygame.FULLSCREEN)
CAR_SIZE = 30, 51
CAR_CENTER = 195, 290
DELTA_DISTANCE = 30
DELTA_ANGLE = 10
WHITE_COLOR = (255, 255, 255, 255)
TRACK = pygame.image.load('track.png').convert_alpha()
TRACK_COPY = TRACK.copy()
FONT = pygame.font.SysFont("bahnschrift", 25)
CLOCK = pygame.time.Clock()
GENERATION = 0


def translate_point(point, angle, distance):
    """
    Get the new co-ordinates of a given point w.r.t an angle and distance from that point

    Args:
        center (tuple): A tuple of x co-ordinate and y co-ordinate
        angle (int): Angle of rotation of the vector
        distance (float): The distance by which the point needs
        to be translated (magnitude of the vector)

    Returns:
        tuple: Translated co-ordinates of the point
    """
    radians = math.radians(angle)
    return int(point[0] + distance * math.cos(radians)),\
        int(point[1] + distance * math.sin(radians))

class Car:
    """
    Implentation of the self driving car
    """
    def __init__(self):
        self.corners = []
        self.edge_points = []
        self.edge_distances = []
        self.travelled_distance = 0
        self.angle = 0
        self.car_center = CAR_CENTER
        self.car = pygame.image.load("car.png").convert_alpha()
        self.car = pygame.transform.scale(self.car, CAR_SIZE)
        self.crashed = False
        self.update_sensor_data()

    def display_car(self):
        """
        Rotate the car and the display it on the screen
        """
        rotated_car = pygame.transform.rotate(self.car, self.angle)
        rect = rotated_car.get_rect(center=self.car_center)
        SCREEN.blit(rotated_car, rect.topleft)

    def crash_check(self):
        """
        Check if any corner of the car goes out of the track
        Returns:
            Bool: Returns True if the car is alive
        """
        for corner in self.corners:
            if TRACK.get_at(corner) == WHITE_COLOR:
                return True
        return False

    def update_sensor_data(self):
        """
        Update the points on the edge of the track
        and the distances between the points and the center of the car
        """
        angles = [360 - self.angle, 90 - self.angle, 180 - self.angle]
        angles = [math.radians(i) for i in angles]
        edge_points = []
        edge_distances = []
        for angle in angles:
            distance = 0
            edge_x, edge_y = self.car_center
            while TRACK_COPY.get_at((edge_x, edge_y)) != WHITE_COLOR:
                edge_x = int(self.car_center[0] + distance * math.cos(angle))
                edge_y = int(self.car_center[1] + distance * math.sin(angle))
                distance += 1
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        self.edge_points = edge_points
        self.edge_distances = edge_distances


    def display_edge_points(self):
        """
        Display lines from center of the car to the edges on the  track
        """
        for point in self.edge_points:
            pygame.draw.line(SCREEN, (0, 255, 0), self.car_center, point)
            pygame.draw.circle(SCREEN, (0, 255, 0), point, 5)

    def update_position(self):
        """
        Update the new position of the car
        """
        self.car_center = translate_point(
            self.car_center, 90 - self.angle, DELTA_DISTANCE)
        self.travelled_distance += DELTA_DISTANCE
        dist = math.sqrt(CAR_SIZE[0]**2 + CAR_SIZE[1]**2)/2
        corners = []
        corners.append(translate_point(
            self.car_center, 60 - self.angle, dist))
        corners.append(translate_point(
            self.car_center, 120 - self.angle, dist))
        corners.append(translate_point(
            self.car_center, 240 - self.angle, dist))
        corners.append(translate_point(
            self.car_center, 300 - self.angle, dist))
        self.corners = corners


def run(genomes, config):
    """
    Runs the game for a specific generation of NEAT Algorithm
    """
    global GENERATION
    GENERATION += 1
    models = []
    cars = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        models.append(net)
        genome.fitness = 0
        cars.append(Car())

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        running_cars = 0

        SCREEN.blit(TRACK, (0, 0))

        for i, car in enumerate(cars):
            if not car.crashed:
                running_cars += 1
                output = models[i].activate(car.edge_distances)
                choice = output.index(max(output))
                if choice == 0:
                    car.angle += DELTA_ANGLE
                elif choice == 1:
                    car.angle -= DELTA_ANGLE
                car.update_position()
                car.display_car()
                car.crashed = car.crash_check()
                car.update_sensor_data()
                genomes[i][1].fitness += car.travelled_distance
                car.display_edge_points()

        if running_cars == 0:
            return
        msg = "Generation: {}, Running Cars: {}".format(GENERATION, running_cars)
        text = FONT.render(msg, True, (0, 0, 0))
        SCREEN.blit(text, (0, 0))
        pygame.display.update()
        CLOCK.tick(10)


neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 "config.txt")

population = neat.Population(neat_config)
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.run(run, 500)
