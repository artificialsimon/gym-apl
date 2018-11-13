import os
import gym
import math
import numpy as np
import random as rd
from gym import error, spaces, utils
from gym.utils import seeding


class Drone(object):
    x = 0
    y = 0
    alt = 0
    head = 0
    x_t_minus_1 = 0
    y_t_minus_1 = 0


class Hiker(object):
    x = 0
    y = 0


objects_code = {'mountain ridge': 0, 'trail': 1, 'shore bank': 2,
                'flight tower': 3, 'cabin': 4, 'stripped road': 5,
                'solo tent': 6, 'runway': 7, 'white Jeep': 8,
                'water': 9, 'pine trees': 10, 'bush': 11,
                'active campfire ring': 12, 'firewatch tower': 13,
                'bushes': 14, 'unstripped road': 15, 'pine tree': 16,
                'blue Jeep': 17, 'grass': 18, 'family tent': 19,
                'small hill': 20, 'box canyon': 21,
                'inactive campfire ring': 22, 'large hill': 23}

class AplDropEnv(gym.Env):
    """ Class for simplified and fast APL on openAI gym interface for dropping
        payload """
    metadata = {'render.modes': ['human']}
    X_MAX = 499
    X_MIN = 0
    Y_MAX = 499
    Y_MIN = 0
    TOP_CAMERA_X = 20
    TOP_CAMERA_Y = 20
    ALTITUDES = np.array([0, 1, 2, 3, 4])
    MAX_DRONE_ALTITUDE = 3
    HEADINGS = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    NORMALISATION_FACTOR = math.sqrt(pow(TOP_CAMERA_X / 2, 2) +
                                     pow(TOP_CAMERA_Y / 2, 2))
    drone = Drone()
    hiker = Hiker()
    OBS_SIZE_X = TOP_CAMERA_X
    OBS_SIZE_Y = TOP_CAMERA_Y * 2 + 1  # Extra column for sensors
    CHECK_ALTITUDE = False
    viewer = None
    viewer_ego = None
    cells = None
    dronetrans = None
    hiker_trans = None
    observations = np.zeros((OBS_SIZE_X, OBS_SIZE_Y), dtype=np.int16)
    fix_map_around_hiker = np.zeros((OBS_SIZE_X, OBS_SIZE_Y), dtype=np.int16)
    dropped = False
    dropped_x = 0
    dropped_y = 0
    DROP_DISTANCE_FACTOR = .0
    number_step = 0

    def __init__(self):
        self.action_space = spaces.Discrete(6)
        # PPO2 only supports Discrete or Box
        # Building a box = camera with altitude + camera with objects + sensor
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8,
                                            shape=(self.OBS_SIZE_X,
                                                   self.OBS_SIZE_Y))
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'godiland_altitude_min0.npa')
        with open(filename, 'rb') as file_p:
            self.full_altitude_map = np.load(file_p)
            file_p.close()
        filename = os.path.join(dirname, 'godiland_objects.npa')
        with open(filename, 'rb') as file_p:
            self.full_objects_map = np.load(file_p)
            file_p.close()
        rd.seed()
        self.reset()

    def step(self, action):
        """ Takes action and returns next state, reward, done flag and info """
        done = False
        self.number_step += 1
        self.drone.x_t_minus_1 = self.drone.x
        self.drone.y_t_minus_1 = self.drone.y
        next_x, next_y, next_alt, next_head = self._one_step(action)
        self.drone.x = next_x
        self.drone.y = next_y
        self.drone.alt = next_alt
        self.drone.head = next_head
        valid_drone_pos = self._is_valid_drone_pos()
        if not valid_drone_pos or self.dropped or \
           self.number_step == 50:
            done = True
        self.observations = self._get_observations(valid_drone_pos)
        reward = self._reward(valid_drone_pos)
        info = {}
        #print(action, reward, self.observations)
        return self.observations, reward, done, info

    def reset(self):
        """ Sets initial state for the env """
        self.number_step = 0
        # Random drone position
        self.drone.x, self.drone.y, self.drone.alt, self.drone.head = \
            self._get_drone_random_pos()
        while not self._is_valid_drone_pos_absoulte():
            self.drone.x, self.drone.y, self.drone.alt, self.drone.head = \
                self._get_drone_random_pos()
        self.fix_map_around_hiker = self._get_map_around(self.drone.x,
                                                         self.drone.y)
        # positioning drone inside camera
        self.drone.x = int(self.TOP_CAMERA_X / 2)
        self.drone.y = int(self.TOP_CAMERA_Y / 2)
        # hiker in same relative pos as drone
        self.hiker.x = self.drone.x
        self.hiker.y = self.drone.y
        self.dropped = False
        return self._get_observations(True)

    def render(self, mode='human', close=False):
        screen_width = self.Y_MAX
        screen_height = self.X_MAX
        cell_width = 20
        cell_height = 20
        if self.viewer_ego is None:
            from gym.envs.classic_control import rendering
            self.viewer_ego = rendering.Viewer(self.OBS_SIZE_Y * cell_width,
                                               self.OBS_SIZE_X * cell_height)
            self.cells = [[rendering.FilledPolygon(
                [(y_cell * cell_width, x_cell * cell_height),
                 ((y_cell + 1) * cell_width, x_cell * cell_height),
                 ((y_cell + 1) * cell_width, (x_cell + 1) * cell_height),
                 (y_cell * cell_width, (x_cell + 1) * cell_width)])
                           for x_cell in range(self.OBS_SIZE_X)]
                          for y_cell in range(self.OBS_SIZE_Y)]
            [[self.viewer_ego.add_geom(self.cells[y_cell][x_cell])
              for x_cell in range(self.OBS_SIZE_X)]
             for y_cell in range(self.OBS_SIZE_Y)]
            [[self.cells[y_cell][x_cell].set_color(rd.random(), rd.random(),
                                                   rd.random())
              for x_cell in range(self.OBS_SIZE_X)]
             for y_cell in range(self.OBS_SIZE_Y)]
        from gym.envs.classic_control import rendering
        self._set_cells_color()
        return self.viewer_ego.render(return_rgb_array=mode == 'rgb_array')

    def seed(self, seed=None):
        return 1

    def _set_cells_color(self):
        for width_cell in range(self.OBS_SIZE_Y):
            for height_cell in range(self.OBS_SIZE_X):
                if self.observations[height_cell][width_cell] == -1:
                    color = (0.0, 0.0, 0.0)
                elif self.observations[height_cell][width_cell] == 0:
                    color = (0.0, 0.45, 0.14)
                elif self.observations[height_cell][width_cell] == 1:
                    color = (0.0, 0.72, 0.22)
                elif self.observations[height_cell][width_cell] == 2:
                    color = (0.0, 0.87, 0.27)
                elif self.observations[height_cell][width_cell] == 3:
                    color = (0.47, 0.99, 0.63)
                elif self.observations[height_cell][width_cell] == 4:
                    color = (0.65, 0.99, 0.76)
                elif self.observations[height_cell][width_cell] == 5:
                    color = (1.0, 0.0, 0.0)
                elif self.observations[height_cell][width_cell] == 6:
                    color = (1.0, 0.2, 0.2)
                elif self.observations[height_cell][width_cell] == 7:
                    color = (1.0, 0.5, 0.5)
                elif self.observations[height_cell][width_cell] == 8:
                    color = (1.0, 0.7, 0.7)
                elif self.observations[height_cell][width_cell] == 255:
                    color = (0.3, 0.7, 0.2)
                else:
                    color = (0.5, self.observations[height_cell][width_cell] / 18., 0.5)
                self.cells[width_cell][height_cell].set_color(color[0],
                                                              color[1],
                                                              color[2])
        self.cells[self.OBS_SIZE_Y - 1][1].set_color(
            0.0, 0.0, self.observations[1][self.OBS_SIZE_Y - 1] / 255)

    def _get_drone_random_pos(self):
        """ Returns random values for initial positions """
        x_pos = rd.randint(200, 250)
        y_pos = rd.randint(200, 250)
        #x_pos = rd.randint(0, self.X_MAX)
        #y_pos = rd.randint(0, self.X_MAX)
        x_pos = 100
        y_pos = 100
        alt = np.random.choice(self.ALTITUDES)
        head = np.random.choice(self.HEADINGS)
        return x_pos, y_pos, 3, 1  # alt, head

    def _is_valid_drone_pos_absoulte(self):
        """ Checks if the drone is inside absolute boundaries or not crashed """
        if self.drone.x < self.X_MIN or \
                self.drone.x > self.X_MAX or \
                self.drone.y < self.Y_MIN or \
                self.drone.y > self.Y_MAX:
            return False
        if not np.isin(self.drone.alt, self.ALTITUDES):
            return False
        if not np.isin(self.drone.head, self.HEADINGS):
            return False
        if self.drone.alt > self.MAX_DRONE_ALTITUDE:
            return False
        if self.drone.alt <= \
           self.full_altitude_map[self.drone.x][self.drone.y]:
            return False
        return True

    def _is_valid_drone_pos(self):
        """ Checks if the drone is inside relative boundaries or not crashed """
        if self.drone.x <  0 or \
                self.drone.x >= self.TOP_CAMERA_X or \
                self.drone.y < 0 or \
                self.drone.y >= self.TOP_CAMERA_Y:
            return False
        if not np.isin(self.drone.alt, self.ALTITUDES):
            return False
        if not np.isin(self.drone.head, self.HEADINGS):
            return False
        if self.drone.alt > self.MAX_DRONE_ALTITUDE:
            return False
        if self.CHECK_ALTITUDE:
            if self.drone.alt <= \
               self.fix_map_around_hiker[self.drone.x][self.drone.y]:
                return False
        return True

    def _get_hiker_random_pos(self):
        """ Returns random position of the hiker """
        x_pos = rd.randint(300, 350)
        y_pos = rd.randint(300, 350)
        #x_pos = rd.randint(0, self.X_MAX)
        #y_pos = rd.randint(0, self.X_MAX)
        #x_pos = 330
        #y_pos = 330
        return x_pos, y_pos

    def _is_valid_hiker_pos(self):
        """ Checks if the hiker is in inside boundaries """
        if self.hiker.x < self.X_MIN or \
                self.hiker.x > self.X_MAX or \
                self.hiker.y < self.Y_MIN or \
                self.hiker.y > self.Y_MAX:
            return False
        return True

    def _one_step(self, action):
        """ Dynamics for actions. Follows APL """
        next_x = self.drone.x
        next_y = self.drone.y
        next_alt = self.drone.alt
        next_head = self.drone.head
        #print(action)
        #if action == 0:  # Action == 0 is no op for GA3C!
        if action == 1:
            next_x += 1
        if action == 2:
            next_y += 1
        if action == 3:
            next_y += -1
        if action == 4:
            next_x += -1
        if action == 5:
            self.dropped = True
            # TODO apl dynamics for dropping final place
            self.dropped_x = self.drone.x
            self.dropped_y = self.drone.y
        return next_x, next_y, next_alt, next_head

    def _reward(self, is_valid_pos):
        """ If the drone is not on a valid position return negative reward,
            else, negative distance to the hiker """
        reward = .0
        if not is_valid_pos:
            return -1
        if self.dropped:
            drop_reward = .0
            distance = self._distance_to_hiker(self.drone.x, self.drone.y,
                                               normalise=True) - 1.
            # TODO add dropping on object probabilities
            if self.fix_map_around_hiker[self.dropped_x][self.dropped_y] <= 0:
                drop_reward = 1.
            reward += drop_reward + self.DROP_DISTANCE_FACTOR * distance
        return reward

    def _get_observations(self, valid_drone_pos):
        obs = np.copy(self.fix_map_around_hiker)
        # drone position
        if valid_drone_pos:
            obs[self.drone.x][self.drone.y] = 9
        if self.dropped:
            obs[self.dropped_x][self.dropped_y] = 255
        #obs[0][self.OBS_SIZE_Y - 1] = self.drone.alt
        return obs

    def _get_map_around(self, x_pos, y_pos):
        """ returns the map around the x and y pos, first half is altitude,
            second is  objects code """
        obs = np.zeros((self.OBS_SIZE_X, self.OBS_SIZE_Y), dtype=np.uint8)
        for x_cell in range(self.TOP_CAMERA_X):
            for y_cell in range(self.TOP_CAMERA_Y):
                try:
                    alt = self.full_altitude_map[
                        int(x_pos - self.TOP_CAMERA_X / 2 + x_cell)][
                            int(y_pos - self.TOP_CAMERA_Y / 2 + y_cell)]
                    obs[x_cell][y_cell] = alt
                    obj = self.full_objects_map[
                        int(x_pos - self.TOP_CAMERA_X / 2 + x_cell)][
                            int(y_pos - self.TOP_CAMERA_Y / 2 + y_cell)]
                    obs[x_cell][y_cell + self.TOP_CAMERA_Y] = obj
                except IndexError:
                    obs[x_cell][y_cell] = -1
                    obs[x_cell][y_cell + self.TOP_CAMERA_Y] = -1
        return obs


    def _distance_to_hiker(self, drone_x, drone_y, normalise=True):
        dist = math.sqrt(pow(self.hiker.x - drone_x, 2) +
                         pow(self.hiker.y - drone_y, 2))
        if normalise:
            dist = dist / self.NORMALISATION_FACTOR
        return dist

    def _heading_to_hiker2(self):
        """  NE, SE, SW, NW
        APL coordinate system is (x, -y) in [0, 500), [0, 500)
        atan2(y, x) """
        radians = math.atan2(self.drone.x - self.hiker.x,
                             self.drone.y - self.hiker.y)
        # Compensating and binning for APL
        if 0 < radians <= math.pi / 2.0:
            return 1
        if math.pi / 2.0 < radians <= math.pi:
            return 2
        if -math.pi < radians <= -math.pi / 2.0:
            return 3
        if -math.pi / 2.0 < radians <= 0:
            return 4
        return 0

    def _heading_to_hiker(self):
        """ 1 N, 2 NE, 3 E, 4 SE, 5 S, 6 SW, 7 W, 8 NW
        APL coordinate system is (x, -y) in [0, 500), [0, 500)
        atan2(y, x) """
        radians = math.atan2(self.drone.x - self.hiker.x,
                             self.drone.y - self.hiker.y)
        # Compensating and binning for APL
        if radians <= 1. / 8. * math.pi and radians > -1. / 8. * math.pi:
            return 3
        if radians <= -1. / 8. * math.pi and radians > -3. / 8. * math.pi:
            return 2
        if radians <= -3. / 8. * math.pi and radians > -5. / 8. * math.pi:
            return 1
        if radians <= -5. / 8. * math.pi and radians > -7. / 8. * math.pi:
            return 8
        if radians <= 3. / 8. * math.pi and radians > 1. / 8. * math.pi:
            return 4
        if radians <= 5. / 8. * math.pi and radians > 3. / 8. * math.pi:
            return 5
        if radians <= 7. / 8. * math.pi and radians > 5. / 8. * math.pi:
            return 6
        if radians <= -7. / 8. * math.pi or radians > 7. / 8. * math.pi:
            return 7
        return 0
