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


class AplEnv(gym.Env):
    """ Class for simplified and fast APL on openAI gym interface """
    metadata = {'render.modes': ['human']}
    MINIMUM_ENV = False
    X_MAX = 499
    X_MIN = 0
    Y_MAX = 499
    Y_MIN = 0
    T_MAX = 100  # episode lenght. Not same as Config.TIME_MAX
    if not MINIMUM_ENV:
        TOP_CAMERA_X = 10
        TOP_CAMERA_Y = 10
    else:
        TOP_CAMERA_X = 2
        TOP_CAMERA_Y = 1
    OBS_SIZE_X = TOP_CAMERA_X
    OBS_SIZE_Y = TOP_CAMERA_Y + 1  # Extra column for sensors
    IMAGE_MULTIPLIER = 8
    ALTITUDES = np.array([0, 1, 2, 3, 4])
    MAX_DRONE_ALTITUDE = 3
    HEADINGS = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    NORMALISATION_FACTOR = 707.106781187
    drone = Drone()
    hiker = Hiker()
    CHECK_ALTITUDE = True
    viewer = None
    viewer_ego = None
    cells = None
    dronetrans = None
    hiker_trans = None
    observations = None
    number_step = 0
    HIKER_SAMPLE_LEFT = True
    # Adding memory to aovid repetition of x, y
    PENALISE_REPETITION = True
    visited_position = set()
    action = -2

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        # PPO2 only supports Discrete or Box
        # Building a box = camera + sensor
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.float64,
                                            shape=(self.OBS_SIZE_X,
                                                   self.OBS_SIZE_Y))
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'godiland_altitude_min0.npa')
        with open(filename, 'rb') as file_p:
            self.full_altitude_map = np.load(file_p)
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
        if not valid_drone_pos:
            done = True
        else:
            done = self._has_drone_arrived_hiker(self.drone.x, self.drone.y)
        if self.number_step == self.T_MAX:
            print("                                          MAX TIME REACHED")
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
        while not self._is_valid_drone_pos():
            self.drone.x, self.drone.y, self.drone.alt, self.drone.head = \
                self._get_drone_random_pos()
        self.drone.x_t_minus_1 = self.drone.x
        self.drone.y_t_minus_1 = self.drone.y
        # Random hiker position
        self.hiker.x, self.hiker.y = self._get_hiker_random_pos()
        while not self._is_valid_hiker_pos():
            self.hiker.x, self.hiker.y = self._get_hiker_random_pos()
        self.visited_position.clear()
        self.action = -2
        return self._get_observations(True)

    def render(self, mode='human', close=False):
        screen_width = self.Y_MAX
        screen_height = self.X_MAX
        cell_width = 20
        cell_height = 20
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, 'godiland500by500.png')
            background = rendering.Image(filename, 499, 499)
            self.viewer.add_geom(background)
            drone = rendering.FilledPolygon([(1, 1), (1, 10), (10, 5)])
            self.dronetrans = rendering.Transform()
            drone.add_attr(self.dronetrans)
            self.viewer.add_geom(drone)
            self.dronetrans.set_translation(self.drone.x, self.drone.y)
            drone.set_color(.8, .6, .3)
            hiker = rendering.FilledPolygon([(1, 1), (1, 10), (10, 10),
                                             (10, 1)])
            self.hiker_trans = rendering.Transform()
            hiker.add_attr(self.hiker_trans)
            self.viewer.add_geom(hiker)
            self.hiker_trans.set_translation(self.hiker.x, self.hiker.y)
            hiker.set_color(0.8, 0.3, 0.5)
        #if self.viewer_ego is None:
            #from gym.envs.classic_control import rendering
            #self.viewer_ego = rendering.Viewer(self.OBS_SIZE_Y * cell_width,
                                               #self.OBS_SIZE_X * cell_height)
            #self.cells = [[rendering.FilledPolygon(
                #[(y_cell * cell_width, x_cell * cell_height),
                 #((y_cell + 1) * cell_width, x_cell * cell_height),
                 #((y_cell + 1) * cell_width, (x_cell + 1) * cell_height),
                 #(y_cell * cell_width, (x_cell + 1) * cell_width)])
                           #for x_cell in range(self.OBS_SIZE_X)]
                          #for y_cell in range(self.OBS_SIZE_Y)]
            #[[self.viewer_ego.add_geom(self.cells[y_cell][x_cell])
              #for x_cell in range(self.OBS_SIZE_X)]
             #for y_cell in range(self.OBS_SIZE_Y)]
            #[[self.cells[y_cell][x_cell].set_color(rd.random(), rd.random(),
                                                   #rd.random())
              #for x_cell in range(self.OBS_SIZE_X)]
             #for y_cell in range(self.OBS_SIZE_Y)]
        from gym.envs.classic_control import rendering
        self.dronetrans.set_translation(self.drone.x, self.drone.y)
        self.hiker_trans.set_translation(self.hiker.x, self.hiker.y)
        #self._set_cells_color()
        point = rendering.Point()
        t = rendering.Transform()
        point.add_attr(t)
        self.viewer.add_geom(point)
        t.set_translation(self.drone.x, self.drone.y)
        point.set_color(.4, .1, .3)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')#, \
            #self.viewer_ego.render(return_rgb_array=mode == 'rgb_array')

    def seed(self, seed=None):
        return 1

    def _get_hiker_random_pos(self):
        """ Returns random position of the hiker """
        #if self.HIKER_SAMPLE_LEFT:
            #x_pos = rd.randint(240, 250)
        #else:
            #x_pos = rd.randint(395, 405)
        #y_pos = rd.randint(279, 479)
        #x_pos = rd.randint(0, self.X_MAX)
        #y_pos = rd.randint(0, self.X_MAX)
        #y_pos = rd.randint(439, 469)
        y_pos = self.drone.y
        x_pos = 309
        #y_pos = 400
        return x_pos, y_pos

    def _get_drone_random_pos(self):
        """ Returns random values for initial positions """
        #if self.HIKER_SAMPLE_LEFT:
            #x_pos = rd.randint(240, 250)
        #else:
            #x_pos = rd.randint(395, 405)
        #self.HIKER_SAMPLE_LEFT = not self.HIKER_SAMPLE_LEFT
        #y_pos = rd.randint(279, 479)
        #x_pos = rd.randint(0, self.X_MAX)
        #y_pos = rd.randint(0, self.X_MAX)
        y_pos = rd.randint(439, 469)
        x_pos = 296
        alt = np.random.choice(self.ALTITUDES)
        head = np.random.choice(self.HEADINGS)
        return x_pos, y_pos, 3, 1  # alt, head

    def _is_valid_drone_pos(self):
        """ Checks if the drone is inside boundaries or not crashed """
        if self.drone.x < self.X_MIN or \
                self.drone.x > self.X_MAX or \
                self.drone.y < self.Y_MIN or \
                self.drone.y > self.Y_MAX:
            print("                         OUTSIDE")
            return False
        if self.PENALISE_REPETITION:
            if self.action != -1:
                if (self.drone.x, self.drone.y) in self.visited_position:
                    print("                         TAILED")
                    return False
                self.visited_position.add((self.drone.x, self.drone.y))
        if not np.isin(self.drone.alt, self.ALTITUDES):
            return False
        if not np.isin(self.drone.head, self.HEADINGS):
            return False
        if self.drone.alt > self.MAX_DRONE_ALTITUDE:
            return False
        if self.CHECK_ALTITUDE:
            if self.drone.alt <= \
               self.full_altitude_map[self.drone.x][self.drone.y]:
                print("                         CRASH")
                return False
        return True

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
        #if action == 0: #  action 0 == NOOP in GA3C
        if action == 0:
            next_x += 1
        if action == 1:
            next_y += 1
        if action == 2:
            next_y += -1
        if action == 3:
            next_x += -1
        self.action = action
        return next_x, next_y, next_alt, next_head

    def _has_drone_arrived_hiker(self, x_pos, y_pos):
        """ Returns true if the dron is on top of the hiker """
        if x_pos == self.hiker.x and y_pos == self.hiker.y:
            return True
        return False

    def _reward(self, is_valid_pos):
        """ If the drone is not on a valid position return negative reward,
            else, negative distance to the hiker """
        if not is_valid_pos:
            return -1.
        if self._has_drone_arrived_hiker(self.drone.x, self.drone.y):
            print("DRONE MADE IT")
            return 1.
        approach = self._distance_to_hiker(self.drone.x_t_minus_1,
                                           self.drone.y_t_minus_1,
                                           normalise=True)\
            - self._distance_to_hiker(self.drone.x, self.drone.y,
                                      normalise=True)
        if approach > 0:
            return .1
        return -.1

    def _get_observations(self, valid_drone_pos):
        obs = np.zeros((self.OBS_SIZE_X, self.OBS_SIZE_Y), dtype=np.float64)
        if not self.MINIMUM_ENV:
            for x_cell in range(self.TOP_CAMERA_X):
                for y_cell in range(self.TOP_CAMERA_Y):
                    try:
                        value = self.full_altitude_map[
                            int(self.drone.x - self.OBS_SIZE_X / 2 + x_cell)][
                                int(self.drone.y - self.OBS_SIZE_Y / 2 +
                                    y_cell)]
                        obs[x_cell][y_cell] = self._image_normalise_altitude(value)
                    except IndexError:
                        # Outside boundaries
                        obs[x_cell][y_cell] = 0.
            # Hiker inside camera
            if self.hiker.x >= self.drone.x - self.TOP_CAMERA_X / 2 and \
               self.hiker.x < self.drone.x + self.TOP_CAMERA_X / 2 and \
               self.hiker.y >= self.drone.y - self.TOP_CAMERA_Y / 2 and \
               self.hiker.y < self.drone.y + self.TOP_CAMERA_Y / 2:
                obs[int(self.TOP_CAMERA_X / 2 + (self.hiker.x - self.drone.x))
                    ][int(self.TOP_CAMERA_Y / 2 + (self.hiker.y - self.drone.y))]\
                            = 255.
        # Heading from drone to hiker
        #obs[0][self.OBS_SIZE_Y - 1] = self._heading_to_hiker()
        # each axis distance
        obs[0][self.OBS_SIZE_Y - 1] = self._image_normalise_axis(
            self.drone.x - self.hiker.x, self.X_MAX)
        obs[1][self.OBS_SIZE_Y - 1] = self._image_normalise_axis(
            self.drone.y - self.hiker.y, self.Y_MAX)
        # Sensors for visited states around
        if self.PENALISE_REPETITION:
            if ((self.drone.x + 1, self.drone.y)) in self.visited_position:
                obs[2][self.OBS_SIZE_Y - 1] = 255
            if ((self.drone.x - 1, self.drone.y)) in self.visited_position:
                obs[3][self.OBS_SIZE_Y - 1] = 255
            if ((self.drone.x + 1, self.drone.y + 1)) in self.visited_position:
                obs[4][self.OBS_SIZE_Y - 1] = 255
            if ((self.drone.x, self.drone.y - 1)) in self.visited_position:
                obs[5][self.OBS_SIZE_Y - 1] = 255
        # Distance from drone to hiker in [0,256)
        #obs[1][self.OBS_SIZE_Y - 1] = self._distance_to_hiker(self.drone.x, self.drone.y) * 255
        # Extending the size of the observation
        obs = np.kron(obs, np.ones([self.IMAGE_MULTIPLIER,
                                    self.IMAGE_MULTIPLIER]))
        return obs

    @staticmethod
    def _image_normalise_axis(distance, max_axis):
        """ distance in [-max_axis, max_axis]
            return [0, 255]
        """
        return (distance + max_axis) / (2. * max_axis) * 255.

    def _image_normalise_altitude(self, alt):
        """ Normalises altitude
        - 255 / max(ALTITUDES) : 255 is reserved for hiker
        % + 1 : 0 is reserved for outside boundaries
        """
        return (alt / max(self.ALTITUDES)) * (
            254. - 255. / max(self.ALTITUDES)) + 1

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
                else:
                    color = (0.5, 0.5, 0.5)
                self.cells[width_cell][height_cell].set_color(color[0],
                                                              color[1],
                                                              color[2])
        self.cells[self.OBS_SIZE_Y - 1][1].set_color(
            0.0, 0.0, self.observations[1][self.OBS_SIZE_Y - 1] / 255)

