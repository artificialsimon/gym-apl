import os
import gym
import math
import time
import numpy as np
import random as rd
from gym import error, spaces, utils
from gym.utils import seeding
#from skimage.draw import circle_perimeter
import skimage.draw as draw


class Drone(object):
    actual_x = 0
    actual_y = 0
    actual_alt = 0
    actual_head = 0
    prev_x = 0
    prev_y = 0
    prev_alt = 0
    prev_head = 0
    dropped = False
    payload_x = 0
    payload_y = 0
    payload_status = None


class Hiker(object):
    x_pos = 0
    y_pos = 0
    alt = 0
    required_payload = {}


class Payload:
    """Drone payloads type"""
    payloads = ["EMPTY", "food", "medicine", "communications"]


class PayloadStatus:
    """Drone payloads type"""
    status = ["OK", "OK_STUCK", "OK_SUNK", "DAMAGED",
              "DAMAGED_STUCK", "DAMAGED_SUNK"]


COLOUR_LEVEL = [(0, 0, 0), (10, 10, 10), (20, 20, 20), (30, 30, 30)]
DRONE_COLOUR = [178, 34, 34]
HIKER_COLOUR = [0, 0, 255]
PAYLOAD_COLOUR = [18, 145, 21]

class AplDropEnv(gym.Env):
    """ Class for simplified and fast APL on openAI gym interface for dropping
        payload """
    metadata = {'render.modes': ['human']}
    X_MAX = 499
    X_MIN = 0
    Y_MAX = 499
    Y_MIN = 0
    HIKER_X = 5
    HIKER_Y = 5
    DRONE_X = 18
    DRONE_Y = 18
    TOP_CAMERA_X = 20
    TOP_CAMERA_Y = 20
    ALTITUDES = np.array([0, 1, 2, 3, 4])
    MAX_DRONE_ALTITUDE = ALTITUDES.max()
    HEADINGS = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    NORMALISATION_FACTOR = math.sqrt(pow(TOP_CAMERA_X, 2) +
                                     pow(TOP_CAMERA_Y, 2) +
                                     pow(max(ALTITUDES), 2))
    IMAGE_MULTIPLIER = 8
    drone = Drone()
    hiker = Hiker()
    OBS_SIZE_X = TOP_CAMERA_X
    OBS_SIZE_Y = TOP_CAMERA_Y  # + 1  # Extra column for sensors
    CHECK_ALTITUDE = True
    viewer_ego = None
    cells = None
    observations = np.zeros((OBS_SIZE_X, OBS_SIZE_Y), dtype=np.float32)
    obs_no_image = np.zeros((OBS_SIZE_X, OBS_SIZE_Y), dtype=np.float32)
    fix_map_around_hiker = np.zeros((OBS_SIZE_X, OBS_SIZE_Y), dtype=np.float32)
    normalised_map_around_hiker = None
    DROP_DISTANCE_FACTOR = 1.0
    MAX_STEPS = 50
    number_step = 0
    # render
    dronetrans = None
    payload_trans = None
    no_movement = False
    # a fix map
    alt_map = np.array(
        [[1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3],
         [1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 0, 0, 0],
         [1, 2, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 0, 0, 0],
         [1, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 3, 3, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0]])

    def __init__(self):
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.float32,
                                            shape=(self.OBS_SIZE_X,
                                                   self.OBS_SIZE_Y))
        rd.seed()
        self.reset()

    def step(self, action):
        """ Takes action and returns next state, reward, done flag and info """
        done = False
        self.number_step += 1
        self.drone.prev_x = self.drone.actual_x
        self.drone.prev_y = self.drone.actual_y
        self.drone.prev_alt = self.drone.actual_alt
        self.drone.prev_head = self.drone.actual_head
        self.drone.actual_x, self.drone.actual_y, self.drone.actual_alt,\
            self.drone.actual_head = self._one_step(action)
        valid_drone_pos = self._is_valid_drone_pos()
        if not valid_drone_pos or self.drone.dropped or \
           self.number_step == self.MAX_STEPS or self._is_done():
            done = True
        self.observations = self._get_observations(valid_drone_pos)
        if action == -1:  # intialisation
            reward = .0
        else:
            reward = self._reward(valid_drone_pos, action)
        info = {}
        #print(action, reward) #, self.observations)
        # print("after step: ", self.drone.actual_x, self.drone.actual_y, self.drone.actual_alt)
        return self.observations, reward, done, info

    def reset(self):
        """ Sets initial state for the env """
        self.number_step = 0
        # Random hiker position
        self.hiker.x_pos = self.HIKER_X
        self.hiker.y_pos = self.HIKER_Y
        self.hiker.alt = 0
        # building the altitude fix map
        self.fix_map_around_hiker = self.alt_map
        self.normalised_map_around_hiker = 255 - np.interp(
            self.fix_map_around_hiker,
            (min(self.ALTITUDES), max(self.ALTITUDES)),
            (0, 1)) * 255
        self.rgb_map_around_hiker = self._to_rgb5(
            self.normalised_map_around_hiker)
        # Random drone position
        self.drone.actual_x, self.drone.actual_y, self.drone.actual_alt,\
            self.drone.actual_head = self._get_drone_random_pos()
        self.drone.payload_x = 0
        self.drone.payload_y = 0
        self.drone.payload_status = None
        self.hiker.required_payload = [Payload.payloads[1],
                                       Payload.payloads[2],
                                       Payload.payloads[3],
                                       Payload.payloads[1]]
        if self.viewer_ego is not None:
            self._reset_viewer()
            if self.drone.dropped:
                time.sleep(1)
        self.drone.dropped = False
        self.no_movement = False
        return self._get_observations(True)

    def _is_done(self):
        if self._distance_to_hiker(self.drone.actual_x,
                                   self.drone.actual_y,
                                   self.drone.actual_alt - 4,
                                   normalise=False) == 0:
            return True
        return False

    @staticmethod
    def _to_rgb5(image):
        image.resize((image.shape[0], image.shape[1], 1))
        return np.repeat(image, 3, 2)

    def render(self, mode='human'):
        img = self.observations.astype(np.uint8)
        #print(img.shape[0])
        #print(img.shape[1])
        #print(img.shape[2])
        #exit(1)
        time.sleep(0.1)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            #from skimage.color import rgb2gray
            #img = rgb2gray(img)
            if self.viewer_ego is None:
                self.viewer_ego = rendering.SimpleImageViewer()
            self.viewer_ego.imshow(img)
            return self.viewer_ego.isopen


    def _reset_viewer(self):
        #self.payload_trans.set_translation(-100, -100)
        return 1

    #def seed(self, seed=None):
        #return 1


    def _get_drone_random_pos(self):
        """ Returns random values for initial positions inside view
            relative to hiker pos
        """
        x_pos = rd.randint(0, self.TOP_CAMERA_X - 1)
        y_pos = rd.randint(0, self.TOP_CAMERA_Y - 1)
        alt = 4  # self.fix_map_around_hiker[x_pos, y_pos] + 1
        head = 0  # np.random.choice(self.HEADINGS)
        return x_pos, y_pos, alt, head

    def _is_valid_drone_pos(self):
        """ Checks if drone is inside relative boundaries or not crashed """
        if self.drone.actual_x < 0 or \
                self.drone.actual_x >= self.TOP_CAMERA_X or \
                self.drone.actual_y < 0 or \
                self.drone.actual_y >= self.TOP_CAMERA_Y:
            #print("fuera")
            return False
        if self.drone.actual_alt > self.MAX_DRONE_ALTITUDE:
            #print("arriba")
            return False
        if self.drone.actual_alt <= self.ALTITUDES.min():
            #print("choco piso")
            return False
        if self.CHECK_ALTITUDE:
            if self.drone.actual_alt <= self.alt_map[self.drone.actual_x, self.drone.actual_y]:
                #print("hit something")
                return False
        return True

    def _get_hiker_random_pos(self):
        """ Returns random position of the hiker within boundaries"""
        #x_pos = rd.randint(self.X_MIN + self.HIKER_RELATIVE_POS + 1,
                           #self.X_MAX - self.HIKER_RELATIVE_POS - 1)
        #y_pos = rd.randint(self.Y_MIN + self.HIKER_RELATIVE_POS + 1,
                           #self.Y_MAX - self.HIKER_RELATIVE_POS - 1)
        x_pos = 5
        y_pos = 10
        return x_pos, y_pos, 0

    def _is_valid_hiker_pos(self):
        """ validate hikers initial position """
        # if self.hiker.alt >= self.MAX_DRONE_ALTITUDE - 1:
        if self.hiker.alt > 0:
            return False
        return True

    def _one_step(self, action):
        """ Dynamics for actions. Follows APL """
        #self.no_movement = False
        #prev_x = self.drone.actual_x
        #prev_y = self.drone.actual_y
        #prev_alt = self.drone.actual_alt
        #prev_head = self.drone.actual_head
        next_x = self.drone.actual_x
        next_y = self.drone.actual_y
        next_alt = self.drone.actual_alt
        next_head = self.drone.actual_head
        if action == 0:  # down
            next_x += 1
        if action == 1:  # right
            next_y += 1
        if action == 2:  # left
            next_y += -1
        if action == 3:  # up
            next_x += -1
        if action == 4:
            next_alt += -1
        if action == 5:
            next_alt += 1
        if action == 6:
            self.drone.dropped, self.drone.payload_x, self.drone.payload_y, \
                self.drone.payload_status = self.drop_payload()
        #self.drone.actual_x = next_x
        #self.drone.actual_y = next_y
        #self.drone.actual_alt = next_alt
        #self.drone.actual_head = next_head
        #if not self._is_valid_drone_pos():
            #self.no_movement = True
            #return prev_x, prev_y, prev_alt, prev_head
        return next_x, next_y, next_alt, next_head

    def drop_payload(self):
        """ Chooses slot to drop, drops and return final position of payload
            and status of the payload
        """
        return True, self.drone.actual_x, self.drone.actual_y, PayloadStatus.status[0]

    def _reward(self, is_valid_pos, action):
        """ Reward for getting closer to the hiker
            negative reward for changing altitude and dropping
        """
        if not is_valid_pos:
            return -1.
        if action in {4, 5, 6}:
            return -1.
        distance = self._distance_to_hiker(self.drone.actual_x,
                                           self.drone.actual_y,
                                           self.drone.actual_alt - 4,
                                           normalise=True)
        if distance == 0:
            #print("made it")
            return 100.
        prev_distance = self._distance_to_hiker(self.drone.prev_x,
                                                self.drone.prev_y,
                                                self.drone.prev_alt - 4,
                                                normalise=True)
        if prev_distance - distance > 0:
            return 1.
        return -1.

    def _get_observations(self, valid_drone_pos):
        obs = np.copy(self.rgb_map_around_hiker)
        obs = np.kron(obs, np.ones([self.IMAGE_MULTIPLIER,
                                    self.IMAGE_MULTIPLIER, 1]))
        if self.drone.dropped:
            payload_row = [np.uint8(self.drone.payload_x * self.IMAGE_MULTIPLIER),
                np.uint8((self.drone.payload_x + 1) * self.IMAGE_MULTIPLIER),
                np.uint8((self.drone.payload_x + 1) * self.IMAGE_MULTIPLIER),
                np.uint8((self.drone.payload_x) * self.IMAGE_MULTIPLIER)]

            payload_col = [np.uint8(self.drone.payload_y * self.IMAGE_MULTIPLIER),
                np.uint8((self.drone.payload_y) * self.IMAGE_MULTIPLIER),
                np.uint8((self.drone.payload_y + 1) * self.IMAGE_MULTIPLIER),
                np.uint8((self.drone.payload_y + 1) * self.IMAGE_MULTIPLIER)]
            rr, cc = draw.polygon(payload_row, payload_col)
            draw.set_color(obs, (rr, cc), PAYLOAD_COLOUR)
        if valid_drone_pos:
            # draw drone
            rr, cc = draw.circle_perimeter(np.uint8((self.drone.actual_x + 0.5) *
                                                    self.IMAGE_MULTIPLIER),
                                           np.uint8((self.drone.actual_y + 0.5) *
                                                    self.IMAGE_MULTIPLIER),
                                           np.uint8(self.IMAGE_MULTIPLIER /
                                                    7.5 * self.drone.actual_alt))
            draw.set_color(obs, (rr, cc), DRONE_COLOUR)
            # draw hiker as an x
            rr, cc = draw.line(np.uint8(self.hiker.x_pos *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8(self.hiker.y_pos *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8((self.hiker.x_pos + 1) *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8((self.hiker.y_pos + 1) *
                                        self.IMAGE_MULTIPLIER))
            draw.set_color(obs, (rr, cc), HIKER_COLOUR)
            rr, cc = draw.line(np.uint8((self.hiker.x_pos + 1) *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8(self.hiker.y_pos *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8(self.hiker.x_pos *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8((self.hiker.y_pos + 1) *
                                        self.IMAGE_MULTIPLIER))
            draw.set_color(obs, (rr, cc), HIKER_COLOUR)
        return obs


    def _distance_to_hiker(self, pos_x, pos_y, alt, normalise=True):
        dist = math.sqrt(pow(self.hiker.x_pos - pos_x, 2) +
                         pow(self.hiker.y_pos - pos_y, 2) +
                         pow(self.hiker.alt - alt, 2))
        if normalise:
            dist = dist / self.NORMALISATION_FACTOR
        return dist
