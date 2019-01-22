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
    x = 0
    y = 0
    alt = 0
    head = 0
    dropped = False
    payload_x = 0
    payload_y = 0
    payload_status = None


class Hiker(object):
    x_global = 0
    y_global = 0
    x_local = 0
    y_local = 0
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
OBJECTS_CODE = {'mountain ridge': 0, 'trail': 1, 'shore bank': 2,
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
    HIKER_RELATIVE_POS = 5
    TOP_CAMERA_X = HIKER_RELATIVE_POS * 2 + 1
    TOP_CAMERA_Y = HIKER_RELATIVE_POS * 2 + 1
    ALTITUDES = np.array([0, 1, 2, 3, 4])
    MAX_DRONE_ALTITUDE = ALTITUDES.max()
    HEADINGS = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    NORMALISATION_FACTOR = math.sqrt(pow(TOP_CAMERA_X / 2, 2) +
                                     pow(TOP_CAMERA_Y / 2, 2) +
                                     pow(max(ALTITUDES), 2))
    IMAGE_MULTIPLIER = 8
    drone = Drone()
    hiker = Hiker()
    OBS_SIZE_X = TOP_CAMERA_X
    OBS_SIZE_Y = TOP_CAMERA_Y  # + 1  # Extra column for sensors
    CHECK_ALTITUDE = False
    viewer_ego = None
    cells = None
    observations = np.zeros((OBS_SIZE_X, OBS_SIZE_Y), dtype=np.float32)
    obs_no_image = np.zeros((OBS_SIZE_X, OBS_SIZE_Y), dtype=np.float32)
    fix_map_around_hiker = np.zeros((OBS_SIZE_X, OBS_SIZE_Y), dtype=np.float32)
    fix_alt_map_around_hiker = np.zeros((OBS_SIZE_X, OBS_SIZE_Y),
                                        dtype=np.float32)
    normalised_map_around_hiker = None
    DROP_DISTANCE_FACTOR = 1.0
    number_step = 0
    # render
    dronetrans = None
    payload_trans = None

    def __init__(self):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.float32,
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
        next_x, next_y, next_alt, next_head = self._one_step(action)
        self.drone.x = next_x
        self.drone.y = next_y
        self.drone.alt = next_alt
        self.drone.head = next_head
        valid_drone_pos = self._is_valid_drone_pos()
        if not valid_drone_pos or self.drone.dropped or \
           self.number_step == 20:
            done = True
        # TODO test delete:
        if self._distance_to_hiker(self.drone.x,
                                   self.drone.y, self.drone.alt - 1,
                                   normalise=False) == 0:
            #print("LLEGUÉ ARRIBA HIKER")
            done = True
        self.observations = self._get_observations(valid_drone_pos)
        if action == -1:  # intialisation
            reward = .0
        else:
            reward = self._reward(valid_drone_pos)
        info = {}
        #print(action, reward) #, self.observations)
        return self.observations, reward, done, info

    def reset(self):
        """ Sets initial state for the env """
        self.number_step = 0
        # Random hiker position
        self.hiker.x_global, self.hiker.y_global, \
            self.hiker.x_local, self.hiker.y_local, \
            self.hiker.alt = self._get_hiker_random_pos()
        while not self._is_valid_hiker_pos():
            self.hiker.x_global, self.hiker.y_global, \
                self.hiker.x_local, self.hiker.y_local, \
                self.hiker.alt = self._get_hiker_random_pos()
        # Random drone position
        self.drone.x, self.drone.y, self.drone.alt, self.drone.head = \
            self._get_drone_random_pos()
        while not self._is_valid_drone_pos():
            self.drone.x, self.drone.y, self.drone.alt, self.drone.head = \
                self._get_drone_random_pos()
        # building the altitude fix map
        self.fix_map_around_hiker = self._get_alt_map_around(
            self.hiker.x_global, self.hiker.y_global)
        self.normalised_map_around_hiker = 255 - np.interp(
            self.fix_map_around_hiker,
            (min(self.ALTITUDES), max(self.ALTITUDES)),
            (0, 1)) * 255
        self.rgb_map_around_hiker = self._to_rgb5(self.normalised_map_around_hiker)
        self.fix_alt_map_around_hiker = \
            self._get_alt_map_around(self.hiker.x_global,
                                     self.hiker.y_global)
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
        return self._get_observations(True)

    #def render(self, mode='human', close=False):
        #from gym.envs.classic_control import rendering
        #cell_width = 20
        #cell_height = 20
        #if self.viewer_ego is None:
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
            #[[self.cells[y_cell][x_cell].set_color(self.obs_no_image[x_cell][y_cell] / 255,
                                                  #self.obs_no_image[x_cell][y_cell] / 255,
                                                  #self.obs_no_image[x_cell][y_cell] / 255)
              #for x_cell in range(self.OBS_SIZE_X)]
             #for y_cell in range(self.OBS_SIZE_Y)]
            ##np.set_printoptions(threshold=np.nan)
            ##print(self.obs_no_image)
            #drone = rendering.FilledPolygon(
                #[(1, 1), (1, cell_width), (cell_width, int(cell_height / 2))])
            #self.dronetrans = rendering.Transform()
            #drone.add_attr(self.dronetrans)
            #self.viewer_ego.add_geom(drone)
            #self.dronetrans.set_translation(self.drone.x * cell_width,
                                            #self.drone.y * cell_height)
            #drone.set_color(.8, .6, .3)
            #payload = rendering.FilledPolygon(
                #[(3, 3), (3, cell_width * .75),
                 #(cell_height * .75, cell_height * .75),
                 #(cell_height * .75, 3)])
            #self.payload_trans = rendering.Transform()
            #payload.add_attr(self.payload_trans)
            #self.viewer_ego.add_geom(payload)
            #self.payload_trans.set_translation(-100, -100)
            #payload.set_color(.3, .3, .8)
        #self.dronetrans.set_translation(self.drone.x * cell_width,
                                        #self.drone.y * cell_height)
        #if self.drone.dropped:
            ##self.dronetrans.set_translation(1, 1)
            #self.payload_trans.set_translation(
                #(self.drone.payload_x) * cell_width,
                #self.drone.payload_y * cell_height)
        #if not self._is_valid_drone_pos():
            #self.dronetrans.set_translation(-100, -100)

        #self._set_cells_color()
        #time.sleep(0.09)
        #return self.viewer_ego.render(return_rgb_array=mode == 'rgb_array')

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

    def _set_cells_color(self):
        #for width_cell in range(self.OBS_SIZE_Y):
            #for height_cell in range(self.OBS_SIZE_X):
                #if self.observations[height_cell][width_cell] == -1:
                    #color = (0.0, 0.0, 0.0)
                #elif self.observations[height_cell][width_cell] == 0:
                    #color = (0.0, 0.45, 0.14)
                #elif self.observations[height_cell][width_cell] == 1:
                    #color = (0.0, 0.72, 0.22)
                #elif self.observations[height_cell][width_cell] == 2:
                    #color = (0.0, 0.87, 0.27)
                #elif self.observations[height_cell][width_cell] == 3:
                    #color = (0.47, 0.99, 0.63)
                #elif self.observations[height_cell][width_cell] == 4:
                    #color = (0.65, 0.99, 0.76)
                #elif self.observations[height_cell][width_cell] == 5:
                    #color = (1.0, 0.0, 0.0)
                #elif self.observations[height_cell][width_cell] == 6:
                    #color = (1.0, 0.2, 0.2)
                #elif self.observations[height_cell][width_cell] == 7:
                    #color = (1.0, 0.5, 0.5)
                #elif self.observations[height_cell][width_cell] == 8:
                    #color = (1.0, 0.7, 0.7)
                #elif self.observations[height_cell][width_cell] == 255:
                    #color = (0.3, 0.7, 0.2)
                #else:
                    #color = (0.5, self.observations[height_cell][width_cell] / 18., 0.5)
                #self.cells[width_cell][height_cell].set_color(color[0],
                                                              #color[1],
                                                              #color[2])
        self.cells[self.OBS_SIZE_Y - 1][0].set_color(
            self.obs_no_image[0][self.OBS_SIZE_Y - 1] / 255,
            self.obs_no_image[0][self.OBS_SIZE_Y - 1] / 255,
            self.obs_no_image[0][self.OBS_SIZE_Y - 1] / 255)
        self.cells[self.OBS_SIZE_Y - 1][1].set_color(
            self.obs_no_image[1][self.OBS_SIZE_Y - 1] / 255,
            self.obs_no_image[1][self.OBS_SIZE_Y - 1] / 255,
            self.obs_no_image[1][self.OBS_SIZE_Y - 1] / 255)
        self.cells[self.OBS_SIZE_Y - 1][2].set_color(
            self.obs_no_image[2][self.OBS_SIZE_Y - 1] / 255,
            self.obs_no_image[2][self.OBS_SIZE_Y - 1] / 255,
            self.obs_no_image[2][self.OBS_SIZE_Y - 1] / 255)
        self.cells[self.OBS_SIZE_Y - 1][3].set_color(
            self.obs_no_image[3][self.OBS_SIZE_Y - 1] / 255,
            self.obs_no_image[3][self.OBS_SIZE_Y - 1] / 255,
            self.obs_no_image[3][self.OBS_SIZE_Y - 1] / 255)

    def _get_drone_random_pos(self):
        """ Returns random values for initial positions inside view
            relative to hiker pos
        """
        x_pos = rd.randint(0, self.TOP_CAMERA_X)
        y_pos = rd.randint(0, self.TOP_CAMERA_Y)
        alt = 3  # np.random.choice(self.ALTITUDES)
        head = np.random.choice(self.HEADINGS)
        return x_pos, y_pos, alt, head

    def _is_valid_drone_pos(self):
        """ Checks if drone is inside relative boundaries or not crashed """
        if self.drone.x < 0 or \
                self.drone.x >= self.TOP_CAMERA_X or \
                self.drone.y < 0 or \
                self.drone.y >= self.TOP_CAMERA_Y:
            #print("fuera")
            return False
        if self.drone.alt > self.MAX_DRONE_ALTITUDE:
            #print("arriba")
            return False
        if self.drone.alt <= self.ALTITUDES.min():
            #print("choco piso")
            return False
        if self.CHECK_ALTITUDE:
            if self.drone.alt <= self.fix_alt_map_around_hiker[self.drone.x,
                                                               self.drone.y]:
                #print("hit something")
                return False
        return True

    def _get_hiker_random_pos(self):
        """ Returns random position of the hiker within boundaries"""
        x_pos = rd.randint(self.X_MIN + self.HIKER_RELATIVE_POS + 1,
                           self.X_MAX - self.HIKER_RELATIVE_POS - 1)
        y_pos = rd.randint(self.Y_MIN + self.HIKER_RELATIVE_POS + 1,
                           self.Y_MAX - self.HIKER_RELATIVE_POS - 1)
        x_pos = 300
        y_pos = 300
        np.set_printoptions(threshold=np.nan)
        return x_pos, y_pos, self.HIKER_RELATIVE_POS, self.HIKER_RELATIVE_POS,\
            self.full_altitude_map[x_pos, y_pos]

    def _is_valid_hiker_pos(self):
        """ validate hikers initial position """
        # if self.hiker.alt >= self.MAX_DRONE_ALTITUDE - 1:
        if self.hiker.alt > 0:
            return False
        return True

    def _one_step(self, action):
        """ Dynamics for actions. Follows APL """
        next_x = self.drone.x
        next_y = self.drone.y
        next_alt = self.drone.alt
        next_head = self.drone.head
        if action == 0:
            next_x += 1
        if action == 1:
            next_y += 1
        if action == 2:
            next_y += -1
        if action == 3:
            next_x += -1
        if action == 4:
            next_alt -= 1
        #if action == 5:
            #self.drone.dropped, self.drone.payload_x, self.drone.payload_y, \
                #self.drone.payload_status = self.drop_payload()
            ## TODO apl dynamics for dropping final place
            ## TODO dropping also advance drone one step forward
            #self.drone.payload_x = self.drone.x
            #self.drone.payload_y = self.drone.y
            ##from pprint import pprint
            ##pprint(vars(self.drone))
        return next_x, next_y, next_alt, next_head

    def drop_payload(self):
        """ Chooses slot to drop, drops and return final position of payload
            and status of the payload
        """
        return True, self.drone.x, self.drone.y, PayloadStatus.status[0]

    def _reward(self, is_valid_pos):
        """ If the drone is not on a valid position return negative reward,
            else, negative distance to the hiker """
        reward = .0
        if not is_valid_pos:
            return -1.
        distance = self._distance_to_hiker(self.drone.x,
                                           self.drone.y,
                                           self.drone.alt - 1,
                                           normalise=True)
        if distance == 0:
            reward = 10.
        else:
            reward = 1. - distance
        #if self.drone.dropped:
            ## TODO reward based on payload status
            #distance = self._distance_to_hiker(self.drone.payload_x,
                                               #self.drone.payload_y,
                                               #self.drone.alt,
                                               #normalise=True)
            ##reward += 10. * (1. - distance + 4 - self.drone.alt) * (4 - self.drone.alt)
            #if distance == 0:
                #reward = 10.
            #else:
                #reward = 1. - distance
        return reward

    def _get_observations(self, valid_drone_pos):
        obs = np.copy(self.rgb_map_around_hiker)
        #if valid_drone_pos:
            ## Drone altitude
            #obs[0, self.OBS_SIZE_Y - 1, 0] = self.drone.alt / \
                #self.ALTITUDES.max() * 255
            #obs[0, self.OBS_SIZE_Y - 1, 1] = self.drone.alt / \
                #self.ALTITUDES.max() * 155
            #obs[0, self.OBS_SIZE_Y - 1, 2] = self.drone.alt / \
                #self.ALTITUDES.max() * 55
            ## Drone heading
            #obs[1, self.OBS_SIZE_Y - 1, :] = self.drone.head / \
                #self.HEADINGS.max() * 255
            ## Drone relative x pos
            #obs[2, self.OBS_SIZE_Y - 1, :] = self.drone.x / self.TOP_CAMERA_X \
                #* 255
            ## Drone relative y pos
            #obs[3, self.OBS_SIZE_Y - 1, :] = self.drone.y / self.TOP_CAMERA_Y \
                #* 255

        obs = np.kron(obs, np.ones([self.IMAGE_MULTIPLIER,
                                    self.IMAGE_MULTIPLIER, 1]))
        if valid_drone_pos:
            # draw drone
            rr, cc = draw.circle_perimeter(np.uint8((self.drone.x + 0.5) *
                                                    self.IMAGE_MULTIPLIER),
                                           np.uint8((self.drone.y + 0.5) *
                                                    self.IMAGE_MULTIPLIER),
                                           np.uint8(self.IMAGE_MULTIPLIER /
                                                    7.5 * self.drone.alt))
            draw.set_color(obs, (rr, cc), DRONE_COLOUR)
            # draw hiker as an x
            rr, cc = draw.line(np.uint8(self.hiker.x_local *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8(self.hiker.y_local *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8((self.hiker.x_local + 1) *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8((self.hiker.y_local + 1) *
                                        self.IMAGE_MULTIPLIER))
            draw.set_color(obs, (rr, cc), HIKER_COLOUR)
            rr, cc = draw.line(np.uint8((self.hiker.x_local + 1) *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8(self.hiker.y_local *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8(self.hiker.x_local *
                                        self.IMAGE_MULTIPLIER),
                               np.uint8((self.hiker.y_local + 1) *
                                        self.IMAGE_MULTIPLIER))
            draw.set_color(obs, (rr, cc), HIKER_COLOUR)

        #print(obs.shape)
        #from matplotlib import pyplot as PLT
        #PLT.imshow(obs.astype(np.uint8))
        #PLT.show()
        #print(obs)
        #exit(1)
        return obs

    # TODO merge _get_alt_map_around with _get_map_around
    def _get_alt_map_around(self, x_pos, y_pos):
        """ returns the map around the x and y pos, with objects altitude """
        map_around = np.zeros((self.OBS_SIZE_X, self.OBS_SIZE_Y),
                              dtype=np.float32)
        for x_cell in range(self.TOP_CAMERA_X):
            for y_cell in range(self.TOP_CAMERA_Y):
                try:
                    alt = self.full_altitude_map[
                        int(x_pos - self.TOP_CAMERA_X / 2 + x_cell) + 1][
                            int(y_pos - self.TOP_CAMERA_Y / 2 + y_cell) + 1]
                    map_around[x_cell][y_cell] = alt
                except IndexError:
                    map_around[x_cell][y_cell] = -1
        return map_around

    def _get_map_around(self, x_pos, y_pos):
        """ returns the map around the x and y pos, with objects code """
        map_around = np.zeros((self.OBS_SIZE_X, self.OBS_SIZE_Y),
                              dtype=np.float32)
        for x_cell in range(self.TOP_CAMERA_X):
            for y_cell in range(self.TOP_CAMERA_Y):
                try:
                    alt = self.full_objects_map[
                        int(x_pos - self.TOP_CAMERA_X / 2 + x_cell)][
                            int(y_pos - self.TOP_CAMERA_Y / 2 + y_cell)]
                    map_around[x_cell][y_cell] = alt
                except IndexError:
                    map_around[x_cell][y_cell] = -1
        return map_around

    def _distance_to_hiker(self, pos_x, pos_y, alt, normalise=True):
        dist = math.sqrt(pow(self.HIKER_RELATIVE_POS - pos_x, 2) +
                         pow(self.HIKER_RELATIVE_POS - pos_y, 2) +
                         pow(self.hiker.alt - alt, 2))
        if normalise:
            dist = dist / self.NORMALISATION_FACTOR
        return dist
