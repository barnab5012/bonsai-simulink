#!/usr/bin/env python3

import logging
import math
from collections import deque

_STEPLIMIT = 480

class Model:
    def load(self, eng):
        """
        Load the specified simulink model.
        """
        eng.eval("load_system('simulink_househeat')", nargout=0)

    def executable_name(self):
        """
        Returns the name of the executable (Simulink Coder Only).
        """
        return "./simulink_househeat"

    def episode_init(self):
        """
        This method is called at the beginning of each episode.
        """
        self.nsteps = 0
        self.action = None
        self.state = None
        self.logged_state = None
        self.reward = None
        self.terminal = None
        self.total_reward = 0.0
        empty_observation = [0.0,0.0,0.0,0.0,0.0]
        self.temperature_difference_history = deque(empty_observation)

    def episode_step(self):
        """
        This method is called at the beginning of each iteration.
        """
        self.nsteps += 1
        
    def convert_config(self, conf):
        """
        Convert the dictionary of config from the brain into an ordered
        list of config for the simulation.
        """
        if len(conf) == 0:
            # In prediction mode the brain doesn't supply config.
            conf['outside_phase'] = 0.0
        return [ conf['outside_phase'], ]

    def convert_input(self, simulator_state):
        """
        Called with a list of inputs from the model,
        returns (state, reward, terminal).
        """

        # First map the ordered state list from the simulation into a
        # state dictionary for the brain.

        self.logged_state = {
            'heat_cost':        simulator_state[0],
            'set_temp':         simulator_state[1],
            'room_temp':        simulator_state[2],
            'room_temp_change':     simulator_state[3],
            'outside_temp':     simulator_state[4],
            'outside_temp_change':  simulator_state[5],
        }

        tdiff = math.fabs(self.logged_state['set_temp'] - self.logged_state['room_temp'])
        self.temperature_difference_history.appendleft(tdiff)
        self.temperature_difference_history.pop()

        # First transform the simulation outputs into a state dictionary for the BRAIN.
        self.state = {
            'heat_cost':        simulator_state[0],
            'temperature_difference':           tdiff,
            'temperature_difference_t1':        self.temperature_difference_history[0],
            'temperature_difference_t2':        self.temperature_difference_history[1],
            'temperature_difference_t3':        self.temperature_difference_history[2],
            'temperature_difference_t4':       self.temperature_difference_history[3],
            'temperature_difference_t5':       self.temperature_difference_history[4],
            'outside_temp_change':  simulator_state[5],
        }

        # Note the tstamp in the logging output.
        self.tstamp = simulator_state[6]
        
        # To compute the reward function value we start by taking the
        # difference between the set point temperature and the actual
        # room temperature.
        tdiff = math.fabs(self.logged_state['set_temp'] - self.logged_state['room_temp'])

        # Raise the difference to the 0.4 power.  The non-linear
        # function enhances the reward distribution near the desired
        # temperature range.  Please refer to the Bonsai training
        # video on reward functions for more details.
        nonlinear_diff = pow(tdiff, 0.4)

        # Scale the nonlinear difference so differences in the range
        # +/- 2 degrees (C) map between 0 and 1.0.
        # 2 degree ^ 0.4 = 1.32
        scaled_diff = nonlinear_diff / 1.32

        # Since we need a positive going reward function, subtract the
        # scaled difference from 1.0.  This reward value will be 1.0
        # when we are precisely matching the set point and will fall
        # to less than 0.0 when we exceed 2 degrees (C) from the set
        # point.
        self.reward = 1.0 - scaled_diff
        
        self.terminal = self.nsteps >= _STEPLIMIT or self.reward < 0.0

        if self.nsteps > 0:
            self.total_reward += self.reward

        return self.state, self.reward, self.terminal

    def convert_output(self, brain_action):
        """
        Called with a dictionary of actions from the brain, returns an
        ordered list of outputs for the simulation model.
        """
        outlist = []
        if brain_action is not None:

            # If you need to clamp the action, for example becuase the TRPO algorithm does not clamp automatically.
            clamped_action = min(1, max(-1, brain_action['heater_on'])) # clamp to [-1,1] 

            # If you need to scale the action, for example to to [0,1]
            scaled_action = (clamped_action +2)/2 + 1 # scale to [0,1]

            self.action = brain_action
            outlist = [ brain_action['heater_on'], ]

        return outlist

    def format_start(self):
        """
        Emit a formatted header and initial state line at the beginning
        of each episode.
        """
        logging.info(" itr  time h =>    cost  set   troom   droom tout dout = t    rwd")
        logging.info("                %7.1f %4.1f %7.1f %7.1f %4.1f %4.1f" % (
            self.logged_state['heat_cost'],
            self.logged_state['set_temp'],
            self.logged_state['room_temp'],
            self.logged_state['room_temp_change'],
            self.logged_state['outside_temp'],
            self.logged_state['outside_temp_change'],
        ))

    def format_step(self):
        """
        Emit a formatted line for each iteration.
        """
        if self.terminal:
            totrwdstr = " %6.3f" % self.total_reward
        else:
            totrwdstr = ""
            
        logging.info(" %3d %5.3f %1.0f => %7.1f %4.1f %7.1f %7.1f %4.1f %4.1f = %i %6.3f%s" % (
            self.nsteps,
            self.tstamp,
            self.action['heater_on'],
            self.logged_state['heat_cost'],
            self.logged_state['set_temp'],
            self.logged_state['room_temp'],
            self.logged_state['room_temp_change'],
            self.logged_state['outside_temp'],
            self.logged_state['outside_temp_change'],
            self.terminal,
            self.reward,
            totrwdstr,
        ))