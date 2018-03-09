#!/usr/bin/env python3

import logging
import math

from collections import deque

_STEPLIMIT = 500

class Model:
    def load(self, eng):
        """
        Load the specified simulink model.
        """
        eng.eval("load_system('rolling_mill')", nargout=0)

    def executable_name(self):
        """
        Returns the name of the executable (Simulink Coder Only).
        """
        return "./rolling_mill"

    def clockdivide_init(self):
        """
        This method is called at the beginning of each simulator episode.
        """
        self.sim_nsteps = 0

    def clockdivide_step(self):
        """
        This method is called each time the simulator takes a step.  It
        should return True if the brain should take this step as well.
        This routine may be used to divide down the simulation clock
        by returning true every N steps.  If dividing the simulation
        clock this routine should return true on the first step of the
        simulator episode (self.sim_nsteps == 0).

        """
        # retval = (self.sim_nsteps % 5) == 0
        retval = True
        self.sim_nsteps += 1
        return retval
    
    def episode_init(self):

        """
        This method is called at the beginning of each brain episode.
        """
        self.brain_nsteps = 0
        self.action = None
        self.state = None
        self.reward = None
        self.terminal = None
        self.fx_deque = None
        self.fy_deque = None

    def episode_step(self):
        """
        This method is called at the beginning of each iteration.
        """
        self.brain_nsteps += 1
        
    def convert_config(self, conf):
        """
        Convert the dictionary of config from the brain into an ordered
        list of config for the simulation.
        """
        if len(conf) == 0:
            # In prediction mode the brain doesn't supply config.
            conf['dummy'] = -1.0
        return [ conf['dummy'], ]

    def convert_input(self, inlist):
        """
        Called with a list of inputs from the model,
        returns (state, reward, terminal).
        """

        # Create recurrence for f_x and f_y.
        if not self.fx_deque:
            self.fx_deque = deque([inlist[0] for nn in range(0, 3)])
        else:
            self.fx_deque.appendleft(inlist[0])
            self.fx_deque.pop()
            
        if not self.fy_deque:
            self.fy_deque = deque([inlist[1] for nn in range(0, 3)])
        else:
            self.fy_deque.appendleft(inlist[1])
            self.fy_deque.pop()
        
        # First map the ordered state list from the simulation into a
        # state dictionary for the brain.
        self.state = {
            'f0x':		self.fx_deque[0],
            'f1x':		self.fx_deque[1],
            'f2x':		self.fx_deque[2],
            'f0y':		self.fy_deque[0],
            'f1y':		self.fy_deque[1],
            'f2y':		self.fy_deque[2],
            'delta_x':		inlist[2],
            'delta_y':		inlist[3],
        }

        # Note the tstamp in the logging output.
        self.tstamp = inlist[5]
        
        # 0.01 ^ 0.4 = .158
        # 0.05 ^ 0.4 = .302
        # 0.10 ^ 0.4 = .398
        # 0.20 ^ 0.4 = .525
        self.reward = 1.0 - pow(abs(self.state['delta_x']) + abs(self.state['delta_y']), 0.4)/0.525

        self.terminal = self.reward < 0.0 or self.brain_nsteps >= _STEPLIMIT

        return self.state, self.reward, self.terminal

    def convert_output(self, act):
        """
        Called with a dictionary of actions from the brain, returns an
        ordered list of outputs for the simulation model.
        """
        if act is not None:
            act['u_x'] *= 30.0
            act['u_y'] *= 20.0
        
        outlist = []
        if act is not None:
            self.action = act
            outlist = [
                act['u_x'],
                act['u_y'],
            ]

        return outlist

    def format_start(self):
        """
        Emit a formatted header and initial state line at the beginning
        of each episode.
        """
        self.total_reward = 0.0
        logging.info("  itr   tm    u_x   u_y =>         f_x         f_y        dx      dy = t    rwd")
        logging.info("                           %11.1f %11.1f   %7.3f %7.3f" % (
            self.state['f0x'],
            self.state['f0y'],
            self.state['delta_x'],
            self.state['delta_y'],
        ))

    def format_step(self):
        """
        Emit a formatted line for each iteration.
        """
        self.total_reward += self.reward
        if self.terminal:
            totrwdstr = " %6.3f" % self.total_reward
        else:
            totrwdstr = ""
            
        logging.info(" %4d %5.3f %5.1f %5.1f => %11.1f %11.1f   %7.3f %7.3f = %i %6.3f%s" % (
            self.brain_nsteps,
            self.tstamp,
            self.action['u_x'],
            self.action['u_y'],
            self.state['f0x'],
            self.state['f0y'],
            self.state['delta_x'],
            self.state['delta_y'],
            self.terminal,
            self.reward,
            totrwdstr,
        ))

