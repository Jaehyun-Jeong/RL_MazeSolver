# -*- coding: utf-8 -*-
import numpy as np
import torch
import pygame
import Maze_Generator as maze_generator
from os.path import exists
from sys import exit

class MazeSolver:
    """
    Solve a maze given in "block form" by maze_generator.py (a NumPy Array with 0 as a corridor and 1 as a wall block; outer edges are wall blocks).
    Show progress graphically. The mouse can be used to pick begin and end positions; maze resolution changed by up & down cursor keys.

    @author: kalle
    """

    def __init__(self, screen, rect, blocks, start_pos, end_pos):

        self.screen = screen
        self.rect = np.asarray(rect, dtype=np.int16)  # the rect inside which the maze resides (top x, top y, width, height)
        self.blocks = blocks  # blocks should be given as (y, x) array, dtype=np.byte, 0 as a corridor and 1 as a wall
        self.start_pos = start_pos.astype(np.int16)
        self.end_pos = end_pos.astype(np.int32)  # the goal to seek
        self.block_size = np.asarray(np.shape(self.blocks))
        self.screen_size = np.asarray(self.screen.get_size())
        self.screen_block_size = np.min(self.rect[2:4] / np.flip(self.block_size))
        self.screen_block_offset = self.rect[0:2] + (self.rect[2:4] - self.screen_block_size * np.flip(self.block_size)) // 2
        self.prev_update = 0
        self.running = True
        self.screen_info = pygame.Surface((1100, 70))

        self.junctions = np.zeros((100, 6), dtype=np.int16)  # will store the junction cell, the cell before that, distance to end, and nr of directions to go
        self.junction_nr = -1
        self.junctions_used = 0

        self.cell_size = 0
        self.slow_mode = False
        self.info_display = False
        self.last_message = ''
        pygame.font.init()
        self.font = pygame.font.SysFont('CourierNew', 16)

        self.background_color = (0, 0, 0)
        self.info_color = (255, 255, 255)
        self.maze_color = (200, 200, 200)
        self.start_color = (50, 50, 200)
        self.end_color = (50, 200, 50)
        self.path_color = (200, 200, 0)
        self.solution_color = (240, 50, 50)

    def draw_cell(self, cell, previous, color):
        # draw passage from cell to neighbor. As these are always adjacent can min/max be used.
        min_coord = (np.flip(np.minimum(cell, previous)) * self.screen_block_size + self.screen_block_offset).astype(np.int16)
        max_coord = (np.flip(np.maximum(cell, previous)) * self.screen_block_size + int(self.screen_block_size) + self.screen_block_offset).astype(np.int16)
        pygame.draw.rect(self.screen, color, (min_coord, max_coord - min_coord))

        if self.slow_mode or pygame.time.get_ticks() > self.prev_update + 16:  # increase the addition for faster but less frequent screen updates
            self.prev_update = pygame.time.get_ticks()
            pygame.display.flip()

            # when performing display flip, handle pygame events as well.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_f:
                        self.toggle_fullscreen()
                    if event.key == pygame.K_m:
                        self.toggle_slow_mode()
        if self.slow_mode:
            pygame.time.wait(3)

    def toggle_info_display(self):

        # switch between a windowed display and full screen
        self.info_display = not(self.info_display)
        if self.info_display:
            self.plot_info(self.last_message)
        else:
            x = (self.screen_size[0] - self.screen_info.get_size()[0]) // 2
            self.screen.fill(self.background_color, (x, 10, self.screen_info.get_size()[0], self.screen_info.get_size()[1]))
            pygame.display.update((x, 10, self.screen_info.get_size()[0], self.screen_info.get_size()[1]))

    def toggle_slow_mode(self):

        # switch between a windowed display and full screen
        self.slow_mode = not(self.slow_mode)
        self.plot_info(self.last_message)

    def plot_info(self, phase_msg):

        # show info

        if self.info_display:

            while self.screen_info.get_locked():
                self.screen_info.unlock()
            self.screen_info.fill(self.background_color)

            y = 0
            x = 0
            info_msg = 'i:     Info display ON/OFF'
            y = self.plot_info_msg(self.screen_info, x, y, info_msg)
            if self.slow_mode:
                info_msg = 'm:     Slow mode ON/OFF (on)'
            else:
                info_msg = 'm:     Slow mode ON/OFF (off)'
            y = self.plot_info_msg(self.screen_info, x, y, info_msg)
            info_msg = 's:     Save .png image'
            y = self.plot_info_msg(self.screen_info, x, y, info_msg)
            info_msg = 'ESC:   Exit'
            y = self.plot_info_msg(self.screen_info, x, y, info_msg)
            y = 0
            x = 310
            pygame.draw.line(self.screen_info, self.info_color, (x - 10, y), (x - 10, y + 75))
            info_msg = 'Space:       Solve maze / next maze'
            y = self.plot_info_msg(self.screen_info, x, y, info_msg)
            info_msg = 'Cursor up:   Increase cell size'
            y = self.plot_info_msg(self.screen_info, x, y, info_msg)
            info_msg = 'Cursor down: Decrease cell size'
            y = self.plot_info_msg(self.screen_info, x, y, info_msg)
            info_msg = 'Mouse:       Select start & end'
            y = self.plot_info_msg(self.screen_info, x, y, info_msg)
            y = 0
            x = 700
            pygame.draw.line(self.screen_info, self.info_color, (x - 10, y), (x - 10, y + 75))
            self.last_message = phase_msg
            y = self.plot_info_msg(self.screen_info, x, y, phase_msg)
            info_msg = f'Maze size: {self.block_size[1]} x {self.block_size[0]} = {(self.block_size[1] * self.block_size[0])} cells'
            y = self.plot_info_msg(self.screen_info, x, y, info_msg)
            if phase_msg[:12] == 'Maze solved.':
                solution_cells = np.count_nonzero(self.blocks == 4)
                visited_cells = solution_cells + np.count_nonzero(self.blocks == 2)
                info_msg = f'Cell size: {self.cell_size:2d}, solution: {solution_cells} cells'
                y = self.plot_info_msg(self.screen_info, x, y, info_msg)
                info_msg = f'(Visited:  {visited_cells}, junctions: {self.junctions_used})'
                y = self.plot_info_msg(self.screen_info, x, y, info_msg)
            else:
                info_msg = f'Cell size: {self.cell_size:2d}'
                y = self.plot_info_msg(self.screen_info, x, y, info_msg)

            # copy to screen
            x = (self.screen_size[0] - self.screen_info.get_size()[0]) // 2
            self.screen.blit(self.screen_info, (x, 10))
            pygame.display.update((x, 10, self.screen_info.get_size()[0], self.screen_info.get_size()[1]))

    def plot_info_msg(self, screen, x, y, msg):

        f_screen = self.font.render(msg, False, self.info_color, self.background_color)
        f_screen.set_colorkey(self.background_color)
        screen.blit(f_screen, (x, y))
        return y + 16

    def toggle_fullscreen(self):

        # toggle between fullscreen and windowed mode
        screen_copy = self.screen.copy()
        pygame.display.toggle_fullscreen()
        self.screen.blit(screen_copy, (0, 0))
        pygame.display.flip()

    def save_image(self):

        # save maze as a png image. Use the first available number to avoid overwriting a previous image.
        for file_nr in range(1, 1000):
            file_name = 'Mazegif_' + ('00' + str(file_nr))[-3:] + '.png'
            if not exists(file_name):
                pygame.image.save(self.screen, file_name)
                break

    def pause(self):

        # wait for exit (window close or ESC key) or continue (space bar) or other user controls

        running = True
        pausing = True
        is_start_pos = True

        while pausing:

            event = pygame.event.wait()  # wait for user input, yielding to other processes

            if event.type == pygame.QUIT:
                pausing = False
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pausing = False
                if event.key == pygame.K_f:
                    self.toggle_fullscreen()
                if event.key == pygame.K_s:
                    # save screen as png image
                    self.save_image()
                if event.key == pygame.K_i:
                    self.toggle_info_display()
                if event.key == pygame.K_m:
                    self.toggle_slow_mode()
                if event.key == pygame.K_DOWN:
                    self.cell_size -= 1
                    if self.cell_size < 1:
                        self.cell_size = 1
                    pausing = False
                if event.key == pygame.K_UP:
                    self.cell_size += 1
                    if self.cell_size > min(self.rect[2], self.rect[3]) // 10:
                        self.cell_size = min(self.rect[2], self.rect[3]) // 10
                    pausing = False
                if event.key == pygame.K_ESCAPE:
                    pausing = False
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # left button: choose start and end position.
                    mouse_pos = np.asarray(pygame.mouse.get_pos())
                    cell = np.flip((mouse_pos - self.screen_block_offset) // (self.screen_block_size)).astype(np.int)
                    # make sure is within array
                    if cell[0] >= 1 and cell[1] >= 1 and cell[0] <= np.shape(self.blocks)[0] - 2 and cell[1] <= np.shape(self.blocks)[1] - 2:
                        # make sure is not in a wall
                        if self.blocks[cell[0], cell[1]] != 1:
                            # position is valid - change start or end position to it
                            if is_start_pos:
                                self.draw_cell(self.start_pos, self.start_pos, self.maze_color)  # previous start cell back to maze_solver color
                                self.start_pos = np.copy(cell)
                                self.draw_cell(cell, cell, self.start_color)  # new start cell to start color
                            else:
                                self.draw_cell(self.end_pos, self.end_pos, self.maze_color)  # previous end cell back to maze_solver color
                                self.end_pos = np.copy(cell)
                                self.draw_cell(cell, cell, self.end_color)  # new end cell to end color
                            is_start_pos = not(is_start_pos)  # alternate between changing start cell and end cell positions
                            pygame.display.flip()

        return running
        
class MazeSolverEnv:
    def __init__(self):
        # Generate and display the Maze. Then solve it.
        # Left mouse button: generate a new maze.
        # ESC or close the window: Quit.

        # set screen size and initialize it
        pygame.display.init()
        disp_size = (1920, 1080)
        disp_size = (640, 400)

        #==================================================
        #for_RL============================================
        #==================================================

        self.disp_size = (100, 100)
        self.num_celltype = 5 # corridor: 0, wall: 1, visited path: 2, unit: 3, goal: 4

        #==================================================

        info_display = False
        self.screen = pygame.display.set_mode(self.disp_size)
        pygame.display.set_caption('Maze Solver / KS 2022. Left Mouse Button to Continue.')
        self.running = True
        self.cell_size = 5

        # initialize maze solver with bogus data to make it available
        self.maze_solver = MazeSolver(self.screen, (0, 0, 100, 100), np.ones((1, 1)), np.array([1, 1]), np.array([1, 1]))
        self.maze_solver.info_display = info_display
        self.maze_solver.cell_size = self.cell_size  # cell size in pixels

        #while running:

        # intialize a maze, given size (y, x)
        self.rect = np.array([0, info_display * 80, self.disp_size[0], self.disp_size[1] - info_display * 80])  # the rect inside which to draw the maze.
        maze = maze_generator.Maze(self.rect[2] // (self.maze_solver.cell_size * 2) - 1, self.rect[3] // (self.maze_solver.cell_size * 2) - 1)
        maze.screen = self.screen  # if this is set, the maze generation process will be displayed in a window. Otherwise not.
        self.screen.fill((0, 0, 0))
        maze.screen_size = np.asarray(self.disp_size)
        maze.screen_block_size = np.min(self.rect[2:4] / np.flip(maze.block_size))
        maze.screen_block_offset = self.rect[0:2] + (self.rect[2:4] - maze.screen_block_size * np.flip(maze.block_size)) // 2
        maze.slow_mode = self.maze_solver.slow_mode

        self.blocks = maze.gen_maze_2D()
        
        # setting number of observations and actions
        # height, width
        self.num_obs = self.blocks.shape
        self.num_action = 4 # e, w, n, s

        start_pos = np.asarray(np.shape(self.blocks), dtype=np.int) - 2  # bottom right corner
        end_pos = np.array([1, 1], dtype=np.int)
        
        # for reset
        self.init_pos = start_pos

        self.maze_solver = MazeSolver(self.screen, self.rect, self.blocks, start_pos, end_pos)

        self.maze_solver.block_size = maze.block_size
        self.maze_solver.cell_size = self.cell_size
        self.maze_solver.plot_info('Generating a maze.')
        self.maze_solver.slow_mode = maze.slow_mode
        self.maze_solver.draw_cell(self.maze_solver.start_pos, self.maze_solver.start_pos, self.maze_solver.start_color)
        self.maze_solver.draw_cell(self.maze_solver.end_pos, self.maze_solver.end_pos, self.maze_solver.end_color)
        pygame.display.flip()
        
        # make path already visited blocks
        self.path_blocks = np.zeros_like(self.blocks)
        
        # make initial observation
        self.init_obs = self.get_obs(self.maze_solver.start_pos, self.maze_solver.end_pos)
        
    def step(self, action_idx):
        try:
            cell = np.copy(self.maze_solver.start_pos)

            # a simple definition of the four neighboring cells relative to current cell
            directions = np.array([
                [-1,  0],  # up
                [ 1,  0],  # down
                [ 0, -1],  # left
                [ 0,  1]   # right
                ], dtype=np.int16)

            cell_neighbors = np.hstack((
                cell + directions,
                np.sum((self.maze_solver.end_pos - cell) * directions, axis=1)[:, None]
                ))

            # pick the ones which are corridors and not visited yet
            valid_neighbors = cell_neighbors[(self.blocks[cell_neighbors[:, 0], cell_neighbors[:, 1]] == 0)]

            # random_valid_idx = np.random.choice(valid_neighbors.shape[0], 1)
            # move unit if valid action_idx accepted
            if cell_neighbors[action_idx].tolist() in valid_neighbors.tolist():
                self.maze_solver.next_pos = np.copy(cell_neighbors[action_idx][:2])
            else:
                self.maze_solver.next_pos = np.copy(self.maze_solver.start_pos)

            self.maze_solver.draw_cell(self.maze_solver.start_pos, self.maze_solver.start_pos, self.maze_solver.path_color)
            self.maze_solver.draw_cell(self.maze_solver.next_pos, self.maze_solver.next_pos, self.maze_solver.start_color)
            pygame.display.flip()

            # 여기에 아래와 같은 형태로 obs, action, reward, next_obs를 추가
            #===================================================
            #obs: [0.02880424 0.00342883 0.04124069 0.02668754]
            #obs type: <class 'numpy.ndarray'>
            #obs: [map_data with unit_pos and end_pos]

            #reward: 1.0
            #reward type: <class 'float'>
            #reward: -1 if not np.array_equal(self.start_pos, self.end_pos) else 0

            #action: [0]
            #action type: <class 'list'>
            # e -> 3, w -> 2, s -> 1, n -> 0

            #next_obs: [ 0.02887282 -0.19225954  0.04177444  0.33209184]
            #next_obs type: <class 'numpy.ndarray'>
            #nest_obs: [next_pos, end_pos, map_data, ...]
            #===================================================

            # remember the paths already visited
            self.path_blocks[self.maze_solver.start_pos[0]][self.maze_solver.start_pos[1]] = 1
            
            # create action
            action = action_idx

            # create reward
            reward = -0.04 if not np.array_equal(self.maze_solver.next_pos, self.maze_solver.end_pos) else 1
            # hit the wall
            if np.array_equal(self.maze_solver.start_pos, self.maze_solver.next_pos):
                reward -= 0.75
            elif self.path_blocks[self.maze_solver.next_pos[0]][self.maze_solver.next_pos[1]] == 1:
                reward -= 0.25
                
            # create done
            done = False if not np.array_equal(self.maze_solver.next_pos, self.maze_solver.end_pos) else True
            
            # create obs, next_obs
            obs = self.get_obs(cell, self.maze_solver.end_pos)
            next_obs = self.get_obs(self.maze_solver.next_pos, self.maze_solver.end_pos)
            
            '''
            obs = self.stacked_obs
            next_obs = self.get_obs(self.maze_solver.next_pos, self.maze_solver.end_pos)
            next_obs = self.get_stacked_obs(next_obs) # stack new obs in stacked_obs
            '''
            
            self.maze_solver.start_pos = np.copy(self.maze_solver.next_pos)

            # .reshape((1,) + obs.shape) set the batch size
            return next_obs, reward, done, obs
        except:
            print("==========================================")
            print("ERROR_OCCURED_01")
            print("actions must be integer number")
            print("actions are in between 0 to 3")
            print("==========================================")
            
    def reset(self, exploring_starts = False, random_goal = False):
        corridor_idxs = np.argwhere(self.blocks == 0) # 0's are corridor
        
        for corridor_idx in corridor_idxs:
            self.maze_solver.draw_cell(corridor_idx, corridor_idx, self.maze_solver.maze_color)
            
        #==========================================================================
        # exploring starts and random goal=========================================
        #==========================================================================
        
        if exploring_starts:
            # pick randomly from corridor indexes list
            start_pos = corridor_idxs[np.random.randint(corridor_idxs.shape[0], size=1)[0], :]
        else:
            start_pos = self.init_pos
        
        if random_goal:
            # pick randomly from corridor indexes list
            end_pos = corridor_idxs[np.random.randint(corridor_idxs.shape[0], size=1)[0], :]
            while np.array_equal(end_pos, start_pos):
                end_pos = corridor_idxs[np.random.randint(corridor_idxs.shape[0], size=1)[0], :]
        else:
            end_pos = np.array([1, 1], dtype=np.int)
            
        #==========================================================================
        
        self.maze_solver.start_pos = start_pos
        self.maze_solver.end_pos = end_pos
        self.maze_solver.draw_cell(self.maze_solver.start_pos, self.maze_solver.start_pos, self.maze_solver.start_color)
        self.maze_solver.draw_cell(self.maze_solver.end_pos, self.maze_solver.end_pos, self.maze_solver.end_color)
        pygame.display.flip()

        return self.init_obs 
    
    def reset_maze(self, exploring_starts = False, random_goal = False):
        # intialize a maze, given size (y, x)
        maze = maze_generator.Maze(self.rect[2] // (self.cell_size * 2) - 1, self.rect[3] // (self.cell_size * 2) - 1)
        maze.screen = self.screen  # if this is set, the maze generation process will be displayed in a window. Otherwise not.
        self.screen.fill((0, 0, 0))
        maze.screen_size = np.asarray(self.disp_size)
        maze.screen_block_size = np.min(self.rect[2:4] / np.flip(maze.block_size))
        maze.screen_block_offset = self.rect[0:2] + (self.rect[2:4] - maze.screen_block_size * np.flip(maze.block_size)) // 2
        maze.slow_mode = self.maze_solver.slow_mode

        self.blocks = maze.gen_maze_2D()
        
        #==========================================================================
        # exploring starts and random goal=========================================
        #==========================================================================
        
        corridor_idxs = np.argwhere(self.blocks == 0) # 0's are corridor
        
        if exploring_starts:
            # pick randomly from corridor indexes list
            start_pos = corridor_idxs[np.random.randint(corridor_idxs.shape[0], size=1)[0], :]
        else:
            start_pos = np.asarray(np.shape(self.blocks), dtype=np.int) - 2  # bottom right corner
        
        if random_goal:
            # pick randomly from corridor indexes list
            end_pos = corridor_idxs[np.random.randint(corridor_idxs.shape[0], size=1)[0], :]
            while np.array_equal(end_pos, start_pos):
                end_pos = corridor_idxs[np.random.randint(corridor_idxs.shape[0], size=1)[0], :]
        else:
            end_pos = np.array([1, 1], dtype=np.int)
            
        #==========================================================================

        self.maze_solver = MazeSolver(self.screen, self.rect, self.blocks, start_pos, end_pos)

        self.maze_solver.block_size = maze.block_size
        self.maze_solver.cell_size = self.cell_size
        self.maze_solver.plot_info('Generating a maze.')
        self.maze_solver.slow_mode = maze.slow_mode
        self.maze_solver.draw_cell(self.maze_solver.start_pos, self.maze_solver.start_pos, self.maze_solver.start_color)
        self.maze_solver.draw_cell(self.maze_solver.end_pos, self.maze_solver.end_pos, self.maze_solver.end_color)
        pygame.display.flip()
        
        # make initial observation
        self.init_obs = self.get_obs(self.maze_solver.start_pos, self.maze_solver.end_pos)
    
    def get_obs(self, start_pos, end_pos):
        
        obs = np.copy(self.blocks)
        
        for idx in np.argwhere(self.path_blocks == 1):
            obs[tuple(idx)] = 2
        
        obs[start_pos[0]][start_pos[1]] = 3 # unit_pos index as 3
        obs[end_pos[0]][end_pos[1]] = 4 # end_pos index as 4
        
        # one hot encoding (1 channel to 5 channel)
        obs = np.array([([channel_num] == obs[..., None]).astype(int).reshape(obs.shape[0], obs.shape[1]) for channel_num in range(0, self.num_celltype)], dtype=np.float32)
        
        return obs
    
    def init_stacked_obs(self, channel, state):
        self.stacked_obs = np.zeros((channel,) + state.shape, dtype = np.float32)
        self.stacked_obs[-1] = np.copy(state)
        self.num_obs_channel = channel
        
        return np.copy(self.stacked_obs).reshape((1,) + self.stacked_obs.shape)
    
    def get_stacked_obs(self, state):
        self.stacked_obs[:self.num_obs_channel - 1] = np.copy(self.stacked_obs[1: self.num_obs_channel])
        self.stacked_obs[-1] = np.copy(state)
        
        return np.copy(self.stacked_obs).reshape((1,) + self.stacked_obs.shape)
    
    def get_ann_obs(self, start_pos, end_pos):
        pos_obs = np.append(np.array(list(start_pos)), np.array(list(end_pos)))
        map_obs = np.copy(self.blocks).flatten()
        
        return np.array([np.append(pos_obs, map_obs)]).astype(np.float32)

    def close(self):
        pygame.quit()
