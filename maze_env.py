#%%
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import torch
import copy
from maze_solver import Maze_solver

class Maze(object):
    # This class contains all the functions to implement the maze environement
    # the agent will have to reach the ending position given a starting position

    def __init__(self,height,width,walls,traps) -> None:
        super().__init__()
        self.Height = height+2
        self.Width = width+2
        self.maze = np.zeros((height+2,width+2),dtype=np.dtype('d'))
        self.maze[:,0] = 2
        self.maze[:,-1] = 2
        self.maze[0,:] = 2
        self.maze[-1,:] =2
        self.initial_position = [1,1]
        # If we want random ening point 
            # self.final_position = [np.random.randint(height, size=1)[0]+1,np.random.randint(width, size=1)[0]+1]
        self.final_position = [height,width]
        self.done = False
        # Ensure the end of the maze isn't positioned at the starting position
        while (self.initial_position == self.final_position):
            self.final_position = [np.random.randint(height, size=1)[0]+1,np.random.randint(width, size=1)[0]+1]
        
        # Place the beginning and ending positions of the maze
        self.maze[self.final_position[0],self.final_position[1]] = 3
        self.maze[self.initial_position[0],self.initial_position[1]] = 1
        self.wall = 0
        self.traps =0
        self.steps =0
        while self.wall in range(walls):
            coordinate_height = np.random.randint(height, size=1)+1
            coordinate_width = np.random.randint(width, size=1)+1
            if self.maze[coordinate_height,coordinate_width] == 0:
                self.maze[coordinate_height,coordinate_width] = 2.0
                self.wall +=1
        while self.traps in range(traps):
            coordinate_height = np.random.randint(height, size=1)+1
            coordinate_width = np.random.randint(width, size=1)+1
            if self.maze[coordinate_height,coordinate_width] == 0:
                self.maze[coordinate_height,coordinate_width] = 6.0
                self.traps +=1
        self.original_maze = copy.copy(self.maze)
        self.position_traps = np.argwhere(self.maze==6.0)
        #Check if maze is doable
        if not Maze_solver(self.maze):
            self.restart_maze(walls,traps,height,width)
        self.render_maze()

    def restart_maze(self,walls,traps,height,width):
        """This function will reset a maze so it's doable

        """
        self.maze = np.zeros((height+2,width+2),dtype=np.dtype('d'))
        self.maze[:,0] = 2
        self.maze[:,-1] = 2
        self.maze[0,:] = 2
        self.maze[-1,:] =2
        # Place the beginning and ending positions of the maze
        self.maze[self.final_position[0],self.final_position[1]] = 3
        self.maze[self.initial_position[0],self.initial_position[1]] = 1
        self.wall = 0
        self.traps =0
        self.steps =0
        while self.wall in range(walls):
            coordinate_height = np.random.randint(height, size=1)+1
            coordinate_width = np.random.randint(width, size=1)+1
            if self.maze[coordinate_height,coordinate_width] == 0:
                self.maze[coordinate_height,coordinate_width] = 2.0
                self.wall +=1
        while self.traps in range(traps):
            coordinate_height = np.random.randint(height, size=1)+1
            coordinate_width = np.random.randint(width, size=1)+1
            if self.maze[coordinate_height,coordinate_width] == 0:
                self.maze[coordinate_height,coordinate_width] = 6.0
                self.traps +=1
        self.original_maze = copy.copy(self.maze)
        self.position_traps = np.argwhere(self.maze==6.0)

        if not Maze_solver(self.maze):
            self.restart_maze(walls,traps,height, width)
        else:
            self.render_maze()
        
    def set_maze(self, new_maze):
        """This function will set the original maze to a certain value.
            It needs to fit the dimenssions of the maze when instantiated

        Parameters
        ----------
        maze : numpy array

        """
        self.original_maze = new_maze
        self.maze = new_maze
        self.reset()
        self.position_traps = np.argwhere(self.maze==6.0)


    def reset(self,Steps = True):
        """Reset the maze by putting agent at beginning of it
        """
        self.maze = copy.copy(self.original_maze)
        self.done = False
        if Steps == True:
            self.steps = 0
        
    def maze_state(self):
        """Return the state of the maze

        Returns
        -------
        maze
            np 2D array
        """
        return self.maze
        
    def render_maze(self) -> None:
        """This function will simply print the current state of the maze in the terminal
        """
        self.vizualize_maze = self.maze.astype(str)
        self.vizualize_maze = np.where(self.vizualize_maze == "1.0","o",self.vizualize_maze)
        self.vizualize_maze= np.where(self.vizualize_maze == "2.0","▧",self.vizualize_maze)
        self.vizualize_maze= np.where(self.vizualize_maze == "0.0"," ",self.vizualize_maze)
        self.vizualize_maze= np.where(self.vizualize_maze == "3.0","⍜",self.vizualize_maze)
        self.vizualize_maze= np.where(self.vizualize_maze == "6.0","⁜",self.vizualize_maze)
        print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
            for row in self.vizualize_maze.tolist()]))


    def reward(self):
        """This function will return the reward.

            Returns
            -------
            reward_val
                Reward of the position, computed as:
                1 if he reaches the end point, the game should then report done
                -0.1/(H x W) if he doesn't reach it on the next step
            done
                Bool to say if the game is over
        """
        self.player_position = np.argwhere(self.maze==1.0).reshape(2)



        if np.array_equal(self.player_position,self.final_position):
                reward_val = 1
                self.done = True

        elif (self.position_traps[:, None] == self.player_position).all(-1).any(-1).any():
            reward_val = (- 0.1/(len(self.maze[0])*len(self.maze[:,0]))) *2

        else :
            reward_val = -0.1/(len(self.maze[0])*len(self.maze[:,0]))

        return reward_val

    def make_move(self,move):
        """This function will check if a move is possible and execute it if it can

        Parameters
        ----------
        move : int
            Move: 0 -> up, 1 -> right , 2 -> down, 3 -> left

        Returns
        -------
        Changes the state of the maze, to take into account the move done
        """
        # Find the positon of the agent
        self.player_position = np.argwhere(self.maze==1.0).reshape(2)
        #print("Position is:" + str(self.player_position))
        
        # Depending on the move, check if he can do it, and if so execute it
        if move == 0 and (self.player_position[0] > 0):
            if self.maze[self.player_position[0]-1,self.player_position[1]] != 2:
                self.maze[self.player_position[0],self.player_position[1]] = 0
                self.maze[self.player_position[0]-1,self.player_position[1]] = 1
                # If the previous position was a trap, reset the trap
                if (self.position_traps[:, None] == self.player_position).all(-1).any(-1).any():
                    self.maze[self.player_position[0],self.player_position[1]] = 6.0
        elif move == 1 and (self.player_position[1] < (self.Width-1)):
            if self.maze[self.player_position[0],self.player_position[1]+1] != 2:
                self.maze[self.player_position[0],self.player_position[1]] = 0
                self.maze[self.player_position[0],self.player_position[1]+1] = 1
                # If the previous position was a trap, reset the trap
                if (self.position_traps[:, None] == self.player_position).all(-1).any(-1).any():
                    self.maze[self.player_position[0],self.player_position[1]] = 6.0
        elif move == 2 and (self.player_position[0] < (self.Height-1)):
            if self.maze[self.player_position[0]+1,self.player_position[1]] != 2:
                self.maze[self.player_position[0],self.player_position[1]] = 0
                self.maze[self.player_position[0]+1,self.player_position[1]] = 1
                # If the previous position was a trap, reset the trap
                if (self.position_traps[:, None] == self.player_position).all(-1).any(-1).any():
                    self.maze[self.player_position[0],self.player_position[1]] = 6.0
        elif move == 3 and (self.player_position[1] > 0):
            if self.maze[self.player_position[0],self.player_position[1]-1] != 2:
                self.maze[self.player_position[0],self.player_position[1]] = 0
                self.maze[self.player_position[0],self.player_position[1]-1] = 1
                # If the previous position was a trap, reset the trap
                if (self.position_traps[:, None] == self.player_position).all(-1).any(-1).any():
                    self.maze[self.player_position[0],self.player_position[1]] = 6.0

        
        

        self.steps +=1
        reward_val = self.reward()
        #(self.Height*self.Width)*1.5

        if self.steps >200:
            self.done = True
            
        return self.maze,reward_val,self.done


if __name__=="__main__":   
    maze = Maze(5,5,3,traps=4)
    print("Starting Position:")
    """new_maze = [[2.0,2.0,2.0,2.0,2.0,2.0,2.0],
                            [2.0,1.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,6.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,6.0,0.0,0.0,2.0,3.0,2.0],
                            [2.0,2.0,2.0,2.0,2.0,2.0,2.0]]
    maze.set_maze(np.array(new_maze))"""
    for i in range(10):
        move = np.random.randint(4,size=1)[0]
        print("Selected move is:"+str(move))
        _,reward,done =maze.make_move(move)
        print("Game done:"+str(done))
        print("Reward:"+str(reward))
        maze.render_maze()
        print("\n")
        if done:
            print("Finished Puzzle")
            break
    maze.reset()

    maze.render_maze()

#%%