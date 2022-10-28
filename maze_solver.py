# This code will test a maze to see if it has a solution, when generated UED it needs to be sure to at least have a solution
# Remind from our maze that 1 is the current position, 2 is a wall, 3 is the ending, (4 is a danger)
import numpy as np

def Maze_solver(maze):
    """Function to find if a maze has a solution, if yes it will return True, if not it will return False

    Parameters
    ----------
    maze : np 2D array

        
    """
    # Start from the end
    position_x, position_y = np.where(maze == 3.0)[0][0], np.where(maze == 3.0)[1][0]
    start_position_x, start_position_y = np.where(maze == 1.0)[0][0], np.where(maze == 1.0)[1][0]


    def is_solvable(maze, start_position=(int(start_position_x),int(start_position_y))):
        seen = set([start_position])
        queue = [start_position]
        while queue:
            i,j = queue.pop(0)
            seen.add((i,j))
            for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni,nj = i+di, j+dj
                if (ni,nj) in seen:
                    continue
                if ni<0 or nj<0 or ni>=len(maze) or nj>=len(maze[0]):
                    continue
                if maze[ni][nj] == 3.0:
                    return True
                if maze[ni][nj] == 2.0:
                    continue
                if maze[ni][nj] == 0.0 or maze[ni][nj] == 6.0:
                    seen.add((ni,nj))
                    queue.append((ni,nj))
        return False

    return is_solvable(list(maze),start_position=(start_position_x, start_position_y))

    