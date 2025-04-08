import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import random
import math

class BrownianRobot:
    def __init__(self, arena_size=10.0, speed=1.5, radius=0.2):
        self.arena_size = arena_size
        self.position = np.array([arena_size/2, arena_size/2])
        self.direction = random.uniform(0, 2*math.pi)  # Random initial direction
        self.speed = speed
        self.radius = radius
        self.collision_count = 0
        self.is_rotating = False
        self.rotation_duration = 0
        self.rotation_time = 0
        self.rotation_speed = random.uniform(0.5, 1.5)  # Radians per time step
        
    def update(self, dt=1.0):
        if self.is_rotating:
            # Perform rotation
            self.direction += self.rotation_speed * dt
            self.rotation_time += dt
            
            # Check if rotation is complete
            if self.rotation_time >= self.rotation_duration:
                self.is_rotating = False
        else:
            # Move forward
            new_position = self.position + np.array([
                math.cos(self.direction) * self.speed * dt,
                math.sin(self.direction) * self.speed * dt
            ])
            
            # Check for boundary collisions
            if (new_position[0] - self.radius < 0 or 
                new_position[0] + self.radius > self.arena_size or
                new_position[1] - self.radius < 0 or 
                new_position[1] + self.radius > self.arena_size):
                
                # Start rotating
                self.is_rotating = True
                self.rotation_duration = random.uniform(0.5, 2.0)  # Random rotation time
                self.rotation_time = 0
                self.rotation_speed = random.uniform(-1.5, 1.5)  # Random rotation direction/speed
                self.collision_count += 1
            else:
                self.position = new_position
    
    def get_state(self):
        """Return the current state of the robot."""
        return {
            'position': self.position.copy(),
            'direction': self.direction,
            'collisions': self.collision_count,
            'is_rotating': self.is_rotating
        }


def simulate_brownian_robot(arena_size=10.0, speed=1.5, radius=0.2, duration=60, fps=30):
    """
    Simulate and animate the Brownian robot.
    """
    robot = BrownianRobot(arena_size, speed, radius)
    dt = 1.0 / fps
    total_frames = duration * fps
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, arena_size)
    ax.set_ylim(0, arena_size)
    ax.set_aspect('equal')
    ax.set_title('Brownian Motion Robot Simulation')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    
    # Create robot representation
    robot_circle = Circle((robot.position[0], robot.position[1]), radius, 
                         fc='blue', ec='black')
    ax.add_patch(robot_circle)
    
    # Create direction indicator
    direction_line = ax.plot([], [], 'r-', linewidth=2)[0]
    
    # Text annotation for collision count
    collision_text = ax.text(0.02, 0.95, 'Collisions: 0', transform=ax.transAxes)
    
    def init():
        """Initialize the animation."""
        robot_circle.center = (robot.position[0], robot.position[1])
        direction_line.set_data(
            [robot.position[0], robot.position[0] + math.cos(robot.direction) * radius * 1.5],
            [robot.position[1], robot.position[1] + math.sin(robot.direction) * radius * 1.5]
        )
        return robot_circle, direction_line, collision_text
    
    def animate(i):
        """Update the animation frame."""
        robot.update(dt)
        
        # Update robot position and direction
        robot_circle.center = (robot.position[0], robot.position[1])
        direction_line.set_data(
            [robot.position[0], robot.position[0] + math.cos(robot.direction) * radius * 1.5],
            [robot.position[1], robot.position[1] + math.sin(robot.direction) * radius * 1.5]
        )
        
        # Change color if rotating
        if robot.is_rotating:
            robot_circle.set_facecolor('orange')
        else:
            robot_circle.set_facecolor('blue')
        
        # Update collision count
        collision_text.set_text(f'Collisions: {robot.collision_count}')
        
        return robot_circle, direction_line, collision_text
    
    # Create and show the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=total_frames, init_func=init,
        blit=True, interval=1000/fps, repeat=False
    )
    
    plt.tight_layout()
    plt.show()
    
    # To save the animation, uncomment the following line:
    # ani.save('brownian_robot.mp4', writer='ffmpeg', fps=fps)


if __name__ == "__main__":
    # Run a sample simulation when executed directly
    simulate_brownian_robot(duration=30)