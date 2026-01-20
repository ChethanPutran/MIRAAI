import numpy as np 
import rclpy 
from rclpy.node import Node 
from std_msgs.msg import String 
from your_robot_pkg.srv import ExecuteTask


# Motion planning
"""
    1. Sampling based planners
        - RRT (Rapidly exploring random tree)
        - RRT*
        - PRM (Probabilistic Roadmap)
    2. Search based planners
        - A*
        - Dijkstra's
        - Greedy
    3. Optimization based planners
        - CHOMP (Covariant Hamiltonian Optimization for MP)
        - TrajOpt / STOMP / GPMP
"""


# RRT *
"""
1. Configuration Space
2. Sampling
3. Nearest Node
4. Steer
5. Collision check
6. Add node
7. Rewire
8. Goal reached
9. Smooth trajectory
"""
# -----------------------------

# Trajectory Smoothing

# -----------------------------

from scipy.interpolate import make_interp_spline

def smooth_trajectory(path, smooth_factor=0.1,use_spline=True,num_points=100):

    if use_spline:
        trajectory = np.array(path)
        n_points = trajectory.shape[1]
        t = np.linspace(0,1,len(trajectory))

        smoothed = []

        for i in range(n_points):
            spline = make_interp_spline(t,trajectory[:,i],k=3) # Cubic B-spline
            t_new = np.linspace(0,1,num_points)
            smoothed.append(spline(t_new))

        smoothed_trajectory = np.stack(smoothed,axis=1)
        return smoothed_trajectory.tolist()

    else:
        smoothed_path = [path[0]] 
        for i in range(1, len(path)-1): 
            prev_point = np.array(path[i-1])
            curr_point = np.array(path[i]) 
            next_point = np.array(path[i+1]) 
            smoothed_point = ((prev_point + curr_point + next_point) / 3.0).tolist() 
            smoothed_path.append(smoothed_point) 
            smoothed_path.append(path[-1]) 
        return smoothed_path


class Node_:
    def __init__(self,config,parent=None,cost=0.0):
        self.config = np.array(config)
        self.parent = parent
        self.cost = cost

class RRTStar:
    # 1. Configuration Space
    def __init__(self,start_config,goal_config,joint_limits,step_size=0.1):
        self.q_start = Node_(start_config)
        self.q_goal = Node_(goal_config)
        self.joint_limits = joint_limits # [(min,max),...]
        self.nodes = [self.q_start]
        self.obstacles = []
        self.step_size = step_size

    def set_objects(self, obstacles):
        self.obstacles = obstacles

    # 2. Sampling the nearest node
    def sample_config(self,goal_sample_rate=0.1):
        if np.random.rand()<goal_sample_rate:
            return self.q_goal.config
        return np.array([np.random.uniform(low,high) for low,high in self.joint_limits])
    
    def nearest_node(self,sample):
        distances = [np.linalg.norm(node.config - sample) for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    # 4. Steer
    def steer(self,from_node:Node_,to_config):
        direction = to_config - from_node.config
        length = np.linalg.norm(direction)
        if length == 0:
            return from_node.config
        direction = direction / length
        new_config = from_node.config + self.step_size*direction

        # Conform the joint value lies with in limits
        return np.clip(new_config,[l[0] for l in self.joint_limits],[l[1] for l in self.joint_limits])
    
    # 5. Collision check
    def is_collision_free(self,config): 
        def check_segment_sphere_collision(p1,p2,sphere_center,sphere_radius):
            # closest point on segment t sphere center
            v = np.array(p2) - np.array(p1)
            w = np.array(sphere_center)-np.array(1)
            t = np.dot(w,v)/np.dot(v,v)
            t = max(0,min(1,t)) # clamp t to [0,1]
            closest_point = np.array(p1)+t*v
            distance = np.linalg.norm(closest_point-np.array(sphere_center))
            return distance < sphere_radius
        
        for i in range(len(config)):
            p1 = config[i]
            p2 = config[i+1]

            for obs in self.obstacles:
                if check_segment_sphere_collision(p1,p2,obs['position'],obs['radius']):
                    return False
        return True
    
    # Main planning loop
    def plan(self,max_iters=100):
        for _ in range(max_iters):
            sample = self.sample_config()
            nearest = self.nearest_node(sample)
            new_config = self.steer(nearest,sample)

            if self.is_collision_free(new_config):
                new_node = Node(new_config,parent=nearest,cost=nearest.cost+np.linalg.norm(new_config-nearest.config))
                self.nodes.append(new_node)

            for node in self.nodes:
                if np.linalg.norm(node.config-new_node.config) < self.step_size*2:
                    new_cost = new_node.cost + np.linalg.norm(node.config - new_node.config)
                    if new_cost< node.cost and self.is_collision_free(node.config):
                        node.parent = new_node
                        node.cost = new_cost

                        if np.linalg.norm(new_node.config - self.q_goal.config) < self.step_size:
                            self.q_goal.parent = new_node
                            return self.retrace_path(self.goal)
        # Planning failed
        return None
    
    def retrace_path(self,node):
        path = []
        while node is not None:
            path.append(node.config)
            node = node.parent
        return path[::-1]
# -----------------------------

# Robot Class with FK and Collision Checking

# -----------------------------

class Robot: 
    def init(self, dh_params): self.dh_params = dh_params

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        for i, theta in enumerate(joint_angles):
            a = self.dh_params[i][1]
            d = self.dh_params[i][2]
            alpha = self.dh_params[i][3]

            T_i = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])
            T = T @ T_i
        return T[:3, 3]

    def is_collision_free(self, joint_angles, obstacles):
        pos = self.forward_kinematics(joint_angles)
        for obs in obstacles:
            if self.check_collision_with_obstacle(pos, obs):
                return False
        return True

    def check_collision_with_obstacle(self, pos, obstacle):
        distance = np.linalg.norm(pos - obstacle["position"])
        return distance < obstacle["radius"]



# -----------------------------

# Placeholder IK Solver

# -----------------------------

def ik_solver(position, orientation): 
    # Replace this with your actual IK function 
    return np.random.uniform(-1.0, 1.0, size=6).tolist() # Dummy joint angles

# -----------------------------

# Placeholder RRT* Planner

# -----------------------------



# -----------------------------

# ROS2 Task Executor Node

# -----------------------------
class Stereo:
    @ staticmethod
    def get_obstacles(self):
        pass

class TaskExecutorNode(Node): 
    def init(self): 
        super().init('task_executor_node') 
        self.srv = self.create_service(ExecuteTask, 'execute_task', self.execute_task_callback) 
        # Replace with your actual DH params 
        self.robot = Robot(dh_params=[[0, 0, 0.1519, np.pi/2], 
                                      [0, -0.24365, 0, 0], 
                                      [0, -0.21325, 0, 0],
                                        [0, 0, 0.11235, np.pi/2], 
                                        [0, 0, 0.08535, -np.pi/2], 
                                        [0, 0, 0.0819, 0] ])

    def execute_task_callback(self, request, response):
        task = eval(request.task_data)
        action = task["action"]
        params = task["parameters"]

        if action == "move":
            from_pos = params["from"]
            to_pos = params["to"]
            orientation = params["orientation"]

            q_start = ik_solver(from_pos, orientation)
            q_goal = ik_solver(to_pos, orientation)

            planner = RRTStar(q_start, q_goal, joint_limits=[(-3.14, 3.14)]*6)

            obstacle_list = Stereo.get_obstacles()
            # obstacle_list = [
            #     {"position": np.array([0.2, 0.2, 0.2]), "radius": 0.05},
            #     {"position": np.array([0.4, 0.3, 0.1]), "radius": 0.07}
            # ]
            planner.set_objects(obstacle_list)
            path = planner.plan()

            smooth_path = smooth_trajectory(path)
            self.send_to_joint_controller(smooth_path)

            response.status = "Task Completed"
            return response

    def send_to_joint_controller(self, smooth_path):
        for joint_angles in smooth_path:
            self.get_logger().info(f"Sending joint angles: {joint_angles}")
            # Send to controller here (e.g., via ROS2 topic or service)

    # -----------------------------

    # Main Function

    # -----------------------------

def main(args=None): 

    task = {
        "action":"move",
        "parameters": {
            "from":[0,0,0],
            "to":[1,1,1],
            "orientation":[1,0.2,0.2,0.2] # Quaternion
        }
    }
    rclpy.init(args=args) 
    node = TaskExecutorNode() 
    rclpy.spin(node) 
    node.destroy_node() 
    rclpy.shutdown()

if __name__ == 'main': 
    main()