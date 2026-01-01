import re
import json
import torch
import numpy as np
from LMP import LMP
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from utils import bytes_to_torch, torch_to_bytes
from scipy.spatial.transform import Rotation as R
ANGLE_OBJECT_INFO = {}
ANGLE_WORLD_INFO = {"erlenmeyer_flask": [0, 0, 0], "magnetic_stirrer": [0, 0, 0]}
class LMP_interface():

    def __init__(self, env, visualizer):
        self._env = env
        self.visualizer = visualizer

        self.fixed_vars = {'np': np}
        self.variable_vars = {
            k: getattr(self, k)
            for k in dir(self) if callable(getattr(self, k)) and not k.startswith("_")
        }
        self.variable_vars.update(self.fixed_vars)
        
        
    # ======================================================
    # == functions exposed to LLM
    # ======================================================

    def get_angle(self, object_name, axis):
        """
        Get the angle of an object on a specified axis.

        Args:
            object_name: Object name as a string.
            axis: Axis to query ('x', 'y', or 'z').

        Returns:
            int: Angle value on the specified axis.
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        
        if object_name not in ANGLE_OBJECT_INFO:
            raise ValueError(f"Object '{object_name}' not found in ANGLE_OBJECT_INFO")
        
        if axis not in axis_map:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        
        return ANGLE_OBJECT_INFO[object_name][axis_map[axis]]
    
    def mllm_inference(self, query="", type="extract", objects=[None],  params=[None]):
        """Extract or refine affordance using MLLM inference."""
        return self._env.vision_infer.mllm_inference(query=query, type=type, objects=objects, params=params)

    def get_2d_coords(self, extracted_anchor):
        """Get 2D coordinates of anchors."""
        return self._env.get_2d_coords(extracted_anchor)

    def map_affordance(self, anchor_coords, objects):
        """
        Check if anchor_coords keys match the objects list.

        Args:
            anchor_coords (dict): Anchors by object name.
            objects (list): Expected object names.

        Returns:
            bool: True if keys match objects exactly.
        """
        self.anchor_coords = anchor_coords
        return list(anchor_coords.keys()) == objects
    
    def register_save_anchors(self, anchor_state):
        """Register and save anchor state."""
        register_anchor_state = self._env.register_anchor(anchor_state)
        self.anchor_state = self._env.get_movable_anchor_pos(register_anchor_state)
        
    def transform_to_label(self, anchor_coords):
        """Map anchor coordinates to label."""
        return self._env.transform_to_label(anchor_coords)

    def transform_to_coords(self, label):
        """Map label to anchor coordinates."""
        return self._env.transform_to_coords(label)
    
    def parse_object_state(self, object):
        """
        Retrieve anchor state for a given object.

        Args:
            object (str): Object name.

        Returns:
            dict: Anchor state of the object.
        """
        return self.anchor_state[object]
    
    def parse_anchor_coords(self, object):
        """
        Get 2D anchor coordinates for a specified object.

        Args:
            object (str): Object name.

        Returns:
            list or dict: 2D anchor coordinates for the object.
        """
        if object not in self.anchor_coords:
            raise KeyError(f"Object '{object}' not found in keypoints.")
        return self.anchor_coords[object]
    
    def parse_object_pose(self, object):
        """
        Get the relative 3D position of an object.

        Args:
            object: Object name.

        Returns:
            np.ndarray: Object position relative to the robot's initial position.
        """
        if object == "workbench":
            object_pose = np.array([0.15, 1.27, 0.91])
        else:
            object_pose = self._env.get_object_pose(object)
            
        return object_pose - self._env.robot_initial_position
    
    def merge_anchor_state(self, anchor_state, refined_anchor, objects=None, type=""):
        """
        Merge refined anchor coordinates into the anchor state.

        Args:
            anchor_state: Existing anchor state dictionary.
            refined_anchor: List of refined anchor coordinates (as arrays).
            objects: List of object names. Defaults to [""] if None.
            type: Anchor type key (e.g., "refined").

        Returns:
            Updated anchor state with new anchor data merged in.
        """
        if objects is None:
            objects = [""]

        for obj, anchor_list in zip(objects, refined_anchor):
            anchor_state.setdefault(obj, {}).setdefault(type, {}).setdefault('init', [])
            anchor_state[obj][type]['init'].append(anchor_list.flatten().tolist()) 

        return anchor_state
    
    def show_pts_points(self, points):
        """
        Visualizes a set of 3D points in the scene points.

        Args:
            points (np.ndarray): Array of 3D points in robot-relative coordinates.
        """
        points = points + self._env.robot_initial_position
        self.visualizer.visualize_points(points)
        
    def get_mask_edge_coords(self, anchor_coords, offset=0):
        """
        Positional Refinement Flow.
            Step 1: Extraction of 3D Coordinates Along Mask Edges.

        Args:
            anchor_coords: [x, y, mask_index] — center pixel and mask index.
            offset: Distance to push edge points away from the center.

        Returns:
            List of (x, y) edge coordinates.
        """
        mask_idx = int(anchor_coords[0][2])
        cx, cy = map(int, anchor_coords[0][:2])

        mask = self._env.clustered_ann[mask_idx].cpu().numpy().astype(bool)
        height, width = mask.shape

        edge_coords = [(cx, cy)]  # include center

        for y in range(height):
            for x in range(width):
                if not mask[y, x]:
                    continue
                # Check if current pixel borders the background
                if (x > 0 and not mask[y, x - 1]) or (x < width - 1 and not mask[y, x + 1]) or \
                (y > 0 and not mask[y - 1, x]) or (y < height - 1 and not mask[y + 1, x]):

                    dx = offset if x > cx else -offset if x < cx else 0
                    dy = offset if y > cy else -offset if y < cy else 0

                    new_x = min(max(x + dx, 0), width - 1)
                    new_y = min(max(y + dy, 0), height - 1)

                    edge_coords.append((new_x, new_y))

        return edge_coords
        
    def pos_anchor_refine(self, coords):
        """
        Positional Refinement Flow.
            Step 2: Density Peak Estimation and Maximum Height Filtering.
            Step 3: Centrally Symmetric Point Pairs as Affordance
        Args:
            coords: 2D coordinates used to locate relevant 3D anchor points.

        Returns:
            np.ndarray: A single 3D point representing the refined anchor location.
        """
        # Get point cloud from VFM camera
        point_cloud = self._env.cams[0].get_obs()['points']
        
        # Extract 3D points and corresponding 2D coords
        points_3d, coords_2d = self._extract_coordinates(coords, point_cloud)
        
        # Normalize 3D points relative to robot base
        points_3d = self._normalize_coordinates(points_3d)
        
        # Visualize
        if self._env.visualize:
            self.show_pts_points(points_3d)
        
        # Step 2: Density Peak Estimation
        peak_z, filtered_points_3d, filtered_coords_2d = self._filter_points_by_density(
            points_3d, coords_2d, mode=1
        )

        # Step 2: Maximum Height Filtering
        filtered_points_3d, filtered_coords_2d = self._filter_by_max_z(filtered_points_3d, filtered_coords_2d, threshold=0.02)
        
        # Step 3: Targets.
        refined_pos_anchor, symmetric_pair_3d = self._find_most_symmetric_pair(
            filtered_coords_2d, filtered_points_3d, coords[0]
        )
        
        # Visualize
        if self._env.visualize:
            self.show_pts_points(np.concatenate((refined_pos_anchor.reshape(1, -1), symmetric_pair_3d), axis=0))
        
        return refined_pos_anchor
    
    def convert_2d_to_3d(self, coords):
        """
        Convert 2D coords to 3D points relative to robot base.

        Args:
            coords: List of (x, y) 2D points.

        Returns:
            np.ndarray: 3D points array.
        """
        points = self._env.cams[0].get_obs()['points']
        coords_3d = [points[int(y), int(x)] for x, y in coords]
        coords_3d = np.stack(coords_3d, axis=0)
        return coords_3d - self._env.robot_initial_position
    
    def get_3d_state(self, anchor_coords):
        """
        Get 3D state from 2D anchor coordinates.

        Args:
            anchor_coords: 2D anchor points.

        Returns:
            np.ndarray: 3D points after bilinear interpolation.
        """
        points = self._env.cams[0].get_obs()['points']
        anchor_state = self._get_3d_point_bilinear(points, anchor_coords, self._env.robot_initial_position)
        return anchor_state
    
    def get_anchor_to_ee(self, anchor_pose):
        """
        Calculate anchor position relative to end-effector (EE) frame.

        Args:
            anchor_pose: Anchor pose (at least 3D position).

        Returns:
            tuple: Anchor position in EE coordinate frame.
        """
        anchor_pos = np.array(anchor_pose[:3])
        ee_pos = self._env.get_ee_pos()
        ee_quat = self._env.get_ee_quat()
        
        relative_pos = anchor_pos - ee_pos
        ee_rot = R.from_quat(ee_quat[[1, 2, 3, 0]])
        anchor_in_ee = ee_rot.inv().apply(relative_pos)
        
        return tuple(anchor_in_ee)
    
    def mppi_exec(self, pre_conditions=None, costs=None, post_conditions=None, gripper_command=None):
        """Communicate with Isaac Gym MPPI server to obtain control commands based on conditions and cost, then execute on the robot."

        Args:
            pre_conditions (callable): Check before action.
            costs (str): Target and cost function.
            post_conditions (callable): Check after action.
            gripper_command (optional): Direct gripper command.

        Returns:
            bool: Success status.
        """
    
        if gripper_command is not None:
            self._env.gripper_execute(gripper_command)
            return True
            
        gripper_command = self._env.last_gripper_command
        
        # Extract and clean target code from costs
        target_code = re.search(r"#\s*Target\s*(.*?)#\s*Cost", costs, re.DOTALL).group(1)
        target_code = "\n".join([line.lstrip() for line in target_code.splitlines()])
        
        # Preprocess cost string (remove target section)
        compute_costs_str = re.sub(r'#\s*Target.*?#\s*Cost', '# Cost', costs, flags=re.DOTALL)
        
        while True:
            if not pre_conditions():
                print("The precondition has been violated.")
                self._env.gripper_execute("open")
                return False
            
            # Update perception-dependent target dict
            target_dict = self._update_target_dict(target_code)
            
            robot_state = self._env.get_robot_state()
            
            action = bytes_to_torch(
                    self._env.planner.run_tamp(
                        torch_to_bytes(robot_state), torch_to_bytes(json.dumps(target_dict)), torch_to_bytes(compute_costs_str)))

            self._env.execute_action(action, gripper_command, self._env.vel_scale)
            
            if post_conditions():
                print("The postcondition has been satisfied.")
                gripper_command = np.float32(0)  
                self._env.last_gripper_command = gripper_command
                action = torch.zeros(9, dtype=torch.float32)
                for _ in range(5): 
                    print(_)
                    self._env.execute_action(action, gripper_command)
                break
            
        return True

    def get_ee_pose(self):
        """Get end-effector pose."""
        return self._env.get_ee_pose()
    
    def get_ee_pos(self):
        """Get end-effector position."""
        return self._env.get_ee_pos()
    
    def get_ee_quat(self):
        """Get end-effector quaternion."""
        return self._env.get_ee_quat()
    
    def compute_pos_diff(self, pos1, pos2, threshold):
        """Check if distance between pos1 and pos2 is within threshold."""
        if isinstance(pos1, (int, float)) and isinstance(pos2, (int, float)):
            result =  abs(pos1 - pos2) < threshold
        else:
            result =  np.linalg.norm(np.subtract(pos1, pos2)) < threshold
        return result
        
    def compute_orientation_diff(self, wxyz1, wxyz2, threshold):
        """Check if orientation difference between two quaternions is within threshold."""
        quat_dot = np.clip(abs(np.dot(wxyz1, wxyz2)), -1.0, 1.0)
        return 2 * np.arccos(quat_dot) < threshold
    
    def get_ee_axis_vector(self, ee_wxyz, axis="z"):
        """
        Calculate end-effector axis direction in world frame.

        Args:
            ee_wxyz (np.array): End-effector quaternion [w, x, y, z].
            axis (str): Axis to compute ('x', 'y', or 'z').

        Returns:
            np.array: Direction vector of the specified axis in world coordinates.
        """
        axis_dict = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}
        if axis not in axis_dict:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

        local_axis = axis_dict[axis]
        quat = R.from_quat([ee_wxyz[1], ee_wxyz[2], ee_wxyz[3], ee_wxyz[0]])  # convert to [x, y, z, w]
        world_axis = quat.apply(local_axis)  # rotate to world frame

        return world_axis

    def compute_vector_diff(self, vector, axis, threshold, degree=0, direction=None):
        """
        Check if vector angle to axis is within threshold.

        Args:
            vector (np.array): Input vector.
            axis (str): 'x', 'y', or 'z'.
            threshold (float): Angle threshold (deg).
            degree (float): Target angle (deg).
            direction (str): 'positive' or 'negative'.

        Returns:
            bool: True if within threshold.
        """
        unit_vec = vector / np.linalg.norm(vector)
        axes = {"x": [1,0,0], "y": [0,1,0], "z": [0,0,1]}
        if axis not in axes:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")
        
        angle_pos = np.degrees(np.arccos(np.clip(np.dot(unit_vec, axes[axis]), -1, 1)))
        angle_neg = np.degrees(np.arccos(np.clip(np.dot(unit_vec, -np.array(axes[axis])), -1, 1)))

        if direction == 'positive':
            return abs(angle_pos - degree) < threshold
        elif direction == 'negative':
            return abs(angle_neg - degree) < threshold

    def is_object_grasped(self, pos):
        """Check if the object is grasped."""
        return self._env.is_object_grasped(pos)

    def back_to_inital_pose(self):
        """Move the robot back to its initial pose."""
        return self._env.back_to_inital_pose()  
    
    def transform_orientation(self, name, xyzw, convention, angle, type='world'):
        """
        Rotate object orientation in either object or world frame.

        Args:
            name (str): Object name for angle lookup.
            xyzw (list): Current quaternion [x, y, z, w].
            convention (str): Euler convention, e.g., 'xyz'.
            angle (list or None): Rotation angles in degrees.
            type (str): 'object' for relative, 'world' for absolute rotation.

        Returns:
            list: Rotated quaternion [w, x, y, z].
        """
        if type == 'object':
            if angle is None:
                for key, value in ANGLE_OBJECT_INFO.items():
                    if name in key:
                        angle = value
                        break
            # Relative rotation in object frame
            rotation = R.from_euler(convention, angle, degrees=True)
            current = R.from_quat(xyzw)
            new_rotation = rotation * current

        elif type == 'world':
            if angle is None:
                for key, value in ANGLE_WORLD_INFO.items():
                    if name in key:
                        angle = value
                        break
            # Absolute rotation in world frame
            new_rotation = R.from_euler('xyz', angle, degrees=True)

        return new_rotation.as_quat()[[3, 0, 1, 2]].tolist()  # Return [w, x, y, z]
    
    # ======================================
    # = internal functions
    # ======================================
    def _update_target_dict(self, target_code):
        """Execute target code to get dynamic goal variables, excluding internal states."""
        local_dict = {}
        exec(target_code, self.variable_vars, local_dict)
        return {k: v for k, v in local_dict.items() if "state" not in k}

    def _extract_coordinates(self, coords, points):
        """
        Extract 3D points for given 2D coordinates (excluding the first).

        Args:
            coords: List of 2D points.
            points: 2D array of 3D points.

        Returns:
            Tuple of 3D points and corresponding 2D points.
        """
        coords_2d = coords[1:]
        coords_3d = [points[int(y), int(x)] for x, y in coords_2d]
        return np.array(coords_3d), np.array(coords_2d)

    def _normalize_coordinates(self, coords):
        """Convert point cloud coordinates from world frame to robot base_link frame."""
        return coords - self._env.robot_initial_position

    def _filter_points_by_density(self, coords_3d, coords_2d, mode=1):
        """
        Filter points by height based on mode.

        Args:
            coords_3d (np.ndarray): Array of 3D points.
            coords_2d (np.ndarray): Corresponding 2D points.
            mode (int): Filtering mode (0: max Z value, 1: KDE density peak).

        Returns:
            target_z (float): Selected Z height value.
            filtered_3d (np.ndarray): 3D points near the target Z.
            filtered_2d (np.ndarray): Corresponding 2D points.
        """
        z_vals = coords_3d[:, 2]

        if mode == 0:
            threshold = 0
            target_z = np.max(z_vals)
        else:
            threshold = 0.01
            kde = gaussian_kde(z_vals, bw_method=0.02)
            z_linspace = np.linspace(np.min(z_vals), np.max(z_vals), 300)
            density = kde(z_linspace)
            peaks, _ = find_peaks(density, height=0.1)
            target_z = z_linspace[peaks[np.argmax(density[peaks])]]

        close_mask = np.abs(z_vals - target_z) < threshold

        if not np.any(close_mask):
            nearest_idx = np.argsort(np.abs(z_vals - target_z))[:2]
            filtered_3d = coords_3d[nearest_idx]
            filtered_2d = coords_2d[nearest_idx]
        else:
            filtered_3d = coords_3d[close_mask]
            filtered_2d = coords_2d[close_mask]

        return target_z, filtered_3d, filtered_2d

    def _filter_by_max_z(self, points_3d, points_2d, threshold=0.02):
        """
        Filter points within a threshold of the maximum Z value.

        Args:
            points_3d (np.ndarray): 3D points array.
            points_2d (np.ndarray): Corresponding 2D points array.
            threshold (float): Allowed Z-value range below the max Z.

        Returns:
            np.ndarray: Filtered 3D points near max Z.
            np.ndarray: Corresponding filtered 2D points.
        """
        max_z = np.max(points_3d[:, 2])

        mask = points_3d[:, 2] >= (max_z - threshold)

        filtered_3d = points_3d[mask]
        filtered_2d = np.array(points_2d)[mask]

        return filtered_3d, filtered_2d

    def _find_most_symmetric_pair(self, filtered_2d_coords, filtered_3d_points, center):
        """
        Find the pair of points whose midpoint is closest to the given center (2D).

        Args:
            filtered_2d_coords (np.ndarray): 2D coordinates of filtered points.
            filtered_3d_points (np.ndarray): Corresponding 3D points.
            center (tuple): Reference center coordinate (x, y).

        Returns:
            np.ndarray: Average 3D coordinates of the most symmetric pair.
            np.ndarray: The 3D coordinates of the pair points.
        """
        min_error = float('inf')
        best_pair = None
        cx, cy = center

        for i in range(len(filtered_2d_coords)):
            for j in range(i + 1, len(filtered_2d_coords)):
                x1, y1 = filtered_2d_coords[i]
                x2, y2 = filtered_2d_coords[j]
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                error = np.hypot(mid_x - cx, mid_y - cy)

                if error < min_error:
                    min_error = error
                    best_pair = (i, j)

        if best_pair is None:
            return None

        i, j = best_pair
        avg_3d = (filtered_3d_points[i] + filtered_3d_points[j]) / 2
        pair_3d = np.vstack([filtered_3d_points[i], filtered_3d_points[j]])

        return avg_3d, pair_3d
    
    def _get_3d_point_bilinear(self, point_cloud, anchor_coords, init_position):
        """
        Args:
            point_cloud (np.ndarray): [H, W, 3] array of 3D points
            anchor_coords (dict): 2D keypoints {obj: {key: [[x, y], ...]}}
            init_position (np.ndarray): reference 3D position to subtract

        Returns:
            dict: {obj: {key: {"init": [3D point list]}}}
        """
        results = {}
        H, W, _ = point_cloud.shape

        for obj, data in anchor_coords.items():
            results[obj] = {}

            for key, coords_list in data.items():
                points = []

                for x, y in [coord[:2] for coord in coords_list]:
                    # Bilinear interpolation
                    x0, y0 = int(np.floor(x)), int(np.floor(y))
                    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)

                    Q11 = point_cloud[y0, x0]
                    Q21 = point_cloud[y0, x1]
                    Q12 = point_cloud[y1, x0]
                    Q22 = point_cloud[y1, x1]

                    dx, dy = x - x0, y - y0
                    R1 = Q11 * (1 - dx) + Q21 * dx
                    R2 = Q12 * (1 - dx) + Q22 * dx
                    P = R1 * (1 - dy) + R2 * dy

                    points.append((P - init_position).tolist())

                results[obj][key] = {"init": points}

        return results

    def _get_rgb_frame(self):
        """
        Fetch and cache the current RGB frame if video saving is enabled.
        """ 
        if self._env.save_video_flag:
            rgb = self._env.cams[1].get_obs()['rgb']
            if len(self._env.video_cache) < self._env.config['video_cache_size']:
                self._env.video_cache.append(rgb)
            else:
                self._env.video_cache.pop(0)
                self._env.video_cache.append(rgb)


def parse_dependencies_by_layer(lmps_config):
    """
    Parses module dependencies based on their 'layer' field ("level-branch").

    Rules:
    - Top-level modules depend on all modules in the next level (all branches).
    - Other modules depend on same-branch modules in the next level.
    - Last-level modules have no dependencies.

    Args:
        lmps_config (dict): {module_name: config_dict}, where config_dict includes a 'layer' key.

    Returns:
        dependencies (dict): {module_name: [dependency_names]}.
        layers (dict): {level: {branch: [module_names]}}.
    """

    layers = {}        # {level: {branch: [module_names]}}
    layer_info = {}    # {module_name: (level, branch)}

    # Parse layer info from configuration
    for name, config in lmps_config.items():
        if 'layer' not in config:
            raise ValueError(f"Module '{name}' missing 'layer' field")

        level, branch = map(int, config['layer'].split('-'))
        layers.setdefault(level, {}).setdefault(branch, []).append(name)
        layer_info[name] = (level, branch)

    dependencies = {}
    all_levels = sorted(layers)
    min_level = all_levels[0]
    max_level = all_levels[-1]

    for name, (level, branch) in layer_info.items():
        deps = []
        next_level = level + 1

        if level < max_level:
            if next_level not in layers:
                raise ValueError(f"Module '{name}' expects next level {next_level}, but it is missing")

            if level == min_level:
                # Top-level modules depend on all modules in the next level (all branches)
                deps = sum(layers[next_level].values(), [])
            else:
                # Other modules depend on modules in the next level with the same branch
                if branch not in layers[next_level]:
                    raise ValueError(f"Module '{name}' expects branch {branch} in level {next_level}, but it is missing")
                deps = layers[next_level][branch]

        # Modules in the last level have no dependencies
        dependencies[name] = deps

    return dependencies, layers

def setup_LMP(env, env_config, visualizer, debug=False):
    """
    Setup and construct LMP modules from configuration.
    
    Args:
        env: environment object
        env_config (dict): environment configuration
        debug (bool): enable debug mode

    Returns:
        lmps (dict): top-level LMP modules
    """
    lmp_config = env_config['lmp_config']
    env_name = env_config['env_name']
    lmps_config = lmp_config['lmps']
    lmp_env = LMP_interface(env, visualizer)

    fixed_vars = {'np': np}
    variable_vars = {
        k: getattr(lmp_env, k)
        for k in dir(lmp_env)
        if callable(getattr(lmp_env, k)) and not k.startswith('_')
    }

    dependencies, layers = parse_dependencies_by_layer(lmps_config)
    constructed_lmps = {}

    # Build modules from bottom layer up
    for level in sorted(layers, reverse=True):
        for branch in sorted(layers[level]):
            for name in layers[level][branch]:
                dep_vars = {dep: constructed_lmps[dep] for dep in dependencies[name]}
                vars_for_lmp = {**variable_vars, **dep_vars}
                constructed_lmps[name] = LMP(
                    name, lmps_config[name], fixed_vars, vars_for_lmp, env.client, debug, env_name
                )

    # Retrieve top-level module (assert exactly one module in the top layer)
    top_level = min(layers)
    top_modules = sum(layers[top_level].values(), [])  # flatten all branches
    assert len(top_modules) == 1, f"Expected a single top-level module, got: {top_modules}"

    task_planner = constructed_lmps[top_modules[0]]
    lmps = {"task_planner_ui": task_planner}

    return lmps
