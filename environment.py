import os
import yaml
import copy
import torch
import imageio
import zerorpc
import datetime
import numpy as np
import omnigibson as og
from openai import OpenAI
from utils import save_image, pose2mat, pose_inv
from functools import partial
from og_utils import OGCamera
import matplotlib.pyplot as plt
from omnigibson.macros import gm
from PIL import Image, ImageDraw, ImageFont
from vision_inference import VisionInference
from omnigibson.robots.franka import FrankaPanda
from scipy.spatial.transform import Rotation as R
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.utils.usd_utils import PoseAPI, mesh_prim_mesh_to_trimesh_mesh, mesh_prim_shape_to_trimesh_mesh
base_dir = os.path.dirname(os.path.abspath(__file__))
# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False
gm.ENABLE_OBJECT_STATES = True
# some customization to the OG functions

FrankaPanda._initialize = ManipulationRobot._initialize

def get_abs_path(rel_path):
    return os.path.join(base_dir, rel_path)

class OGEnv:
    def __init__(self, config, scene_file, objects_file, verbose=False):
        # Basic setup
        self.video_cache = []
        self.config = config

        # Robot start position
        self.robot_initial_position = self.config['robot']['robot_config']['position']

        # Load scene file
        self.config['scene']['scene_file'] = scene_file

        # Load VFM config
        self.vfm_config = self.config['vfm']
        
        # Init OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY", self.config["api_key"])
        self.client = OpenAI(base_url=self.config["base_url"], api_key=api_key)

        # Load objects from file
        with open(objects_file, 'r') as f:
            data = yaml.safe_load(f)
        self.config['objects'] = [
            {key: obj[key] for key in obj} for obj in data["objects"]
        ]

        # Create environment
        og_env_args = dict(
            objects=self.config['objects'],
            scene=self.config['scene'],
            robots=[self.config['robot']['robot_config']],
            env=self.config['og_sim']
        )
        self.og_env = og.Environment(og_env_args)
        self.og_env.scene.update_initial_state()
        
        # robot vars
        self.robot = self.og_env.robots[0]
        
        # Warm up simulator
        for _ in range(10):
            og.sim.step()

        # Init cameras
        self._initialize_cameras(self.config['camera'])
        self.resolution = self.cams[0].resolution
        self.target_size = (self.resolution, self.resolution)
        
        # Vision and visualization flags
        self.debug = self.config['debug']
        self._back_to_initial_done = False
        self.grid_size = self.config['grid_size']
        self.visualize = self.config["visualize"]
        self.save_video_flag = self.config['save_video_flag']

        # Setup MPPI planner
        self.planner = zerorpc.Client()
        self.last_gripper_command = np.float32(1)
        self.vel_scale = self.config["vel_scale"]
        self.planner.connect(self.config["zerorpc"])
        
        # Init VFM module
        self.vision_infer = VisionInference(self)
        
    # ======================================
    # = internal functions
    # ======================================
    
    def _normalize_mask(self, coords):
        """
        Step 1: Normalize mask and get transform params.

        Args:
            coords (np.ndarray): coords + mask idx.

        Returns:
            tuple or None: (bin_mask, params) or None if empty.
        """

        # get mask idx and mask array
        mask_idx = int(coords[0][2])
        raw_mask = self.clustered_ann[mask_idx].cpu().numpy()

        # binarize mask (1 or 0)
        binary_mask = (raw_mask > 0).astype(np.uint8)

        # output size and padding
        target_w, target_h = self.target_size
        max_h, max_w = target_h, target_w
        pad_h, pad_w = 30, 40

        # size after padding
        resized_h = max_h - pad_h
        resized_w = max_w - pad_w

        # get mask pixel coords
        ys, xs = np.where(binary_mask == 1)
        if ys.size == 0 or xs.size == 0:
            return None

        # bounding box coords
        top, bottom = ys.min(), ys.max()
        left, right = xs.min(), xs.max()

        width = right - left + 1
        height = bottom - top + 1

        # width/height ratio
        aspect = width / height

        # new size with aspect
        new_h = resized_h
        new_w = int(new_h * aspect)
        if new_w > resized_w:
            new_w = resized_w
            new_h = int(new_w / aspect)

        # scale factors for x and y
        scale_x = width / new_w
        scale_y = height / new_h

        # offset to center mask
        offset_x = (max_w - new_w) // 2
        offset_y = (max_h - new_h) // 2

        # collect transform params
        params = {
            'scale_x': scale_x,
            'scale_y': scale_y,
            'top_left_x': offset_x,
            'top_left_y': offset_y,
            'left': left,
            'top': top,
            'new_width': new_w,
            'new_height': new_h,
            'right': right,
            'bottom': bottom,
        }

        return binary_mask, params

    def _construct_grid_and_labels(self, mask, params):
        """
        Step 2: Draw grid on resized mask and embed labels.

        Args:
            mask (np.ndarray): Binary mask.
            params (dict): Transform parameters from normalization step.

        Returns:
            Tuple[Image.Image, Dict[int, Tuple[float, float]]]: 
                - Grid image with labels,
                - Mapping of label index to center coords.
        """
        # Grid cell size
        grid_cols, grid_rows = self.grid_size
        canvas_w, canvas_h = self.target_size
        cell_w = canvas_w // grid_cols
        cell_h = canvas_h // grid_rows

        # Resize mask to fit canvas
        crop = mask[params['top']:params['bottom'] + 1, params['left']:params['right'] + 1]
        resized_mask = np.array(
            Image.fromarray(crop).resize((params['new_width'], params['new_height']), Image.NEAREST)
        )

        # Build background image and paste resized mask
        bg_color = (68, 1, 84, 255)
        fg_color = (253, 231, 36, 255)
        canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

        rgba = np.zeros((resized_mask.shape[0], resized_mask.shape[1], 4), dtype=np.uint8)
        rgba[resized_mask == 0] = bg_color
        rgba[resized_mask == 1] = fg_color
        mask_rgba = Image.fromarray(rgba, mode="RGBA")
        canvas.paste(mask_rgba, (params['top_left_x'], params['top_left_y']))

        # Draw grid
        draw = ImageDraw.Draw(canvas)
        for col in range(1, grid_cols):
            x = col * cell_w
            draw.line([(x, 0), (x, canvas_h)], fill=(160, 160, 160), width=1)
        for row in range(1, grid_rows):
            y = row * cell_h
            draw.line([(0, y), (canvas_w, y)], fill=(160, 160, 160), width=1)

        # Add labels and compute centers
        label_coords = {}
        font = ImageFont.truetype(self.config['ttf'], size=35)
        canvas_gray = np.array(canvas.convert("L"))
        label_id = 1

        for row in range(grid_rows):
            for col in range(grid_cols):
                x0 = col * cell_w
                y0 = row * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                cell = canvas_gray[y0:y1, x0:x1]

                if np.sum(cell == 215) > 0.1 * cell_w * cell_h:
                    ys, xs = np.where(cell == 215)
                    center_x = np.mean(xs) + x0
                    center_y = np.mean(ys) + y0
                    label_coords[label_id] = (center_x, center_y)

                    text_x = x0 + (cell_w - font.getsize(str(label_id))[0]) / 2
                    text_y = y0 + (cell_h - font.getsize(str(label_id))[1]) / 2
                    draw.text((text_x, text_y), str(label_id), fill="red", font=font)

                    label_id += 1

        return canvas, label_coords


    def _associate_centroid_with_label(self, coords, params, label_coords):
        """
        Step 3: Map original point to scaled space and extract label.

        Args:
            coords (np.ndarray): Original input coordinates.
            params (dict): Transform parameters from step 1.
            label_coords (dict): Mapping from label ID to cell center (x, y).

        Returns:
            int or None: Associated label ID or None if not found.
        """
        # Extract original point
        orig_x, orig_y = coords[0][:2]

        # Map to scaled space
        x_scaled = (orig_x - params['left']) / params['scale_x'] + params['top_left_x']
        y_scaled = (orig_y - params['top']) / params['scale_y'] + params['top_left_y']

        # Determine grid cell
        cell_w = self.target_size[0] // self.grid_size[0]
        cell_h = self.target_size[1] // self.grid_size[1]
        col_idx = int(x_scaled // cell_w)
        row_idx = int(y_scaled // cell_h)

        # Find label in corresponding cell
        for label_id, (center_x, center_y) in label_coords.items():
            if row_idx * cell_h <= center_y < (row_idx + 1) * cell_h and \
            col_idx * cell_w <= center_x < (col_idx + 1) * cell_w:
                return label_id

        return None

    def _visualize_mapping(self, img1, img2, points1, points2, titles, labels1, labels2, save_name):
        """
        Visualization for coordinate mapping.

        Args:
            img1 (np.ndarray or PIL.Image): Left-side image (original or scaled).
            img2 (np.ndarray or PIL.Image): Right-side image (scaled or original).
            points1 (List[Tuple[float, float]]): Points on img1.
            points2 (List[Tuple[float, float]]): Corresponding points on img2.
            titles (Tuple[str, str]): Titles for the two subplots.
            labels1 (List[str]): Labels for points on img1.
            labels2 (List[str]): Labels for points on img2.
            save_name (str): Key for saving path in vfm_config.
        """

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img1)
        axes[0].set_title(titles[0])
        axes[1].imshow(img2)
        axes[1].set_title(titles[1])

        for (x1, y1), (x2, y2), label1, label2 in zip(points1, points2, labels1, labels2):
            axes[0].scatter(x1, y1, color='blue', s=20, label=label1)
            axes[1].scatter(x2, y2, color='red', s=20, label=label2)

        axes[0].legend()
        axes[1].legend()

        plt.savefig(get_abs_path(self.vfm_config[save_name]), bbox_inches='tight', pad_inches=0)
        plt.show(block=False)
        plt.pause(0.3)
    
    # ======================================
    # = public functions
    # ======================================
    
    def vision_inference(self):
        """Run VFM vision inference."""
        img_tensor = self.cams[0].get_obs()['rgb']
        save_image(img_tensor, get_abs_path(self.vfm_config["image"]))
        ann = self.vision_infer.vfm_inference()
        self.clustered_ann, self.affordance_coords = self.vision_infer.affordance_process(ann)
        self.vision_infer.delete_model()
        return ann
    
    def transform_to_label(self, coords):
        """
        Geometric Refinement Flow:
            Step 1: mask normalization
            Step 2: grid construction and label embedding
            Step 3: centroid mapping and label association

        Args:
            coords (np.ndarray): Input point and mask index.

        Returns:
            int or None: Assigned grid label or None if unmatched.
        """
        # Step 1: normalize mask and compute transform params
        mask, transform_params = self._normalize_mask(coords)

        # Step 2: construct grid & label
        canvas, label_coords = self._construct_grid_and_labels(mask, transform_params)

        # Step 3: map point to label
        label = self._associate_centroid_with_label(coords, transform_params, label_coords)

        # Save intermediate images
        Image.fromarray((mask * 255).astype(np.uint8)).save(get_abs_path(self.vfm_config['refined_mask']))
        canvas.save(get_abs_path(self.vfm_config['refined_label_grid']))

        # Visualization
        if self.visualize:
            orig_pt = (coords[0][0], coords[0][1])
            scaled_pt = (
                (orig_pt[0] - transform_params['left']) / transform_params['scale_x'] + transform_params['top_left_x'],
                (orig_pt[1] - transform_params['top']) / transform_params['scale_y'] + transform_params['top_left_y']
            )
            self._visualize_mapping(
                img1=mask,
                img2=canvas,
                points1=[orig_pt],
                points2=[scaled_pt],
                titles=("Original Image", "Scaled Image"),
                labels1=["Original Point"],
                labels2=["Scaled Point"],
                save_name='mapped_forward_mask'
            )

        # Save parameters for reverse transform
        self.params = {
            "mask": mask,
            "canvas": canvas,
            "label_coords": label_coords,
            "transform_params": transform_params,
        }

        return label

    def transform_to_coords(self, scaled_labels):
        """
        Geometric Refinement Flow:
            Step 5: Refined Anchors as Affordance Targets

        Args:
            scaled_labels (List[int]): List of predicted labels on scaled grid.

        Returns:
            List[Tuple[float, float] or None]: List of corresponding coordinates on original image.
        """
        params = self.params
        transform_params = params["transform_params"]
        label_centers = params["label_coords"]

        original_coords = []

        for label in scaled_labels:
            if label not in label_centers:
                original_coords.append(None)
                continue

            x_scaled, y_scaled = label_centers[label]
            x_orig = (x_scaled - transform_params['top_left_x']) * transform_params['scale_x'] + transform_params['left']
            y_orig = (y_scaled - transform_params['top_left_y']) * transform_params['scale_y'] + transform_params['top']
            original_coords.append((x_orig, y_orig))

            print(f"Label {label} → Original Coord: ({x_orig:.1f}, {y_orig:.1f})")

        # Visualization
        if self.visualize:
            points1, points2 = [], []
            labels1, labels2 = [], []

            for idx, label in enumerate(scaled_labels):
                if original_coords[idx] is None:
                    continue
                x_scaled, y_scaled = label_centers[label]
                x_orig, y_orig = original_coords[idx]
                points1.append((x_scaled, y_scaled))
                points2.append((x_orig, y_orig))
                labels1.append(f"Scaled Point {idx}")
                labels2.append(f"Original Point {idx}")

            self._visualize_mapping(
                img1=params["canvas"],
                img2=params["mask"],
                points1=points1,
                points2=points2,
                titles=("Scaled Image", "Original Image"),
                labels1=labels1,
                labels2=labels2,
                save_name='mapped_backward_mask'
            )


        return original_coords


    def get_2d_coords(self, extracted_anchor):
        """
        Get 2D coordinates for extracted anchors.

        Args:
            extracted_anchor (dict): Keys are types, values contain 'extracted' list of IDs.

        Returns:
            dict: Same keys, values have 'extracted' list of (x, y, id) tuples.

        """
        if self.debug:
            total_ids = sum(len(item['extracted']) for item in extracted_anchor.values())
            user_input = input(f"Confirm IDs. Current: {extracted_anchor}. Enter {total_ids} IDs separated by spaces: ").strip()

            if user_input == "":
                print("No input, keep original IDs.")
            else:
                user_input_list = user_input.split()
                if len(user_input_list) != total_ids:
                    raise ValueError(f"Input count {len(user_input_list)} does not match expected {total_ids}.")

                idx = 0
                for key in extracted_anchor:
                    count = len(extracted_anchor[key]['extracted'])
                    extracted_anchor[key]['extracted'] = user_input_list[idx:idx + count]
                    idx += count

            print(f"Final IDs: {extracted_anchor}")

        anchor_coords = {}
        for key, item in extracted_anchor.items():
            coords = []
            for id in item['extracted']:
                if str(id) in self.affordance_coords:
                    coords.append((self.affordance_coords[str(id)][0], self.affordance_coords[str(id)][1], id))
            anchor_coords[key] = {'extracted': coords}
        return anchor_coords
    
    def gripper_execute(self, gripper_command):
        """Execute open/close gripper command."""
        action_pos = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        if gripper_command == "open":
            gripper_command = np.float32(1)
        elif gripper_command == "close":
            gripper_command = np.float32(0)
        self.last_gripper_command = gripper_command
        for _ in range(10): 
            self.execute_action(action_pos, gripper_command)

            
    def get_cam_obs(self):
        """Get observations from all cameras."""
        self.last_cam_obs = dict()
        for cam_id in self.cams:
            self.last_cam_obs[cam_id] = self.cams[cam_id].get_obs()  # each containing rgb, depth, points, seg
        return self.last_cam_obs
    
    def register_anchor(self, anchor_dict):
        """
        Register anchors by finding closest mesh points in the environment and storing related data.

        Args:
            anchor_dict (dict): Data structured by object names and types ('extracted', 'refined').

        Returns:
            dict: Copy of anchor_dict with added 'move' info: tuples of (mesh path, mesh pose, position).
        """
        exclude_names = {'wall', 'floor', 'ceiling', 'table', 'frankapanda', 'robot', 'workbench'}
        results = copy.deepcopy(anchor_dict)

        for obj_name, anchor_types in anchor_dict.items():
            for anchor_type in ['extracted', 'refined']:
                if anchor_type not in anchor_types:
                    continue

                anchor_source = anchor_types[anchor_type]

                move_results = []
                for values in anchor_source.values():
                    for anchor_position in values:
                        anchor_position = np.add(anchor_position, self.robot_initial_position).reshape(1, -1) 
                        closest_distance = float('inf')
                        closest_prim_path, closest_point, closest_obj = None, None, None

                        for obj in self.og_env.scene.objects:
                            if any(name in obj.name.lower() for name in exclude_names):
                                continue
                            for link in obj.links.values():
                                if not hasattr(link, 'visual_meshes'):
                                    continue
                                for mesh in link.visual_meshes.values():
                                    mesh_prim_path = mesh.prim_path
                                    mesh_type = mesh.prim.GetPrimTypeInfo().GetTypeName()
                                    trimesh_object = mesh_prim_mesh_to_trimesh_mesh(mesh.prim) if mesh_type == 'Mesh' else mesh_prim_shape_to_trimesh_mesh(mesh.prim)
                                    trimesh_object.apply_transform(PoseAPI.get_world_pose_with_scale(mesh.prim_path))
                                    points_transformed = trimesh_object.sample(1000)

                                    dists = np.linalg.norm(points_transformed - anchor_position, axis=1)
                                    point = points_transformed[np.argmin(dists)]
                                    distance = np.linalg.norm(point - anchor_position)

                                    if distance < closest_distance:
                                        closest_distance = distance
                                        closest_prim_path, closest_point, closest_obj = mesh_prim_path, point, obj

                        move_results.append((closest_prim_path, PoseAPI.get_world_pose(closest_prim_path), anchor_position))

                results[obj_name][anchor_type]["move"] = move_results

        return results
    
    def execute_action(self, action, gripper_command, vel_scale=0.3):
        """
        Execute robot action by scaling velocities and appending gripper command.

        Args:
            action (Tensor): Robot velocity commands in joint space.
            gripper_command (float): Gripper control input.
            vel_scale (float, optional): Velocity scaling factor. Default is 0.3.

        Returns:
            None
        """
        action_panda = {}
        action = action.cpu().numpy() * vel_scale
        action_panda[self.robot.name] = np.append(action[:-2], gripper_command)
        self.og_env.step(action_panda[self.robot.name])
        
    def get_anchor_position(self, register_anchor):
        """
        Get current 3D position and orientation of a anchor relative to robot start.

        Args:
            register_anchor (tuple): (prim_path, initial_pose, anchor_point)

        Returns:
            list: Position (xyz) relative to robot start + rotation quaternion
        """
        closest_prim_path, init_pose, closest_point = register_anchor
        prim_path_xyzw = np.array(init_pose[1])
        init_pose = pose2mat(init_pose)
        centering_transform = pose_inv(init_pose)
        centered = np.dot(centering_transform, np.append(closest_point, 1))[:3]
        curr_pose = pose2mat(PoseAPI.get_world_pose(closest_prim_path))
        anchor_positions = np.dot(curr_pose, np.append(centered, 1))[:3] - self.robot_initial_position
        return np.hstack((anchor_positions, prim_path_xyzw)).tolist()

    def get_movable_anchor_pos(self, register_anchor_state):
        """
        Wraps static anchor states with position retrieval functions.

        Args:
            register_anchor_state (dict): Static states of anchors.

        Returns:
            dict: Anchor states updated with callable position getters,
                merged with initial orientations.
        """
        results = copy.deepcopy(register_anchor_state)

        for obj_name, anchor_types in register_anchor_state.items():  
            for anchor_type in ['extracted', 'refined']: 
                if anchor_type in anchor_types:
                    results[obj_name][anchor_type]["move"] = [
                        partial(self.get_anchor_position, anchor) for anchor in anchor_types[anchor_type]["move"]
                    ]
        return self.merge_orientation_to_init(results)
    
    def merge_orientation_to_init(self, data):
        """
        Appends orientation (quaternion) from registered anchors to each init position.

        Args:
            data (dict):  Data with 'init' positions and 'move' callbacks.

        Returns:
            dict: Updated anchor data with orientation merged into each init.
        """
        for obj_name, obj_data in data.items():
            if 'extracted' in obj_data:
                extracted_init = obj_data['extracted']['init']
                moves = obj_data['extracted']['move']  
                for i, init_list in enumerate(extracted_init):
                    merge_part = moves[i]()  
                    merge_part = merge_part[3:]  
                    init_list.extend(merge_part)  

            if 'refined' in obj_data:
                refined_init = obj_data['refined']['init']
                moves = obj_data['refined']['move'] 
                for i, init_list in enumerate(refined_init):
                    merge_part = moves[i]() 
                    merge_part = merge_part[3:]  
                    init_list.extend(merge_part)  

        return data
            
    def get_object_pose(self, obj_name):
        """
        Retrieves the current 3D position of the specified object.

        Args:
            obj_name (str): Name of the target object.

        Returns:
            np.ndarray: 3D position of the object in world coordinates.
        """
        target_pose = self.og_env.scene.object_registry("name", obj_name).get_position_orientation()[0]
        return target_pose.numpy()
    
    def is_object_grasped(self, pos):
        """
        Checks if the end-effector is close enough to the target position.

        Args:
            pos (np.ndarray): Target position to check against the end-effector.

        Returns:
            bool: True if the end-effector is within the threshold of the target, else False.
        """
        return np.linalg.norm(self.get_ee_pos() - pos) < 0.02
    
    def back_to_inital_pose(self):
        """
        Moves the robot to a predefined initial joint position once.

        Returns:
            bool: Always returns True after moving or confirming initial pose.
        """
        if not self._back_to_initial_done:
            initial_pos = torch.tensor([0.00, -1.3, 0.00, -2.87, 0.00, 1.57, 0.75, 0.00, 0.00])
            self.robot.set_joint_positions(initial_pos)
            self._back_to_initial_done = True
            
        return True
    
    def get_robot_state(self):
        """
        Returns the current joint positions and velocities of the FrankaPanda robot.

        Returns:
            torch.Tensor: A tensor containing [pos1, vel1, pos2, vel2, ..., pos7, vel7, 0, 0, 0, 0].
                        The last four zeros are placeholders for the gripper fingers.
        """
        assert isinstance(self.robot, FrankaPanda), "The IK solver assumes the robot is a FrankaPanda robot"
        arm_joint_state = []
        arm = self.robot.default_arm
        arm_joint_names = self.robot.arm_joint_names[arm]
        for arm_joint_name in arm_joint_names:
            position = self.robot.joints[arm_joint_name].get_state()[0]
            velocity = self.robot.joints[arm_joint_name].get_state()[1]
            arm_joint_state.append(position)
            arm_joint_state.append(velocity)
        arm_joint_state.extend([0, 0, 0, 0]) #multi-finger in isaacgym
        
        return torch.tensor(arm_joint_state)

    def get_ee_pose(self):
        """
        Calculate the robot's end-effector pose in the world coordinate frame.

        Returns:
            np.ndarray: Array with position (x, y, z), Euler angles (roll, pitch, yaw), 
                        and quaternion orientation (w, x, y, z).
        """
        hand_position, hand_quat_xyzw = self.robot.get_relative_eef_pose(arm='default', mat=False)
        # Combine hand position and quaternion (qx, qy, qz, qw)
        hand_pose = np.concatenate([hand_position, hand_quat_xyzw])

        # Convert hand quaternion (qx, qy, qz, qw) to rotation matrix
        hand_rot_matrix = R.from_quat(hand_pose[3:]).as_matrix()

        # Construct homogeneous transform matrix of hand in world frame
        hand_transform = np.eye(4)
        hand_transform[:3, :3] = hand_rot_matrix
        hand_transform[:3, 3] = hand_pose[:3]

        # Define fixed transform from hand frame to end-effector frame
        hand_to_ee_transform = np.eye(4)
        hand_to_ee_transform[:3, 3] = [0, 0, 0.1034]
        hand_to_ee_rot = R.from_euler('x', 180, degrees=True).as_matrix()
        hand_to_ee_transform[:3, :3] = hand_to_ee_rot

        # Calculate full end-effector transform in world frame
        ee_transform = hand_transform @ hand_to_ee_transform

        # Extract position from transform
        ee_position = ee_transform[:3, 3]
        # Convert rotation matrix to Euler angles (radians)
        ee_euler = R.from_matrix(ee_transform[:3, :3]).as_euler('xyz', degrees=False)
        # Convert rotation matrix to quaternion, reorder to (w, x, y, z)
        ee_quat_wxyz = np.roll(R.from_matrix(ee_transform[:3, :3]).as_quat(), 1)

        # Concatenate position, Euler angles, and quaternion into one array
        ee_pose = np.concatenate([ee_position, ee_euler, ee_quat_wxyz])
        
        return ee_pose

    def get_ee_pos(self):
        """
        Returns the end-effector's position [x, y, z].
        """
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        """
        Returns the end-effector's orientation as a quaternion [w, x, y, z].
        """
        return self.get_ee_pose()[6:]
    
    def get_ee_rpy(self):
        """
        Returns the end-effector's orientation as Euler angles [roll, pitch, yaw].
        """
        return self.get_ee_pose()[3:6]
    
    def save_video(self, save_path=None):
        """
        Saves cached video frames to an MP4 file.

        Args:
            save_path (str, optional): Path to save the video. If None, saves with a timestamped filename.

        Returns:
            str: Full path of the saved video file.
        """
        save_dir = os.path.join(base_dir, 'videos')
        os.makedirs(save_dir, exist_ok=True)
        
        if save_path is None:
            save_path = os.path.join(save_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.mp4')
            
        video_writer = imageio.get_writer(save_path, fps=30)
        
        for rgb in self.video_cache:
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.detach().cpu().numpy()
            if not isinstance(rgb, np.ndarray):
                raise ValueError("Expected rgb to be a numpy array, got: {}".format(type(rgb)))
            video_writer.append_data(rgb)
            
        video_writer.close()
        
        return save_path

    # ======================================
    # = internal functions
    # ======================================
    def _initialize_cameras(self, cam_config):
        """
        Initialize cameras based on the given configuration.
        
        Args:
            cam_config (dict): Mapping from camera IDs to their position and orientation.
        """
        self.cams = dict()
        for cam_id in cam_config:
            cam_id = int(cam_id)
            self.cams[cam_id] = OGCamera(self.og_env, cam_config[cam_id])
        for _ in range(10): og.sim.render()