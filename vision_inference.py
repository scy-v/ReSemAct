import re
import os
import sys
import json
import torch
import base64
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from visualizer import img_show, img_save
from PIL import Image, ImageDraw, ImageFont
base_dir = os.path.dirname(os.path.abspath(__file__))
fastsam_path = os.path.join(base_dir, 'FastSAM')

if fastsam_path not in sys.path:
    sys.path.insert(0, fastsam_path)
from FastSAM.fastsam import FastSAM, FastSAMPrompt

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf;-8')
        
def get_abs_path(rel_path):
    return os.path.join(base_dir, rel_path)

class VisionInference():
    def __init__(self, env):
        # Environment and config
        self.env_config = env.config
        self.vfm_config = self.env_config['vfm']
        self.client = env.client
        self.device = self.vfm_config["device"]

        # Path setup
        self.image = get_abs_path(self.vfm_config['image'])
        self.clustered_anchor = get_abs_path(self.vfm_config['clustered_anchor'])
        self.extracted_mask = get_abs_path(self.vfm_config['extracted_mask'])
        self.refined_label_grid = get_abs_path(self.vfm_config['refined_label_grid'])
        self.mllm_inference_dir = get_abs_path(self.vfm_config["mllm_inference"])
        
        # Initialize visual feature module
        self.init_vfm()
        
    def init_vfm(self):
        """Init FastSam Model."""
        self.fastsam_model = FastSAM(self.vfm_config["weight"])
        self.prompt_process = FastSAMPrompt(self.image, "", device=self.device)
        
    def vfm_inference(self):
        """Run FastSAM visual inference"""
        everything_results = self.fastsam_model(self.image, device=self.device, retina_masks=True, imgsz=1024, conf=self.vfm_config["conf"], iou=self.vfm_config["iou"])
        self.prompt_process.results = everything_results
        ann = self.prompt_process.everything_prompt()
        
        if self.env_config["visualize"]:
            self.fastsam_plot(ann, self.vfm_config["mask"])
        return ann
    
    def delete_model(self):
        del self.fastsam_model  # Delete the model instance to free up memory
        torch.cuda.empty_cache()  # Empty the cache to release unused memory
        
    def affordance_process(self, ann):
        """
            Step 1: Mask Filtering.
            Step 2: Visual Consistency Clustering.
        """
        #mark
        marked_img, _ = self.mark_affordance_number(ann)
        img_save(marked_img, output=self.vfm_config["anchor"])
        if self.env_config["visualize"]:
            img_show(marked_img, title='marked_ann')

        # filter mask
        filtered_ann = self.filter_masks(ann)
        self.fastsam_plot(filtered_ann, self.vfm_config["filtered_mask"])
        
        #mark
        marked_img, _ = self.mark_affordance_number(filtered_ann)
        img_save(marked_img, output=self.vfm_config["filtered_anchor"])
        if self.env_config["visualize"]:
            img_show(marked_img, title='marked_filtered_ann')

        #cluster mask
        clustered_ann = self.cluster_masks(filtered_ann)
        self.fastsam_plot(clustered_ann, self.vfm_config["clustered_mask"])
        
        #mark
        marked_img, affordance_coords = self.mark_affordance_number(clustered_ann)
        img_save(marked_img, output=self.vfm_config["clustered_anchor"])
        if self.env_config["visualize"]:
            img_show(marked_img, title='marked_cluster_ann')
        
        return clustered_ann, affordance_coords
    
    def fastsam_plot(self, ann, output):
        """Plot and save annotations"""
        self.prompt_process.plot(annotations=ann,output_path=output,)
        
    def _build_extraction_prompt(self, query, objects):
        """Build a prompt message for the "extract" type."""
        base64_img = encode_image(self.image)
        base64_anchor = encode_image(self.clustered_anchor)
        with open(os.path.join(self.mllm_inference_dir, self.vfm_config["extraction_prompt"]), 'r') as f:
            self.prompt_extract = f.read()
            
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_extract.format(query=query, objects=objects)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}"
                        }
                    }, 
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_anchor}"
                        }
                    },
                ]
            }
        ]
        return messages
    
    def _build_refinement_prompt(self, query, params):
        """Build a prompt message for the "geometry_refine" type."""
        base64_img = encode_image(self.image), 
        base64_anchor = encode_image(self.clustered_anchor)
        base64_extrac_mask = encode_image(self.extracted_mask)
        base64_refined_label_grid = encode_image(self.refined_label_grid)
        extracted_anchor, label_at_scaled = params
        with open(os.path.join(self.mllm_inference_dir, self.vfm_config["refinement_prompt"]), 'r') as f:
            self.prompt_refine = f.read()
            
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_refine.format(extracted_anchor=extracted_anchor, mark_label=label_at_scaled, query=query)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}"
                        }
                    }, 
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_anchor}"
                        }
                    },                    
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_extrac_mask}"
                        }
                    }, 
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_refined_label_grid}"
                        }
                    },
                ]
            }
        ]
        return messages
    
    def mllm_inference(self, query="", type="extract", objects=[None], params=[None]):
        """
        Perform multi-modal large language model (MLLM) inference using different prompt types.

        Args:
            query (str): The input query string for the prompt.
            type (str): The type of prompt to build and inference to run. Supported: "extract", "geometry_refine".
            objects (list): List of objects for the "extract" prompt type.
            params (list): List of parameters for the "geometry_refine" prompt type.

        Returns:
            dict or str: Parsed inference result as a dictionary for "extract" type, 
                        or a specific label string for "geometry_refine" type.
        """
        # Select the correct prompt builder based on the query type
        prompt_builders = {
            "extract": self._build_extraction_prompt,
            "geometry_refine": self._build_refinement_prompt
        }
        
        # Validate prompt type
        if type not in prompt_builders:
            raise ValueError(f"Unsupported query type: {type}")
        
        # Build the messages for the chosen prompt type
        messages = prompt_builders[type](query, params) if type == "geometry_refine" else prompt_builders[type](query, objects)
        
        # Call the MLLM model to get inference results
        result_info = self.client.chat.completions.create(
            model=self.vfm_config["model"],
            messages=messages,
            temperature=self.vfm_config["temperature"],
            stream=False
        ).choices[0].message.content
        
        print(result_info)
        
        # Clean and parse the returned JSON content
        result_info_dict = self._clean_and_parse_json(result_info)
        
        # Return result based on the prompt type
        if type == "geometry_refine":
            return result_info_dict.get("label")
        
        return result_info_dict


    def _clean_and_parse_json(self, content: str) -> dict:
        """Clean and parse JSON content from a string."""
        
        cleaned_content = re.sub(r'```json|```', '', content).strip()
        
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    def filter_masks(self, annotation):
        """Step 1: Mask Filtering."""
        annotation = annotation.cpu().numpy()
        filtered_by_area = []
        filtered_final = []

        img_area = annotation[0].shape[0] * annotation[0].shape[1]
        min_area_ratio, max_area_ratio = self.vfm_config["filter_scale"][1], self.vfm_config["filter_scale"][0]
        overlap_threshold_other = self.vfm_config["overlap_other"]
        overlap_threshold_self = self.vfm_config["overlap_self"]
        contained_num_threshold = self.vfm_config["contained_num"]

        # Area-based Filtering
        for mask in annotation:
            mask_bool = mask.astype(bool)
            area = np.sum(mask_bool)
            if area < img_area * min_area_ratio or area > img_area * max_area_ratio:
                continue
            filtered_by_area.append(mask_bool)

        # Structural Independence Filtering
        for i, mask_i in enumerate(filtered_by_area):
            area_i = np.sum(mask_i)
            contained_count = 0

            for j, mask_j in enumerate(filtered_by_area):
                if i == j:
                    continue
                area_j = np.sum(mask_j)
                overlap_area = np.sum(mask_i & mask_j)

                if area_j > 0 and area_i > 0:
                    overlap_ratio_j = overlap_area / area_j
                    overlap_ratio_i = overlap_area / area_i

                    if overlap_ratio_j >= overlap_threshold_other and overlap_ratio_i >= overlap_threshold_self:
                        contained_count += 1

            if contained_count < contained_num_threshold:
                filtered_final.append(mask_i)

        return torch.from_numpy(np.array(filtered_final)).to(self.device)

    def cluster_masks(self, annotation):
        """Step 2: Visual Consistency Clustering."""
        annotation = annotation.cpu().numpy()
        centers = []

        # Calculate center points of each mask
        for mask in annotation:
            y_coords, x_coords = np.where(mask)
            if x_coords.size > 0 and y_coords.size > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                centers.append((center_x, center_y))

        centers_np = np.array(centers)
        # Perform DBSCAN clustering on centers
        clustering = DBSCAN(eps=self.vfm_config["cluster"], min_samples=1).fit(centers_np)
        labels = clustering.labels_

        clustered_masks = []
        # Merge masks within the same cluster
        for label in set(labels):
            indices = np.where(labels == label)[0]
            combined_mask = np.zeros_like(annotation[0], dtype=bool)
            for idx in indices:
                combined_mask |= annotation[idx]
            clustered_masks.append(combined_mask)

        return torch.from_numpy(np.array(clustered_masks)).to(self.device)

    def mark_affordance_number(self, annotation):
        """
        Draws a rectangular marker with an index number at the center of each annotation mask.

        Args:
            annotation (Tensor): A tensor of boolean masks indicating regions.

        Returns:
            marked_img (PIL.Image): Image with drawn anchor rectangles and indices.
            affordance_coords (dict): Mapping from index (str) to (x, y) center coordinates.
        """
        # Load and convert the image to RGB
        image = Image.open(self.image).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Load the font (fallback to default if custom font not found)
        try:
            font_path = get_abs_path(self.env_config["ttf"])
            font = ImageFont.truetype(font_path, self.vfm_config["font_size"])
        except IOError:
            font = ImageFont.load_default()

        # Convert annotation tensor to NumPy array
        annotation = annotation.cpu().numpy()

        affordance_coords = {}

        outer_offset = self.vfm_config["outer_offset"]
        inner_offset = self.vfm_config["inner_offset"]

        for idx, mask in enumerate(annotation):
            # Find pixel coordinates where mask is True
            y_indices, x_indices = np.where(mask)

            if len(x_indices) == 0 or len(y_indices) == 0:
                continue  # Skip empty masks

            # Calculate the center coordinate of the mask
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))
            affordance_coords[str(idx)] = (center_x, center_y)
            
            # Draw inner rectangle (white background)
            draw.rectangle(
                [
                    (center_x - inner_offset, center_y - inner_offset),
                    (center_x + inner_offset, center_y + inner_offset)
                ],
                fill='white'
            )
            
            # Draw outer rectangle (black border)
            draw.rectangle(
                [
                    (center_x - outer_offset, center_y - outer_offset),
                    (center_x + outer_offset, center_y + outer_offset)
                ],
                outline='black',
                width=self.vfm_config["outer_offset"] - self.vfm_config["inner_offset"]
            )

            # Prepare text and compute its position (centered)
            text = str(idx)
            text_width, text_height = draw.textsize(text, font=font)
            text_x = center_x - text_width // 2
            text_y = center_y - text_height // 2 - 2

            # Draw the text in a soft red color (Crimson)
            draw.text(
                (text_x, text_y),
                text,
                fill=(255, 20, 20),  # Crimson red
                font=font
            )

        marked_img = image
        return marked_img, affordance_coords