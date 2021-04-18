# The code in this file performs experiment using different methods of fusion. 
# Fusion in a process to merge RGBD images at various frames to generate a canonical pose & warp field at each timestep 
# The method described by the paper is to use DynamicFusion by Richard Newcombe et al 2015 CVPR 

# Library imports
import os
import sys
import open3d as o3d
import numpy as np

sys.path.append("../") # Making it easier to load modules 
sys.path.append("../csrc") # For loading nnrt module 


# Modules
from utils import image_proc
import utils.viz_utils as viz_utils
import utils.visualization.line_mesh as line_mesh_utils

# Modules for Fusion
from frame_loader import RGBDVideoLoader
from run_model import Deformnet_runner
from tsdf import TSDFVolume,pycuda_ctx


class DynamicFusion():
	def __init__(self, seqpath):
		"""
			Initialize the class which includes, loading Deformnet, loading dataloader for frames
		
			@params:
				seqpath => Location of RGBD sequences (must contain a color dir, depth dir, and interincs.txt)
				source_frame => Source frame over which the canonical pose is generated
		"""
		self.dataloader = RGBDVideoLoader(seqpath)
		self.model = Deformnet_runner() # Class to run deformnet on data

	def plot_frame(self,input_data,model_data):
	

		# Params for visualization correspondence info 
		weight_thr = 0.3
		weight_scale=1

		# Source
		source_pcd = viz_utils.get_pcd(input_data["source"])

		# keep only object using the mask
		valid_source_mask = np.moveaxis(model_data["valid_source_points"], 0, -1).reshape(-1).astype(np.bool)
		source_object_pcd = source_pcd.select_by_index(np.where(valid_source_mask)[0])

		# Source warped
		warped_deform_pred_3d_np = image_proc.warp_deform_3d(
			input_data["source"], input_data["graph"]["pixel_anchors"], input_data["graph"]["pixel_weights"], input_data["graph"]["graph_nodes"], model_data["node_rotations"], model_data["node_translations"]
		)
		source_warped = np.copy(input_data["source"])
		source_warped[3:, :, :] = warped_deform_pred_3d_np
		warped_pcd = viz_utils.get_pcd(source_warped).select_by_index(np.where(valid_source_mask)[0])
		warped_pcd.paint_uniform_color([1, 0.706, 0]) 

		# TARGET
		target_pcd = viz_utils.get_pcd(input_data["target"])
		# o3d.visualization.draw_geometries([source_pcd])
		# o3d.visualization.draw_geometries([source_object_pcd])
		# o3d.visualization.draw_geometries([warped_pcd])
		# o3d.visualization.draw_geometries([target_pcd])

		####################################
		# GRAPH #
		####################################
		rendered_graph = viz_utils.create_open3d_graph(input_data["graph"]["graph_nodes"],input_data["graph"]["graph_edges"])
		rendered_deformed_graph = viz_utils.create_open3d_graph(model_data["deformed_graph_nodes"],input_data["graph"]["graph_edges"])

		# Correspondences
		# Mask
		mask_pred_flat = model_data["mask_pred"].reshape(-1)
		valid_correspondences = model_data["valid_correspondences"].reshape(-1).astype(np.bool)
		# target matches
		target_matches = np.moveaxis(model_data["target_matches"], 0, -1).reshape(-1, 3)
		target_matches = viz_utils.transform_pointcloud_to_opengl_coords(target_matches)


		# "Good" matches
		good_mask = valid_correspondences & (mask_pred_flat >= weight_thr)
		good_matches_set,good_weighted_matches_set = viz_utils.create_matches_lines(good_mask,np.array([0.0, 0.8, 0]),np.array([0.8, 0, 0.0]),source_pcd,target_matches,mask_pred_flat,weight_thr,weight_scale)

		bad_mask = valid_correspondences & (mask_pred_flat < weight_thr)
		bad_matches_set,  bad_weighted_matches_set = viz_utils.create_matches_lines(bad_mask,np.array([0.0, 0.8, 0]), np.array([0.8, 0, 0.0]),source_pcd,target_matches,mask_pred_flat,weight_thr,weight_scale)

		####################################
		# Generate info for aligning source to target (by interpolating between source and warped source)
		####################################
		warped_points = np.asarray(warped_pcd.points)
		valid_source_points = np.asarray(source_object_pcd.points)
		assert warped_points.shape[0] == np.asarray(source_object_pcd.points).shape[0], f"Warp points:{warped_points.shape} Valid Source Points:{valid_source_points.shape}"
		line_segments = warped_points - valid_source_points
		line_segments_unit, line_lengths = line_mesh_utils.normalized(line_segments)
		line_lengths = line_lengths[:, np.newaxis]
		line_lengths = np.repeat(line_lengths, 3, axis=1)

		####################################
		# Draw 
		####################################

		geometry_dict = {
			"source_pcd": source_pcd, 
			"source_obj": source_object_pcd, 
			"target_pcd": target_pcd, 
			"graph":      rendered_graph,
			# "deformed_graph":    rendered_deformed_graph
		}

		alignment_dict = {
			"valid_source_points": valid_source_points,
			"line_segments_unit":  line_segments_unit,
			"line_lengths":        line_lengths
		}

		matches_dict = {
			"good_matches_set":          good_matches_set,
			"good_weighted_matches_set": good_weighted_matches_set,
			"bad_matches_set":           bad_matches_set,
			"bad_weighted_matches_set":  bad_weighted_matches_set
		}

		#####################################################################################################
		# Open viewer
		#####################################################################################################
		manager = viz_utils.CustomDrawGeometryWithKeyCallback(
			geometry_dict, alignment_dict, matches_dict
		)
		manager.custom_draw_geometry_with_key_callback()		


	def estimate_warp_field_parameters(self,input_data,show_frame=False):
		"""
			Run deformnet to get corresspondence
		"""

		# Extract result
		model_data = self.model(input_data["source"],input_data["target"],input_data["target_boundary_mask"],input_data["intrinsics"],\
			input_data["graph"]["graph_nodes"],input_data["graph"]["graph_edges"],input_data["graph"]["graph_edges_weights"],input_data["graph"]["graph_clusters"],\
			input_data["graph"]["pixel_weights"],input_data["graph"]["pixel_anchors"],\
			input_data["num_nodes"])

		if show_frame: self.plot_frame(input_data,model_data)

		return model_data

	def update_canonical_pose(self,warp_field):
		"""
			Given the warp warp_field.graph_edges = field for the previouse canonical pose,
			perform transformation to get canonical pose in target frame
			run ICP to get correspondence
			find unregistered points, 
				if they satisfy criteria: 
					add points to canonical pose
					estimate their deformation
			perform inverse transform to get the updated canonical pose  
		"""
		pass

	def run(self,source_frame=0,skip=1):
		
		"""
			Run dynamic fusion to get results
		"""

		input_data = self.dataloader.get_data(source_frame,source_frame) # Get input data
		mask = np.all(input_data["graph"]["pixel_anchors"] >= 0,axis=2)
		# Create a new tsdf volume
		tsdf = TSDFVolume(np.max(input_data["source"][-1,mask]),0.01,input_data["intrinsics"],use_gpu=True)

		color_im = 256*np.moveaxis(input_data["source"][:3,:,:],0,-1).copy()
		depth_im = input_data["source"][-1,:,:].copy()

		color_im[mask==0,:] = 0
		color_im = color_im.astype('uint8')
		depth_im[mask==0] = 0

		tsdf.integrate(color_im,depth_im)
		verts,face,norms,colors = tsdf.get_mesh()


		mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),o3d.utility.Vector3iVector(face))
		mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype('float64')/255)
		vis = o3d.visualization.Visualizer()
		vis.create_window()
		vis.add_geometry(mesh)
		opt = vis.get_render_option()
		opt.background_color = np.asarray([0, 0, 0])
		# o3d.visualization.draw_geometries([mesh])
		vis.run()
		vis.destroy_window()

		# Run from source_frame + 1 to T
		for i in range(source_frame+skip,self.dataloader.get_video_length(),skip):	
			input_data = self.dataloader.get_data(source_frame,i) # Get input data

			if updated_graph is not None: input_data["graph"] = updated_graph # Do not use orignal graph, instead use the update version based on canonical pose

			# No updates when i == source_frame
			if i != source_frame:
				# 1. source_input_data = tsdf.raycast() # First step is to get the estimated depth and RGB image
				warp_field_data = self.estimate_warp_field_parameters(input_data)
				print(f"Estimated Warpfield Parameters for Frame:{i} Info: {warp_field_data['convergence_info']}")

				# 2. create_warp_field(warp_field_data); # Create new warp field
					# 2.1. KDTree for finding nearest neigbhours 
					# 2.2. warp_voxel_space(x,y,z)
				# 3. tsdf.nonRigidTsdfFusion(warp_field,target_data); Perfrom surface fusion with the live frame
				# 4. new_mesh = tsdf.surfaceExtractionMC(); # Extract the new canonical pose using marching cubes
				# 5. new_nodes = insertNewDeformNodes(); # Find new nodes to be inserted
				# 6. warp_field.graph_edges = updateRegularizationGraph(); # Update 
				# 7. warp_field.updateKDParameters();

				

		return tsdf


	def generate_video(self,tsdf,warp_fields,save_path=None):
		pass

# Run the module
if __name__ == "__main__":
	
	seq_path = None
	source_frame = 0
	skip = 1
	if len(sys.argv) <= 1:
		raise IndexError("Usage python3 example_video.py <path to data> <source frame | optional (default=0)> <skip frame | optional (default=1)>")
	if len(sys.argv) > 1:
		seq_path = sys.argv[1]
	if len(sys.argv) > 2:
		source_frame = int(sys.argv[2])
	if len(sys.argv) > 3:
		skip = int(sys.argv[3])

	method = DynamicFusion(seq_path)
	method.run(source_frame=source_frame,skip=skip)
	# method.generate_video(tsdf,warp_fields,save_path="./results/dynamic_fusion.gif")
