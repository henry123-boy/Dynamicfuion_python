# The code in the this file creates a class to run deformnet using torch
import os
import torch

# Modules (make sure modules are visible in sys.path)
from model.model import DeformNet

import options as opt

class Deformnet_runner():
	"""
		Runs deformnet to outputs result
	"""
	def __init__(self):

		#####################################################################################################
		# Options
		#####################################################################################################

		# We will overwrite the default value in options.py / settings.py
		opt.use_mask = True
		
		#####################################################################################################
		# Load model
		#####################################################################################################

		saved_model = opt.saved_model

		assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."
		pretrained_dict = torch.load(saved_model)

		# Construct model
		self.model = DeformNet().cuda()

		if "chairs_things" in saved_model:
			self.model.flow_net.load_state_dict(pretrained_dict)
		else:
			if opt.model_module_to_load == "full_model":
				# Load completely model            
				self.model.load_state_dict(pretrained_dict)
			elif opt.model_module_to_load == "only_flow_net":
				# Load only optical flow part
				model_dict = self.model.state_dict()
				# 1. filter out unnecessary keys
				pretrained_dict = {k: v for k, v in pretrained_dict.items() if "flow_net" in k}
				# 2. overwrite entries in the existing state dict
				model_dict.update(pretrained_dict) 
				# 3. load the new state dict
				self.model.load_state_dict(model_dict)
			else:
				print(opt.model_module_to_load, "is not a valid argument (A: 'full_model', B: 'only_flow_net')")
				exit()

		self.model.eval()

	def __call__(self,source,target,target_boundary_mask,intrinsics,\
			graph_nodes,graph_edges,graph_edges_weights,graph_clusters,\
			pixel_weights,pixel_anchors,\
			num_nodes):
		
		#####################################################################################################
		# Predict deformation
		#####################################################################################################

		# Move to device and unsqueeze in the batch dimension (to have batch size 1)
		source_cuda               = torch.from_numpy(source).cuda().unsqueeze(0)
		target_cuda               = torch.from_numpy(target).cuda().unsqueeze(0)
		target_boundary_mask_cuda = torch.from_numpy(target_boundary_mask).cuda().unsqueeze(0)
		graph_nodes_cuda          = torch.from_numpy(graph_nodes).cuda().unsqueeze(0)
		graph_edges_cuda          = torch.from_numpy(graph_edges).cuda().unsqueeze(0)
		graph_edges_weights_cuda  = torch.from_numpy(graph_edges_weights).cuda().unsqueeze(0)
		graph_clusters_cuda       = torch.from_numpy(graph_clusters).cuda().unsqueeze(0)
		pixel_anchors_cuda        = torch.from_numpy(pixel_anchors).cuda().unsqueeze(0)
		pixel_weights_cuda        = torch.from_numpy(pixel_weights).cuda().unsqueeze(0)
		intrinsics_cuda           = torch.from_numpy(intrinsics).cuda().unsqueeze(0)

		num_nodes_cuda            = torch.from_numpy(num_nodes).cuda().unsqueeze(0)

		# Run Neural Non Rigid tracking and obtain results
		with torch.no_grad():
			model_data = self.model(
				source_cuda, target_cuda, 
				graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda, 
				pixel_anchors_cuda, pixel_weights_cuda, 
				num_nodes_cuda, intrinsics_cuda, 
				evaluate=True, split="test"
			)

		# Post Process output   
		model_data["node_rotations"]    = model_data["node_rotations"].view(num_nodes, 3, 3).cpu().numpy()
		model_data["node_translations"] = model_data["node_translations"].view(num_nodes, 3).cpu().numpy()
		
		assert model_data["mask_pred"] is not None, "Make sure use_mask=True in options.py"
		model_data["mask_pred"] = model_data["mask_pred"].view(-1, opt.image_height, opt.image_width).cpu().numpy()

		# Correspondence info
		xy_coords_warped,\
		source_points, valid_source_points,\
		target_matches, valid_target_matches,\
		valid_correspondences, deformed_points_idxs, deformed_points_subsampled = model_data["correspondence_info"]

		model_data["target_matches"]        = target_matches.view(-1, opt.image_height, opt.image_width).cpu().numpy()
		model_data["valid_source_points"]   = valid_source_points.view(-1, opt.image_height, opt.image_width).cpu().numpy()
		# model_data["valid_target_matches"]  = valid_target_matches.view(-1, opt.image_height, opt.image_width).cpu().numpy()
		model_data["valid_correspondences"] = valid_correspondences.view(-1, opt.image_height, opt.image_width).cpu().numpy()
		model_data["deformed_graph_nodes"] = graph_nodes + model_data["node_translations"]

		return model_data
