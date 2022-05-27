import gc
import os
import argparse
from datetime import datetime
from shutil import copyfile
import sys
import numpy as np
from timeit import default_timer as timer
import math

from alignment import evaluate, nn_utilities, DeformNet

import torch
import open3d.core as o3c
from tensorboardX import SummaryWriter

from alignment.default import load_default_nnrt_network
from data import DeformDataset
from alignment import DeformLoss, SnapshotManager, TimeStatistics
from settings import process_arguments, Parameters
import ext_argparse

from settings.model import get_saved_model

if __name__ == "__main__":
    args = process_arguments()
    torch.set_num_threads(Parameters.training.num_threads.value)
    torch.backends.cudnn.benchmark = False

    # Training set
    train_labels_name = Parameters.training.train_labels_name.value

    # Validation set
    validation_labels_name = Parameters.training.validation_labels_name.value

    timestamp = Parameters.training.timestamp.value

    experiment_name = Parameters.training.experiment.value

    #####################################################################################
    # Ask user input regarding the use of data augmentation
    #####################################################################################
    # Confirm hyperparameters
    print("===== TRAINING HYPERPARAMETERS =====")
    ext_argparse.dump(Parameters.training)

    print()
    print("train_labels_name        ", train_labels_name)
    print("validation_labels_name          ", validation_labels_name)
    print()

    #####################################################################################
    # Creating tf writer and folders 
    #####################################################################################
    # Writer initialization.
    tf_runs = os.path.join(Parameters.path.nn_data_directory.value, "tf_runs")
    log_name = "{0}_{1}".format(timestamp, experiment_name)
    log_dir = os.path.join(tf_runs, log_name)

    train_log_dir = log_dir + "/" + train_labels_name
    val_log_dir = log_dir + "/" + validation_labels_name
    if train_log_dir == val_log_dir:
        train_log_dir = train_log_dir + "_0"
        val_log_dir = val_log_dir + "_1"

    train_writer = SummaryWriter(train_log_dir)
    val_writer = SummaryWriter(val_log_dir)

    # Copy the current options to the log directory.
    options_file_in = args.settings_file
    options_file_out = os.path.join(log_dir, "settings.yaml")
    copyfile(options_file_in, options_file_out)

    # Creation of alignment dir.
    training_models = os.path.join(Parameters.path.nn_data_directory.value, "models")
    if not os.path.exists(training_models): os.mkdir(training_models)
    saving_model_dir = os.path.join(training_models, log_name)
    if not os.path.exists(saving_model_dir): os.mkdir(saving_model_dir)

    #####################################################################################
    # Initializing: alignment, criterion, optimizer, learning scheduler...
    #####################################################################################
    # Load alignment, linear_loss and optimizer.
    saved_model = get_saved_model()

    iteration_number = 0

    if Parameters.training.use_pretrained_model.value:
        model = load_default_nnrt_network(o3c.Device.CUDA)
    else:
        model = DeformNet(False).cuda()

    # Criterion.
    criterion = DeformLoss(Parameters.training.loss.lambda_flow.value, Parameters.training.loss.lambda_graph.value,
                           Parameters.training.loss.lambda_warp.value, Parameters.training.loss.lambda_mask.value,
                           Parameters.training.loss.flow_loss_type.value)

    # Count parameters.
    n_all_model_params = int(sum([np.prod(p.size()) for p in model.parameters()]))
    n_trainable_model_params = \
        int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    print("Number of parameters: {0} / {1}".format(n_trainable_model_params, n_all_model_params))

    n_all_flownet_params = int(sum([np.prod(p.size()) for p in model.flow_net.parameters()]))
    n_trainable_flownet_params = \
        int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.flow_net.parameters())]))
    print("-> Flow network: {0} / {1}".format(n_trainable_flownet_params, n_all_flownet_params))

    n_all_masknet_params = int(sum([np.prod(p.size()) for p in model.mask_net.parameters()]))
    n_trainable_masknet_params =\
        int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.mask_net.parameters())]))
    print("-> Mask network: {0} / {1}".format(n_trainable_masknet_params, n_all_masknet_params))
    print()

    learning_parameters = Parameters.training.learning
    loss_parameters = Parameters.training.loss

    # Set up optimizer.
    if Parameters.training.learning.use_adam.value:
        optimizer = torch.optim.Adam(model.parameters(), learning_parameters.learning_rate.value,
                                     weight_decay=learning_parameters.weight_decay.value)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_parameters.learning_rate.value,
                                    momentum=learning_parameters.momentum.value,
                                    weight_decay=learning_parameters.weight_decay.value)

    # Initialize training.
    train_writer.add_text("/hyperparams",
                          "Batch size: " + str(learning_parameters.batch_size.value)
                          + ",\nLearning rate:" + str(learning_parameters.learning_rate.value)
                          + ",\nEpochs: " + str(learning_parameters.epochs.value)
                          + ",\nuse_flow_loss: " + str(loss_parameters.use_flow_loss.value)
                          + ",\nuse_graph_loss: " + str(loss_parameters.use_graph_loss.value)
                          + ",\nuse_mask: " + str(Parameters.deform_net.use_mask.value)
                          + ",\nuse_mask_loss: " + str(loss_parameters.use_mask_loss.value))

    # Initialize snaphost manager for alignment snapshot creation.
    snapshot_manager = SnapshotManager(log_name, saving_model_dir)

    # We count the execution time between evaluations.
    time_statistics = TimeStatistics()

    # Learning rate scheduler.
    if learning_parameters.use_lr_scheduler.value:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_parameters.step_lr.value, gamma=0.1, last_epoch=-1)
        for i in range(iteration_number):
            scheduler.step()

    # Compute the number of worker threads for data loading.
    # 0 means that the base thread does all the job (that makes sense when hdf5 is already loaded into memory).
    num_train_workers = Parameters.training.num_worker_threads.value
    num_val_workers = Parameters.training.num_worker_threads.value

    #####################################################################################
    # Create datasets and dataloaders
    #####################################################################################
    complete_cycle_start = timer()

    #####################################################################################
    # VAL dataset
    #####################################################################################
    val_dataset = DeformDataset(
        Parameters.path.dataset_base_directory.value, validation_labels_name,
        Parameters.alignment.image_width.value,
        Parameters.alignment.image_height.value,
        Parameters.alignment.max_boundary_distance.value
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, shuffle=Parameters.training.shuffle.value,
        batch_size=learning_parameters.batch_size.value, num_workers=num_val_workers,
        collate_fn=DeformDataset.collate_with_padding, pin_memory=True
    )

    print("Num. validation samples: {0}".format(len(val_dataset)))

    if len(val_dataset) < learning_parameters.batch_size.value:
        print()
        print("Reduce the batch_size, since we only have {} validation samples but you indicated a batch_size of {}".format(
            len(val_dataset), learning_parameters.batch_size.value)
        )
        exit()

    #####################################################################################
    # TRAIN dataset
    #####################################################################################
    train_dataset = DeformDataset(
        Parameters.path.dataset_base_directory.value, train_labels_name,
        Parameters.alignment.image_width.value,
        Parameters.alignment.image_height.value,
        Parameters.alignment.max_boundary_distance.value
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=learning_parameters.batch_size.value,
        shuffle=Parameters.training.shuffle.value, num_workers=num_train_workers,
        collate_fn=DeformDataset.collate_with_padding, pin_memory=True
    )

    print("Num. training samples: {0}".format(len(train_dataset)))
    print()

    if len(train_dataset) < learning_parameters.batch_size.value:
        print()
        print("Reduce the batch_size, since we only have {} training samples but you indicated a batch_size of {}".format(
            len(train_dataset), learning_parameters.batch_size.value)
        )
        exit()

    # Execute training.
    try:
        for epoch in range(0, learning_parameters.epochs.value):
            print()
            print()
            print("Epoch: {0}".format(epoch))

            num_consecutive_all_invalid_batches = 0

            model.train()
            for i, data in enumerate(train_dataloader):
                #####################################################################################
                # Validation.
                #####################################################################################
                if Parameters.training.do_validation.value and \
                        iteration_number % learning_parameters.evaluation_frequency.value == 0:
                    model.eval()

                    eval_start = timer()

                    # Compute train and validation metrics.
                    num_samples = Parameters.training.num_samples_eval.value
                    # We evaluate on approximately 1000 samples.
                    num_eval_batches = math.ceil(num_samples / learning_parameters.batch_size.value)

                    print()
                    print("Train evaluation")
                    train_losses, train_metrics = evaluate(model, criterion, train_dataloader, num_eval_batches, "train")

                    print()
                    print("Val   evaluation")
                    val_losses, val_metrics = evaluate(model, criterion, val_dataloader, num_eval_batches, "val")

                    train_writer.add_scalar('Loss/Loss', train_losses["total"], iteration_number)
                    train_writer.add_scalar('Loss/Flow', train_losses["flow"], iteration_number)
                    train_writer.add_scalar('Loss/Graph', train_losses["graph"], iteration_number)
                    train_writer.add_scalar('Loss/Warp', train_losses["warp"], iteration_number)
                    train_writer.add_scalar('Loss/Mask', train_losses["mask"], iteration_number)

                    train_writer.add_scalar('Metrics/EPE_2D_0', train_metrics["epe2d_0"], iteration_number)
                    train_writer.add_scalar('Metrics/EPE_2D_2', train_metrics["epe2d_2"], iteration_number)
                    train_writer.add_scalar('Metrics/Graph_Error_3D', train_metrics["epe3d"], iteration_number)
                    train_writer.add_scalar('Metrics/EPE_3D', train_metrics["epe_warp"], iteration_number)
                    train_writer.add_scalar('Metrics/ValidRatio', train_metrics["valid_ratio"], iteration_number)

                    val_writer.add_scalar('Loss/Loss', val_losses["total"], iteration_number)
                    val_writer.add_scalar('Loss/Flow', val_losses["flow"], iteration_number)
                    val_writer.add_scalar('Loss/Graph', val_losses["graph"], iteration_number)
                    val_writer.add_scalar('Loss/Warp', val_losses["warp"], iteration_number)
                    val_writer.add_scalar('Loss/Mask', val_losses["mask"], iteration_number)

                    val_writer.add_scalar('Metrics/EPE_2D_0', val_metrics["epe2d_0"], iteration_number)
                    val_writer.add_scalar('Metrics/EPE_2D_2', val_metrics["epe2d_2"], iteration_number)
                    val_writer.add_scalar('Metrics/Graph_Error_3D', val_metrics["epe3d"], iteration_number)
                    val_writer.add_scalar('Metrics/EPE_3D', val_metrics["epe_warp"], iteration_number)
                    val_writer.add_scalar('Metrics/ValidRatio', val_metrics["valid_ratio"], iteration_number)

                    print()
                    print()
                    print("Epoch number {0}, Iteration number {1}".format(epoch, iteration_number))
                    print("{:<40} {}".format("Current Train Loss TOTAL", train_losses["total"]))
                    print("{:<40} {}".format("Current Train Loss FLOW", train_losses["flow"]))
                    print("{:<40} {}".format("Current Train Loss GRAPH", train_losses["graph"]))
                    print("{:<40} {}".format("Current Train Loss WARP", train_losses["warp"]))
                    print("{:<40} {}".format("Current Train Loss MASK", train_losses["mask"]))
                    print()
                    print("{:<40} {}".format("Current Train EPE 2D_0", train_metrics["epe2d_0"]))
                    print("{:<40} {}".format("Current Train EPE 2D_2", train_metrics["epe2d_2"]))
                    print("{:<40} {}".format("Current Train EPE 3D", train_metrics["epe3d"]))
                    print("{:<40} {}".format("Current Train EPE Warp", train_metrics["epe_warp"]))
                    print("{:<40} {}".format("Current Train Solver Success Rate", train_metrics["valid_ratio"]))
                    print()

                    print("{:<40} {}".format("Current Val Loss TOTAL", val_losses["total"]))
                    print("{:<40} {}".format("Current Val Loss FLOW", val_losses["flow"]))
                    print("{:<40} {}".format("Current Val Loss GRAPH", val_losses["graph"]))
                    print("{:<40} {}".format("Current Val Loss WARP", val_losses["warp"]))
                    print("{:<40} {}".format("Current Val Loss MASK", val_losses["mask"]))
                    print()
                    print("{:<40} {}".format("Current Val EPE 2D_0", val_metrics["epe2d_0"]))
                    print("{:<40} {}".format("Current Val EPE 2D_2", val_metrics["epe2d_2"]))
                    print("{:<40} {}".format("Current Val EPE 3D", val_metrics["epe3d"]))
                    print("{:<40} {}".format("Current Val EPE Warp", val_metrics["epe_warp"]))
                    print("{:<40} {}".format("Current Val Solver Success Rate", val_metrics["valid_ratio"]))

                    print()

                    time_statistics.eval_duration = timer() - eval_start

                    # We compute the time of IO as the complete time, subtracted by all processing time.
                    time_statistics.io_duration += (timer() - complete_cycle_start - time_statistics.train_duration - time_statistics.eval_duration)

                    # Set CUDA_LAUNCH_BLOCKING=1 environmental variable for reliable timings. 
                    print("Cycle duration (s): {0:3f} (IO: {1:3f}, TRAIN: {2:3f}, EVAL: {3:3f})".format(
                        timer() - time_statistics.start_time, time_statistics.io_duration, time_statistics.train_duration,
                        time_statistics.eval_duration
                    ))
                    print("FORWARD: {0:3f}, LOSS: {1:3f}, BACKWARD: {2:3f}".format(
                        time_statistics.forward_duration, time_statistics.loss_eval_duration, time_statistics.backward_duration
                    ))
                    print()

                    time_statistics = TimeStatistics()
                    complete_cycle_start = timer()

                    sys.stdout.flush()

                    model.train()

                else:
                    sys.stdout.write("\r############# Train iteration: {0} / {1} (of Epoch {2}) - {3}".format(
                        iteration_number % learning_parameters.evaluation_frequency.value + 1,
                        learning_parameters.evaluation_frequency.value, epoch, experiment_name)
                    )
                    sys.stdout.flush()

                #####################################################################################
                # Train.
                #####################################################################################
                # Data loading.
                source, target, target_boundary_mask, \
                optical_flow_gt, optical_flow_mask, scene_flow_gt, scene_flow_mask, \
                graph_nodes, graph_edges, graph_edges_weights, translations_gt, graph_clusters, \
                pixel_anchors, pixel_weights, num_nodes, intrinsics, sample_idx = data

                source = source.cuda()
                target = target.cuda()
                target_boundary_mask = target_boundary_mask.cuda()
                optical_flow_gt = optical_flow_gt.cuda()
                optical_flow_mask = optical_flow_mask.cuda()
                scene_flow_gt = scene_flow_gt.cuda()
                scene_flow_mask = scene_flow_mask.cuda()
                graph_nodes = graph_nodes.cuda()
                graph_edges = graph_edges.cuda()
                graph_edges_weights = graph_edges_weights.cuda()
                translations_gt = translations_gt.cuda()
                graph_clusters = graph_clusters.cuda()
                pixel_anchors = pixel_anchors.cuda()
                pixel_weights = pixel_weights.cuda()
                intrinsics = intrinsics.cuda()

                train_batch_start = timer()

                #####################################################################################
                # Forward pass.
                #####################################################################################
                train_batch_forward_pass = timer()

                model_data = model(
                    source, target,
                    graph_nodes, graph_edges, graph_edges_weights, graph_clusters,
                    pixel_anchors, pixel_weights,
                    num_nodes, intrinsics
                )

                time_statistics.forward_duration += (timer() - train_batch_forward_pass)

                # Invalidate too for too far away estimations, since they can produce
                # noisy gradient information.
                if Parameters.training.gn_invalidate_too_far_away_translations.value:
                    with torch.no_grad():
                        batch_size = model_data["node_translations"].shape[0]
                        for i in range(batch_size):
                            if not model_data["valid_solve"][i]: continue

                            num_nodes_i = int(num_nodes[i])
                            assert num_nodes_i > 0

                            diff = model_data["node_translations"][i, :num_nodes_i, :] - translations_gt[i, :num_nodes_i, :]
                            epe = torch.norm(diff, p=2, dim=1)
                            mean_error = epe.sum().item() / num_nodes_i

                            if mean_error > Parameters.training.gn_max_mean_translation_error.value:
                                print("\t\tToo big mean translation error: {}".format(mean_error))
                                model_data["valid_solve"][i] = 0

                with torch.no_grad():
                    # Downscale groundtruth flow
                    flow_gts, flow_masks = nn_utilities.downscale_gt_flow(
                        optical_flow_gt, optical_flow_mask,
                        Parameters.alignment.image_height.value,
                        Parameters.alignment.image_width.value
                    )

                    # Compute mask gt for mask baseline
                    xy_coords_warped, source_points, valid_source_points, target_matches, \
                    valid_target_matches, valid_correspondences, deformed_points_idxs, \
                    deformed_points_subsampled = model_data["correspondence_info"]

                    mask_gt, valid_mask_pixels = nn_utilities.compute_baseline_mask_gt(
                        xy_coords_warped,
                        target_matches, valid_target_matches,
                        source_points, valid_source_points,
                        scene_flow_gt, scene_flow_mask, target_boundary_mask,
                        Parameters.training.baseline.max_pos_flowed_source_to_target_dist.value,
                        Parameters.training.baseline.min_neg_flowed_source_to_target_dist.value
                    )

                    # Compute deformed point gt
                    deformed_points_gt, deformed_points_mask = nn_utilities.compute_deformed_points_gt(
                        source_points, scene_flow_gt,
                        model_data["valid_solve"], valid_correspondences,
                        deformed_points_idxs, deformed_points_subsampled
                    )

                #####################################################################################
                # Loss.
                #####################################################################################
                train_batch_loss_eval = timer()

                # Compute Loss
                loss = criterion(
                    flow_gts, model_data["flow_data"], flow_masks,
                    translations_gt, model_data["node_translations"], model_data["deformations_validity"],
                    deformed_points_gt, model_data["deformed_points_pred"], deformed_points_mask,
                    model_data["valid_solve"], num_nodes,
                    model_data["mask_pred"], mask_gt, valid_mask_pixels
                )

                time_statistics.loss_eval_duration += (timer() - train_batch_loss_eval)

                #####################################################################################
                # Backprop.
                #####################################################################################
                train_batch_backprop = timer()

                # We only backprop if any of the losses is non-zero.
                if Parameters.training.loss.use_flow_loss.value or \
                        Parameters.training.loss.use_mask_loss.value or torch.sum(model_data["valid_solve"]) > 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if learning_parameters.use_lr_scheduler.value:
                        scheduler.step()

                else:
                    print("No valid linear_loss, skipping backpropagation!")

                time_statistics.backward_duration += (timer() - train_batch_backprop)

                time_statistics.train_duration += (timer() - train_batch_start)

                if iteration_number % learning_parameters.evaluation_frequency.value == 0:
                    # Store the latest alignment snapshot, if the required elased time has passed.
                    snapshot_manager.save_model(model, iteration_number)

                iteration_number = iteration_number + 1

            gc.collect()
            torch.cuda.empty_cache()

            print()
            print("Epoch {} complete".format(epoch))
            print("-------------------------------------------------------------------")
            print("-------------------------------------------------------------------")

    except (KeyboardInterrupt, TypeError, ConnectionResetError) as err:
        # We also save the latest alignment snapshot at interruption.
        snapshot_manager.save_model(model, iteration_number, final_iteration=True)
        raise err

    train_writer.close()
    val_writer.close()

    print()
    print("I'm done")
