#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image  

from GRAINet import run_train, run_test, create_plots, run_prediction_orthophoto, read_rgb, setup_parser, collect_cv_data, create_k_fold_split_indices



#### Setup argument parser with default values
parser = setup_parser()
args, unknown = parser.parse_known_args()

# set output directory
parent_dir = 'grainnet_workdir'
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

# image dataset with ground truth 
args.data_npz_path = os.path.join('data', 'preprocessed_data.npz')

# full orthophoto image (for predicting a map)
args.image_path = os.path.join('data', 'orthoimage.tif')

# manually created mask to select regions of interest on the gravel bar
ortho_mask_path = os.path.join('data', 'mask.tif')

# evaluation metrics
metrics_keys = ('kld', 'iou')
# metrics_keys = ('mae', 'rmse')

args.output_dm = True
args.volume_weighted = True
args.loss_key = 'kld'
args.verbose = 1
args.nb_epoch = 20



### Random cross-validation: Split data into 10 random folds (non-overlapping subsets)
num_folds= 10

# load dataset
data = np.load(args.data_npz_path, allow_pickle=True)

# set output path to save indices (dataset wrapper will loaded these indices)
args.randCV_indices_path = os.path.join(parent_dir, 'random_{}_fold_indices.npy'.format(num_folds))

# create the non-overlapping data splits
indices_list = create_k_fold_split_indices(data=data, out_path=args.randCV_indices_path, num_folds=num_folds)



### Run the training and evaluation
# to evaluate over all samples:
N_runs = num_folds 

for test_fold_index in range(N_runs):
    args.test_fold_index = test_fold_index
    
    args.experiment_dir = os.path.join(parent_dir, 'loss_{}'.format(args.loss_key), 'testfold_{}'.format(args.test_fold_index))
    print('******************')
    print('TEST FOLD: ', args.test_fold_index)
    print(args.experiment_dir)

    # train the CNN
    print('training...')
    run_train(args)
    
    # test the best solution on the test data
    print('testing...')
    run_test(args)
    create_plots(args)



#### Collect and evaluate cross-validation results  
_, _, dm_results_dict = collect_cv_data(parent_dir=parent_dir, loss_keys=(args.loss_key,))

print('Number of test samples: ', dm_results_dict[args.loss_key]['dm_pred'].shape)
print('Mean absolute error: {:.1f} cm'.format(np.mean(np.abs(dm_results_dict[args.loss_key]['dm_true'] - dm_results_dict[args.loss_key]['dm_pred']))))



### Plot test results
if args.output_dm:
    mi, ma = 0, 20
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.scatter(dm_results_dict[args.loss_key]['dm_true'], dm_results_dict[args.loss_key]['dm_pred'])
    plt.xlabel('Ground truth mean diameter [cm]')
    plt.ylabel('Predicted mean diameter [cm]')
    plt.axis('square')
    plt.xlim(mi, ma)
    plt.ylim(mi, ma)
    plt.plot((mi, ma), (mi, ma), 'k--')
    plt.grid()
    plt.show()



### Plot test samples in the first subset (testfold with index 0)

N_plots = len(indices_list[0])  # number of samples in fold 0
fig, axes = plt.subplots(nrows=2, ncols=int(N_plots/2), figsize=(15,10))
axes = axes.flatten()
predictions_fold_0 = dm_results_dict[args.loss_key]['dm_pred'][0:N_plots]

# sort samples based on predictions
sorted_indices = np.argsort(predictions_fold_0)

for ax_i in range(min(len(sorted_indices),len(axes))):
    ax=axes[ax_i]
    i = sorted_indices[ax_i]
    sample_i = indices_list[0][i]

    ax.imshow(np.array(data['images'][sample_i], dtype=np.uint8))
    ax.axis('off')
    ax.set_title('Pred: {:.1f}\nTrue: {:.1f}'.format(dm_results_dict[args.loss_key]['dm_pred'][i], dm_results_dict[args.loss_key]['dm_true'][i]))    
        
plt.tight_layout()



### Predict the mean diameter map for the entire orthophoto
if args.output_dm:
    # predict with model from last fold
    args.inference_path = args.experiment_dir
    predictions_ortho, predictions_2D_map, orthophoto = run_prediction_orthophoto(args)
    np.save("output_demo_dm/predictions_2D_map_ortho_3_8_20_10f",predictions_2D_map)



### Visualize orthophoto and prediction map (mean diameter)
# mask the prediction map
Image.MAX_IMAGE_PIXELS = None
mask = np.array(Image.open(ortho_mask_path))
predictions_2D_map_masked = np.ma.masked_where(mask!=1, predictions_2D_map)

if args.output_dm:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

    # plot ortho RGB image
    orthophoto = np.array(orthophoto, dtype=np.uint8)
    axes[0].imshow(orthophoto)
    axes[0].axis('off')

    # plot prediction map
    pred_map = axes[1].imshow(predictions_2D_map_masked, aspect=891/356, cmap='cividis', 
                              vmin=0, vmax=15, interpolation='nearest')
    plt.colorbar(mappable=pred_map, label='Grain size: mean diameter [cm]')
    axes[1].grid()