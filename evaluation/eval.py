"""
Evaluate HUMUS-Net models trained on the fastMRI dataset.
Make sure to define the following for the validation dataset:
    --checkpoint_file: path to the saved model checkpoint to be evaluated 
    --data_path: path to the fastMRI dataset root
    --gpus: number of GPUs for validation
    --accelerations: undersampling ratio in kspace domain, 8 in all experiments
    --center_fractions: describes number of center lines used in the mask,  0.04 in all experiments
    --mask_type: random is used in all experiments
"""
import os, sys
import pathlib
from argparse import ArgumentParser
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute())   )

import pytorch_lightning as pl
import torch
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from  pl_modules.humus_module import HUMUSNetModule

from data.data_transforms import HUMUSNetDataTransform
from pl_modules.fastmri_data_module import FastMriDataModule

# Imports for logging and other utility
from pytorch_lightning.plugins import DDPPlugin
import yaml
from utils import load_args_from_config

# Global variables for saving
SAVE_IMAGES = True
OUTPUT_DIR = None
BATCH_COUNTER = 0

def set_output_directory(output_dir):
    """Set the output directory for saving images"""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reconstructions'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    print(f"Output directory set to: {output_dir}")

def normalize_for_display(image):
    """Normalize image for display"""
    if image is None:
        return None
    
    # Convert to magnitude if complex
    if np.iscomplexobj(image):
        image = np.abs(image)
    
    # Normalize to [0, 1]
    img_min, img_max = float(image.min()), float(image.max())
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image)
    
    return image

def save_reconstruction_data(target, reconstruction, fname, slice_num, batch_idx):
    """Save reconstruction data as h5 and create visualization"""
    global OUTPUT_DIR, BATCH_COUNTER
    
    if not SAVE_IMAGES or not OUTPUT_DIR:
        return
    
    try:
        # Clean filename for saving
        fname_clean = str(fname).replace('/', '_').replace('\\', '_')
        base_filename = f"{fname_clean}_slice_{slice_num:03d}"
        
        # Save as h5 file
        h5_path = os.path.join(OUTPUT_DIR, 'reconstructions', f"{base_filename}.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('target', data=target, compression='gzip')
            f.create_dataset('reconstruction', data=reconstruction, compression='gzip')
            f.attrs['filename'] = base_filename
            f.attrs['target_shape'] = str(target.shape)
            f.attrs['reconstruction_shape'] = str(reconstruction.shape)
        
        # Create visualization
        create_comparison_visualization(target, reconstruction, base_filename)
        
        if BATCH_COUNTER % 10 == 0:
            print(f"Saved reconstruction for {base_filename}")
        
        BATCH_COUNTER += 1
        
    except Exception as e:
        print(f"Error saving reconstruction for batch {batch_idx}: {e}")

def create_comparison_visualization(target, reconstruction, filename):
    """Create and save visualization comparing target and reconstruction"""
    try:
        # Handle different data shapes
        if len(target.shape) > 2:
            target_slice = target[0] if target.shape[0] == 1 else target[target.shape[0]//2]
            recon_slice = reconstruction[0] if reconstruction.shape[0] == 1 else reconstruction[reconstruction.shape[0]//2]
        else:
            target_slice = target
            recon_slice = reconstruction
        
        # Normalize for display
        target_norm = normalize_for_display(target_slice)
        recon_norm = normalize_for_display(recon_slice)
        
        # Calculate error map
        error_map = np.abs(target_norm - recon_norm)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Target
        im1 = axes[0].imshow(target_norm, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Target (Ground Truth)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Reconstruction
        im2 = axes[1].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('HUMUS-Net Reconstruction')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Error map
        im3 = axes[2].imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
        axes[2].set_title('Reconstruction Error')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        # Calculate metrics
        mse = np.mean((target_norm - recon_norm) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        corr = np.corrcoef(target_norm.flatten(), recon_norm.flatten())[0, 1]
        
        plt.suptitle(f'{filename}\nPSNR: {psnr:.2f} dB | MSE: {mse:.6f} | Corr: {corr:.4f}', fontsize=12)
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(OUTPUT_DIR, 'visualizations', f"{filename}_comparison.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualization for {filename}: {e}")
        plt.close('all')

# Monkey patch the validation_step method to add saving
original_validation_step = HUMUSNetModule.validation_step

def enhanced_validation_step(self, batch, batch_idx):
    """Enhanced validation step that saves reconstructions"""
    # Call original validation step
    output = original_validation_step(self, batch, batch_idx)
    
    # Save reconstruction if enabled
    if SAVE_IMAGES and OUTPUT_DIR:
        try:
            target = output['target'].cpu().numpy()
            reconstruction = output['output'].cpu().numpy()
            fname = output['fname'][0] if isinstance(output['fname'], (list, tuple)) else str(output['fname'])
            slice_num = output['slice_num'].item() if hasattr(output['slice_num'], 'item') else int(output['slice_num'])
            
            save_reconstruction_data(target, reconstruction, fname, slice_num, batch_idx)
        except Exception as e:
            print(f"Error in enhanced validation step: {e}")
    
    return output

# Apply the monkey patch
HUMUSNetModule.validation_step = enhanced_validation_step

def cli_main(args):
    pl.seed_everything(args.seed)

    # Set up output directory
    if hasattr(args, 'output_dir') and args.output_dir:
        set_output_directory(str(args.output_dir))
    else:
        # Default output directory next to checkpoint
        checkpoint_dir = os.path.dirname(args.checkpoint_file)
        default_output = os.path.join(checkpoint_dir, 'evaluation_results')
        set_output_directory(default_output)

    # ------------
    # model
    # ------------
    if args.challenge == 'multicoil':
        model = HUMUSNetModule.load_from_checkpoint(args.checkpoint_file)
        hparams = torch.load(args.checkpoint_file)['hyper_parameters']
        num_adj_slices = hparams['num_adj_slices']
        uniform_train_resolution = hparams.get('img_size', hparams.get('uniform_train_resolution', None))
    else:
        raise ValueError('Single-coil data not supported.')
    model.eval()
    
    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    
    # use fixed masks for val transform
    val_transform = HUMUSNetDataTransform(uniform_train_resolution=uniform_train_resolution, mask_func=mask)
    
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=None,
        val_transform=val_transform,
        test_transform=None,
        test_split=None,
        test_path=None,
        sample_rate=getattr(args, 'sample_rate', None),
        volume_sample_rate=getattr(args, 'volume_sample_rate', 1.0),
        batch_size=1,
        num_workers=getattr(args, 'num_workers', 4),
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        combine_train_val=False,
        train_scanners=getattr(args, 'train_scanners', None),
        val_scanners=getattr(args, 'val_scanners', None),
        combined_scanner_val=getattr(args, 'combined_scanner_val', False),
        num_adj_slices=num_adj_slices,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, plugins=DDPPlugin(find_unused_parameters=False), logger=False)
        
    # ------------
    # run
    # ------------
    print(f"Starting evaluation with image saving...")
    print(f"Checkpoint: {args.checkpoint_file}")
    print(f"Data path: {args.data_path}")
    print(f"Results will be saved to: {OUTPUT_DIR}")
    print(f"Acceleration: {args.accelerations}")
    print(f"Center fractions: {args.center_fractions}")
    print("-" * 60)
    
    trainer.validate(model, datamodule=data_module)
    
    # Print summary
    if OUTPUT_DIR:
        print(f"\n" + "="*60)
        print(f"VALIDATION COMPLETE!")
        print(f"Reconstructions saved to: {os.path.join(OUTPUT_DIR, 'reconstructions')}")
        print(f"Visualizations saved to: {os.path.join(OUTPUT_DIR, 'visualizations')}")
        
        # Count saved files
        recon_dir = os.path.join(OUTPUT_DIR, 'reconstructions')
        viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')
        
        if os.path.exists(recon_dir):
            h5_files = len([f for f in os.listdir(recon_dir) if f.endswith('.h5')])
            print(f"Total H5 files saved: {h5_files}")
        
        if os.path.exists(viz_dir):
            png_files = len([f for f in os.listdir(viz_dir) if f.endswith('.png')])
            print(f"Total visualization files saved: {png_files}")
        
        print(f"="*60)

def build_args():
    parser = ArgumentParser()

    # basic args
    backend = "ddp"
    batch_size = 1

    # client arguments
    parser.add_argument(
        '--checkpoint_file', 
        type=pathlib.Path,          
        help='Path to the checkpoint to load the model from.',
    )
    
    # Add output directory argument
    parser.add_argument(
        '--output_dir',
        type=pathlib.Path,
        default=None,
        help='Directory to save reconstructed images and visualizations. If not specified, saves next to checkpoint.',
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.04],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[8],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        challenge="multicoil",
        mask_type="random",  # random masks for knee data
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
        accelerations=[8], # default experimental setup: 8x acceleration
        center_fractions=[0.04]
    )
    
    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=1,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
    )

    args = parser.parse_args()

    return args

def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)

if __name__ == "__main__":
    run_cli()