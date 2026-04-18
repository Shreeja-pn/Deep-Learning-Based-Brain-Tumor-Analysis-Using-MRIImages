import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from monai.networks.nets import UNet
from scipy.ndimage import zoom

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Resize Volume
# -------------------------------------------------
def resize_volume(volume, target_shape=(128, 128, 128)):
    factors = (
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2],
    )
    return zoom(volume, factors, order=1)


# -------------------------------------------------
# Normalize MRI
# -------------------------------------------------
def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-5)


# -------------------------------------------------
# Load Segmentation Model
# -------------------------------------------------
def load_model(model_path):

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


# -------------------------------------------------
# Tumor Volume Calculation
# -------------------------------------------------
def calculate_tumor_volumes(pred_mask_path):

    nii = nib.load(pred_mask_path)
    mask = nii.get_fdata()

    voxel_spacing = nii.header.get_zooms()
    voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]

    volumes = {}

    labels = {
        1: "Necrotic Core",
        2: "Edema",
        3: "Enhancing Tumor"
    }

    for label, name in labels.items():

        voxel_count = np.sum(mask == label)
        volume_mm3 = voxel_count * voxel_volume
        volume_cm3 = volume_mm3 / 1000

        volumes[name] = round(volume_cm3, 2)

    total_voxels = np.sum(mask > 0)
    total_volume_cm3 = (total_voxels * voxel_volume) / 1000

    volumes["Total Tumor Volume"] = round(total_volume_cm3, 2)

    return volumes


# -------------------------------------------------
# Extract Tumor Crop (for classification)
# -------------------------------------------------
def extract_tumor_crop(flair_volume, pred_mask):

    coords = np.where(pred_mask > 0)

    if len(coords[0]) == 0:
        z = pred_mask.shape[2] // 2
        return flair_volume[:, :, z]

    z = coords[2][len(coords[2]) // 2]

    flair_slice = flair_volume[:, :, z]
    mask_slice = pred_mask[:, :, z]

    y, x = np.where(mask_slice > 0)

    if len(y) == 0:
        return flair_slice

    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()

    crop = flair_slice[y_min:y_max, x_min:x_max]

    return crop


# -------------------------------------------------
# Main Prediction Function
# -------------------------------------------------
def predict(model, t1_path, t1ce_path, t2_path, flair_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    # -----------------------------
    # Load MRI volumes
    # -----------------------------
    t1_nii = nib.load(t1_path)

    t1 = normalize(t1_nii.get_fdata())
    t1ce = normalize(nib.load(t1ce_path).get_fdata())
    t2 = normalize(nib.load(t2_path).get_fdata())
    flair = normalize(nib.load(flair_path).get_fdata())

    # -----------------------------
    # Resize volumes
    # -----------------------------
    t1 = resize_volume(t1)
    t1ce = resize_volume(t1ce)
    t2 = resize_volume(t2)
    flair = resize_volume(flair)

    # -----------------------------
    # Stack channels
    # -----------------------------
    volume = np.stack([t1, t1ce, t2, flair], axis=0)
    volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).to(device)

    # -----------------------------
    # Segmentation inference
    # -----------------------------
    with torch.no_grad():

        output = model(volume_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]

    # -----------------------------
    # Extract tumor crop
    # -----------------------------
    tumor_crop = extract_tumor_crop(flair, pred)

    crop_path = os.path.join(output_folder, "tumor_crop.png")

    plt.figure(figsize=(4,4))
    plt.imshow(tumor_crop, cmap="gray")
    plt.axis("off")
    plt.savefig(crop_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # -----------------------------
    # Save predicted mask
    # -----------------------------
    pred_mask_path = os.path.join(output_folder, "pred_mask.nii.gz")

    pred_nii = nib.Nifti1Image(pred.astype(np.uint8), t1_nii.affine)
    nib.save(pred_nii, pred_mask_path)

    # -----------------------------
    # Create overlay visualization
    # -----------------------------
    slice_idx = pred.shape[-1] // 2

    flair_slice = flair[:, :, slice_idx]
    pred_slice = pred[:, :, slice_idx]

    overlay = np.zeros((*pred_slice.shape, 4))

    overlay[pred_slice == 1] = [1, 1, 0, 0.4]   # yellow
    overlay[pred_slice == 2] = [1, 0, 0, 0.5]   # red
    overlay[pred_slice == 3] = [0, 1, 1, 0.5]   # cyan

    overlay_path = os.path.join(output_folder, "overlay.png")

    plt.figure(figsize=(6,6))
    plt.imshow(flair_slice, cmap="gray")
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # -----------------------------
    # Calculate tumor volumes
    # -----------------------------
    volumes = calculate_tumor_volumes(pred_mask_path)

    return overlay_path, volumes