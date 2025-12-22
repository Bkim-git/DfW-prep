# DfW-prep pipeline

DfW-prep converts raw UAV videos into a **temporally resampled**, **ortho-rectified**, and **georeferenced** grayscale video product that is suitable for subsequent **kinematic depth inversion** analyses.

---

## 1. Purpose and scope

DfW-Prep is a pre-processing pipeline that standardizes UAV imagery into a stable world-coordinate video. The pipeline combines attitude-only rectification, interactive ground control point (GCP) tracking, frame-wise georeferencing, and global co-registration onto a common grid.

---

## 2. Processing workflow

DfW-Prep executes the following stages in sequence:

1. **Load**
   - Read configuration and inputs
   - Resample video to a target frame rate
   - Cache the resampled grayscale sequence to disk

2. **Orthorectification**
   - Apply a planar homography derived from **pitch** and **roll** to stabilize the horizon and reduce perspective distortion

3. **GCP tracking**
   - Select GCPs in the first frame
   - Track GCPs through time using pyramidal **Lucas–Kanade optical flow**
   - Detect anomalous tracking events and handle them using the pipeline’s anomaly logic

4. **Georeferencing and warping**
   - For each frame, estimate a **2D similarity transform** (scaled rotation plus translation) mapping tracked pixel GCPs to fixed world GCPs
   - Optionally apply temporal smoothing to the tracked GCP trajectories
   - Warp each frame into world coordinates using a user-defined **ground sampling distance (GSD)**
   - Record the per-frame world-coordinate extent

5. **Visualization**
   - Render a diagnostic movie that overlays warped frames on an optional basemap and GCP markers

6. **Co-registration**
   - Resample all warped frames onto a **single global world grid**
   - Export the final standardized video and a global extent file, formatted for direct use as inputs to the DfW algorithm


### Compact flow (conceptual)

```text
Raw videos (.mp4 / .mov)
  -> temporal resampling to target FPS and grayscale
  -> pitch-roll homography rectification
  -> GCP selection and optical-flow tracking
  -> frame-wise similarity transform estimation
  -> per-frame world warp and extent computation
  -> global-grid co-registration and export
```

---
## 3. Directory structure and required inputs

Each processing case must follow a fixed directory structure to ensure consistent data handling.

- Case identifier: `Case`
- Inputs directory: `Cases/Case/inputs`
- Outputs directory: `Cases/Case/outputs`

### Recommended layout

```text
Cases/
  Case/
    inputs/
      params.yaml
      gcps.csv
      basemap.png                # optional
      *.mp4                      # one or more
      *.mov                      # one or more
    outputs/
      __Vid_resampled.npy
      __Initial-selection.npy
      __gcps_px_predefined.npy
      Visualization.mp4
      Processed.mp4
      extent.txt
```

### Required input files (`Cases/Case/inputs`)

- `params.yaml`  
  Configuration file specifying the processing parameters. This includes, but is not limited to:
  - target frame rate `FPS`
  - camera pitch and roll angles `PITCH`, `ROLL`
  - camera focal length `FOCAL_LENGTH`
  - desired ground sampling distance `GSD`
  - parameters controlling GCP tracking and optional temporal smoothing

- `gcps.csv`  
  Comma-separated file defining the ground control points in a fixed world coordinate system. The pipeline interprets the numeric values in **columns 2 and 3** as planar world coordinates (Easting, Northing) associated with each GCP.

- Raw video files (`.mp4` or `.mov`)  
  One or more raw UAV video files constituting the primary observational input to the pipeline.

### Optional input files (`Cases/Case/inputs`)

- `basemap.png`  
  A georeferenced or visually aligned basemap image used exclusively for diagnostic and qualitative visualization. This file does not influence the numerical processing stages or the resulting georeferencing.

## 4. Output products and formats

All output products are written to `Cases/Case/outputs`. The outputs are designed to support both intermediate quality control and direct downstream use in the DfW depth inversion stage.

- `__Vid_resampled.npy`  
  Cached NumPy array containing the grayscale video sequence after temporal resampling to the target frame rate. This prevents repeated decoding and resampling of the raw video files across runs.

- `__Initial-selection.npy`  
  NumPy file storing the pixel coordinates of the ground control points (GCPs) manually selected in the first frame. These coordinates define the initial conditions for subsequent GCP tracking.

- `__gcps_px_predefined.npy`  
  NumPy array containing the tracked GCP pixel coordinates for all frames, with shape `T x N_gcp x 2`, where `T` is the number of frames and `N_gcp` is the number of GCPs.

- `Visualization.mp4`  
  Diagnostic video showing the georeferenced frames overlaid on the optional basemap and GCP markers. This output is intended for qualitative assessment and does not affect numerical processing.

- `Processed.mp4`  
  Final co-registered grayscale video in which all frames are resampled onto a single global world-coordinate grid. This standardized product is the primary video input to the DfW depth inversion algorithm.

- `extent.txt`  
  Text file specifying the global world-coordinate extent associated with `Processed.mp4`, formatted as `(x_min, x_max, y_min, y_max)`. This file provides the spatial referencing required by the downstream DfW processing stage.
