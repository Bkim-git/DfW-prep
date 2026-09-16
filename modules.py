import os, cv2, yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter

matplotlib.rcParams["font.family"] = "Arial"

def Load(Case, lenvid = None, verbose = True):
    """
    Load grayscale video frames with optional temporal downsampling
    and corresponding GCP coordinates.
    
    Parameters
    ----------
    Case : str
        Case directory name under "Cases/".
    vidfps : float or None
        Target frame rate for temporal subsampling.
    lenvid : float or None
        Maximum video length in seconds to load.
    
    Returns
    -------
    params : dict
        Parameter dictionary
    """
    
    # DIRECTORY GENERATION
    cache = os.path.join("__pycache__")
    inputdir = os.path.join("Cases", Case, "inputs")
    savedir = os.path.join("Cases", Case, "outputs")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    
    # LOAD YAML PARAMETERS
    config_path = os.path.join(inputdir, 'params.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)
    else:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    # LOAD GCPs
    gcppath = os.path.join(inputdir, "gcps.csv")
    gcps = np.genfromtxt(gcppath, delimiter=",", skip_header=1, dtype=str)
    
    params.update({'savedir': savedir})    
    
    if verbose:
        print('----------------- Parameters -----------------')
        for key, value in params.items():
            print(f'{key} : {value}')
        print('----------------------------------------------\n')
    
    # LOAD BASEMAP
    path = os.path.join(inputdir, "basemap.png")
    basemap = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) if os.path.exists(path) else None

    params.update({'basemap': basemap,
                   'gcps': gcps[:,1:3].astype(np.float32)
                   })

    # LOAD VIDEOS
    vidfps = params["FPS"]
    path = os.path.join(savedir, "__Vid_resampled.npy")
    
    if not os.path.exists(path):
        vids = sorted(
            os.path.join(inputdir, f)
            for f in os.listdir(inputdir)
            if f.lower().endswith((".mp4", ".mov"))
        )
            
        if not vids:
            raise FileNotFoundError(f"No video files found in {inputdir}")
    
        frames = []
    
        for vid_idx, filename in enumerate(vids):
            cap = cv2.VideoCapture(filename)
            fps = cap.get(cv2.CAP_PROP_FPS)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
            total_frames = nframes if lenvid is None else min(
                nframes, int(np.floor(lenvid * fps))
            )
    
            if vidfps is None or vidfps >= fps:
                sample_dt = 1.0 / fps
            else:
                sample_dt = 1.0 / vidfps
    
            next_sample_time = 0.0
    
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
    
                t = i / fps
    
                if t + 1e-9 >= next_sample_time:
                    if verbose:
                        percent = 100.0 * (i + 1) / total_frames
                        print(
                            f"\r- Importing video data: (Video {vid_idx + 1}/{len(vids)}, "
                            f"{os.path.basename(filename)}) "
                            f"Frame {i + 1}/{total_frames}({percent:.0f}%) |     ",
                            end="",
                        )
    
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    next_sample_time += sample_dt
            
            if len(frames) > lenvid*vidfps:
                break
            
            cap.release()

        np.save(path, frames, allow_pickle=True)
        if verbose:  
             print()
    
    else:
        print("- Importing video data: Resampled video matrix exists")
    
    # sys.stdout.write("\n")
    return params

#%%
class Orthorectification:
    """
    Pitch–roll-only image rectification using a planar homography.
    """
    R_BODY_TO_CAM = np.array(
        [[-1, 0, 0],
         [ 0,-1, 0],
         [ 0, 0,-1]],
        dtype=float
    )

    def __init__(self, params):
        self.dir    = params["savedir"]
        path = os.path.join(self.dir, "__Vid_resampled.npy")

        self.video = np.load(path, mmap_mode="r")
        self.frames = len(self.video)
        self.H0, self.W0 = self.video[0].shape
        self.gsd_ratio = params.get("RECTIFY_GSD_RATIO", 0.5)
        
        self.pitch = np.pi / 2 + np.deg2rad(params["PITCH"])
        self.roll = np.deg2rad(params["ROLL"])

        focal_length = params["FOCAL_LENGTH"]
        self.K = np.array(
            [[focal_length, 0,  self.W0 / 2],
             [0,  focal_length, self.H0 / 2],
             [0,  0,  1]],
            dtype=float
        )

        R_cam = self.R_camera_PR(self.pitch, self.roll)
        self.H = self.K @ R_cam @ np.linalg.inv(self.K)
        
    @staticmethod
    def R_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1,0,0],
                         [0,c,-s],
                         [0,s,c]], float)
    @staticmethod
    def R_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[ c,0,s],
                         [ 0,1,0],
                         [-s,0,c]], float)
    @classmethod
    def R_camera_PR(cls, pitch, roll):
        """
        Camera rotation matrix including body-to-camera convention.
        """
        Rpr = cls.R_x(pitch) @ cls.R_y(roll)
        return cls.R_BODY_TO_CAM @ Rpr

    def compute_output_geometry(self):
        """
        Compute output image size and homography translation so that
        the warped image is fully contained, with controlled output GSD.
    
        Returns
        -------
        out_W : int
            Output image width.
        out_H : int
            Output image height.
        T : ndarray
            Translation matrix (3x3).
        S : ndarray
            Isotropic scaling matrix (3x3).
        """
        ratio = self.gsd_ratio  # e.g. 0.3
    
        corners = np.array(
            [[0,       0,       1],
             [self.W0, 0,       1],
             [self.W0, self.H0, 1],
             [0,       self.H0, 1]],
            dtype=float
        ).T
    
        warped = self.H @ corners
        warped /= warped[2]
    
        # -------------------------------------------------
        # apply output scaling (GSD redefinition)
        # -------------------------------------------------
        S = np.array(
            [[ratio, 0, 0],
             [0, ratio, 0],
             [0, 0,     1]],
            dtype=float
        )
    
        warped_s = S @ warped
    
        xmin, xmax = warped_s[0].min(), warped_s[0].max()
        ymin, ymax = warped_s[1].min(), warped_s[1].max()
    
        out_W = int(np.ceil(xmax - xmin))
        out_H = int(np.ceil(ymax - ymin))
    
        T = np.array(
            [[1, 0, -xmin],
             [0, 1, -ymin],
             [0, 0,    1]],
            dtype=float
        )
    
        return out_W, out_H, T, S

    def run(self, replace=False, verbose=True):
        """
        Apply pitch–roll rectification with controlled output GSD.
        """
        out_W, out_H, T, S = self.compute_output_geometry()
    
        # full warp including GSD scaling
        H_warp = T @ S @ self.H
    
        rectified = []
    
        for i, frame in enumerate(self.video):
            if verbose:
                percent = 100.0 * (i + 1) / self.frames
                print(
                    f"\r- Rectification in progress: {percent:3.0f} % | "
                    f"Frame {i + 1}/{self.frames}",
                    end="",
                )
    
            rect = cv2.warpPerspective(
                frame,
                H_warp,
                (out_W, out_H),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
    
            rectified.append(rect)
        
        if verbose:
            print()

        return rectified
    
#%%
class Track_GCPs:
    def __init__(self, vid, params):
        """
        Initialize the GCP selector and tracker.
    
        Parameters
        ----------
        vid : list or ndarray
            Sequence of image frames.
        numgcps : int
            Number of GCPs to be selected and tracked.
        max_displacement : float
            Threshold for allowable displacement ratio between consecutive frames.
        """
        self.vid              = vid
        self.original_image   = vid[0].copy()
        self.image            = self.original_image.copy()
        self.numgcps          = len(params["gcps"])
        self.dragging         = False
        self.selected_points  = []
        self.start_x          = self.start_y = self.end_x = self.end_y = 0
        self.windowname       = f"SELECT {self.numgcps} POINTS"
        self.gcps_px          = None
        self.max_displacement = params["MAX_PIXEL_DISPLACEMENT"]
        self.winsiz           = params["WINDOW_SIZE"]
        self.maxpyrlevel      = params["MAX_PYRAMID_LEVEL"]
        self.current_roi      = [0, self.original_image.shape[1], 0, self.original_image.shape[0]]
        self.savedir          = params["savedir"]

    def calculate_original_coords(self, x, y):
        """
         Convert display coordinates to original image coordinates
         based on the current ROI.
         """
        xmin, xmax, ymin, ymax = self.current_roi
        h, w = self.image.shape[:2]
        ox = int(xmin + x * (xmax - xmin) / w)
        oy = int(ymin + y * (ymax - ymin) / h)
        return ox, oy

    def update_display(self):
        """
        Update the displayed image according to the current ROI
        and overlay selected GCPs.
        """
        xmin, xmax, ymin, ymax = self.current_roi
        roi = self.original_image[max(0, ymin):max(1, ymax), max(0, xmin):max(1, xmax)].copy()
        for px, py in self.selected_points:
            if xmin <= px < xmax and ymin <= py < ymax:
                cv2.circle(roi, (int(px - xmin), int(py - ymin)), 5, (0, 0, 255), -1)
        self.image = roi
        cv2.imshow(self.windowname, roi)

    def reset_zoom(self):
        """
        Reset the ROI to the full image.
        """
        self.current_roi = [0, self.original_image.shape[1], 0, self.original_image.shape[0]]

    def click_event(self, event, x, y, flags, param):
        """
        Mouse callback for interactive zooming and GCP selection.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = x, y
            self.dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            img = self.image.copy()
            cv2.rectangle(img, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow(self.windowname, img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.end_x, self.end_y = x, y
            w, h = abs(self.end_x - self.start_x), abs(self.end_y - self.start_y)

            if w > 10 and h > 10:
                x0, x1 = sorted([self.start_x, self.end_x])
                y0, y1 = sorted([self.start_y, self.end_y])
                xmin, ymin = self.calculate_original_coords(x0, y0)
                xmax, ymax = self.calculate_original_coords(x1, y1)
                self.current_roi = [xmin, xmax, ymin, ymax]
                self.zoom_level = self.original_image.shape[1] / (xmax - xmin)
            self.update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_points.append(self.calculate_original_coords(x, y))
            if len(self.selected_points) < self.numgcps:
                self.reset_zoom()
            self.update_display()
            if len(self.selected_points) == self.numgcps:
                cv2.destroyAllWindows()

    def manual_selection_window(self, frame, windowname):
        """
        Open an interactive window for manual GCP selection on a given frame.
        """
        self.original_image = frame.copy()
        self.selected_points = []
        self.reset_zoom()
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        H, W = frame.shape[:2]
        cv2.resizeWindow(windowname, W, H)
        cv2.setMouseCallback(windowname, self.click_event)
        self.update_display()
        cv2.waitKey(1)

        while len(self.selected_points) < self.numgcps:
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("z")):
                self.reset_zoom()
                self.update_display()

        cv2.destroyAllWindows()
        return np.asarray(self.selected_points, np.float32)

    def FeatureTracking(self):
        """
        Track selected GCPs through all frames using pyramidal Lucas–Kanade
        optical flow with displacement-ratio based quality control.
        """
        frames = self.vid
        T = len(frames)
        gcps_init = np.asarray(self.selected_points, np.float32)
        prev_pts = gcps_init.reshape(-1, 1, 2)
        prev_gray = frames[0].copy()

        lk_params = dict(winSize=(self.winsiz, self.winsiz), maxLevel=self.maxpyrlevel,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        gcp_tracks = [gcps_init.copy()]
        ratio_th = self.max_displacement
        min_abs = 10.0
        prev_disp = np.full(self.numgcps, min_abs)
        
        disp_prev = []
        for t in range(1, T):
            gray = frames[t]
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            valid = status.reshape(-1).astype(bool)
                        
            if not np.any(valid):
                prev_gray = gray.copy()
                continue

            updated = prev_pts.reshape(-1, 2).copy()
            updated[valid] = next_pts.reshape(-1, 2)[valid]

            disp = np.linalg.norm(
                updated[valid] - prev_pts.reshape(-1, 2)[valid],
                axis=1
            )
            
            disp_prev.append(np.nanmax(disp))
            ratio = abs(np.nanmax(disp) - np.nanmedian(disp_prev)) / np.nanmedian(disp_prev)
            percent = 100.0 * (t + 1) / T
            print(
                f"\r- GCP-tracking in progress: {percent:3.0f}% | "
                f"Frame {t + 1:4.0f}/{T} | "
                f"max_disp={np.nanmax(disp):2.2f} px | "
                f"anomaly={np.nanmax(ratio)*100:3.0f}%              ",
                end="",
            )

            if np.any(ratio > ratio_th) or valid.sum() == 0:
                new_gcps = self.manual_selection_window(gray, self.windowname)
                prev_disp[:] = min_abs
                prev_pts = new_gcps.reshape(-1, 1, 2)
                prev_gray = gray.copy()
                gcp_tracks.append(new_gcps.copy())
                continue

            prev_disp[valid] = np.maximum(disp, 0.1)
            
            updated_out = updated.copy()
            updated_out[~valid] = np.nan
            gcp_tracks.append(updated_out.copy())
            # gcp_tracks.append(updated.copy())
            prev_pts = updated.reshape(-1, 1, 2)
            prev_gray = gray.copy()

        self.gcps_px = gcp_tracks
        print("")

    def run(self, replace=False):
        """
        Execute the full workflow: initial GCP selection, tracking,
        and saving the resulting trajectories.
        """
        path = os.path.join(self.savedir, "__gcps_px_predefined.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path) and not replace:
            return np.load(path)

        path_ini = os.path.join(self.savedir, "__Initial-selection.npy")
        os.makedirs(os.path.dirname(path_ini), exist_ok=True)

        if os.path.exists(path_ini):
            self.selected_points = np.load(path_ini).tolist()
        else:
            self.manual_selection_window(self.vid[0], self.windowname)
            np.save(path_ini, np.asarray(self.selected_points))

        self.FeatureTracking()
        np.save(path, np.asarray(self.gcps_px))
        cv2.destroyAllWindows()

        return np.asarray(self.gcps_px)
        
#%%
class Georeferencing:
    """
    Estimate frame-wise 2D georeferencing transforms from fixed world GCPs
    and time-varying pixel GCP tracks, with optional temporal smoothing.
    """
    def __init__(self, params, pts_px):
        """
        Parameters
        ----------
        pts_world : ndarray (N x 2)
            Fixed world coordinates of GCPs.
        pts_px_list : list of ndarray
            Pixel coordinates of GCPs for each snapshot.
        gsd : float
            Ground sampling distance (world units per pixel).
        smoothing_sigma : float or None
            Standard deviation for temporal Gaussian smoothing of GCP tracks.
        """
        self.W = np.asarray(params["gcps"], np.float32)
        
        if np.any(np.isnan(pts_px)): 
            t = np.arange(pts_px.shape[0])
            for i in range(pts_px.shape[1]):
                for j in range(pts_px.shape[2]):
                    x = pts_px[:, i, j]
                    m = ~np.isnan(x)
                    pts_px[:, i, j] = np.interp(t, t[m], x[m])
        
        
        self.P_list_raw = [np.asarray(p, np.float32) for p in pts_px]
        self.gsd = params["GSD"]
        self.smoothing = params.get("MOTION_SMOOTHING")

        self.num_snapshots = len(self.P_list_raw)
        self.M_list = []
        self.params = params

        if len(self.W) < 2:
            raise ValueError("At least two world control points are required.")

        self.P_list = self._smooth_gcps()
        self._precalculate_all_M()

    @staticmethod
    def _is_rotation_jump(theta_prev, theta_curr, max_deg=90.0):
        if np.isnan(theta_prev) or np.isnan(theta_curr):
            return False
        dtheta = np.abs(
            np.unwrap([theta_prev, theta_curr])[1]
            - np.unwrap([theta_prev, theta_curr])[0]
        )
        return np.rad2deg(dtheta) > max_deg

    def _smooth_gcps(self):
        """
        Apply temporal Savitzky–Golay smoothing to pixel GCP tracks.
        """
        if self.smoothing is None or self.smoothing <= 0:
            return self.P_list_raw
    
        tracks = np.asarray(self.P_list_raw)

        T = tracks.shape[0]
        w = int(round(2 * float(self.smoothing) + 1))
        w = max(3, w)
        if w > T:
            w = T if (T % 2 == 1) else max(3, T - 1)
        if w % 2 == 0:
            w = w - 1 if w > 3 else w + 1
        if w > T:
            return self.P_list_raw
    
        polyorder = 2
        if polyorder >= w:
            polyorder = w - 1
    
        smoothed = savgol_filter(
            tracks,
            window_length=w,
            polyorder=polyorder,
            axis=0,
            mode="interp"
        )
        return [smoothed[t] for t in range(self.num_snapshots)]

    def _precalculate_all_M(self):
        """
        Precompute affine transforms for all snapshots.
        """
        self.M_list = [
            self._estimate_single_M(self.W, P)
            for P in self.P_list
        ]
        
    @staticmethod
    def _compute_scale_from_distances(W, P):
        """
        Estimate scale from pairwise distance ratios.
        """
        s = []
        n = len(W)
        for i in range(n):
            for j in range(i + 1, n):
                dW = np.linalg.norm(W[i] - W[j])
                dP = np.linalg.norm(P[i] - P[j])
                if dP > 0:
                    s.append(dW / dP)
        return np.median(s) if s else 1.0
        
    @classmethod
    def _estimate_single_M(cls, W, P):
        cW = W.mean(axis=0)
        cP = P.mean(axis=0) 

        Wc = W - cW
        Pc = P - cP
        
        Pc_aligned = Pc.copy()
        Pc_aligned[:, 1] *= -1 

        H = Pc_aligned.T @ Wc 
        U, _, Vt = np.linalg.svd(H)

        d = np.diag([1, 1])
        if np.linalg.det(Vt.T @ U.T) < 0:
            d[-1, -1] = -1
        
        R = Vt.T @ d @ U.T

        s = cls._compute_scale_from_distances(W, P)

        rotated_Pc_aligned = (R @ Pc_aligned.T).T 
        dot_sum = np.sum(rotated_Pc_aligned * Wc)

        if dot_sum < 0:
            s *= -1

        R_s = s * R
        T_y_flip = np.diag([1, -1])
        M_final_rotation = R_s @ T_y_flip

        t = cW - M_final_rotation @ cP 

        M = np.zeros((2, 3))
        M[:, :2] = M_final_rotation
        M[:, 2] = t
        return M

    def _calculate_warp_params(self, img_shape, M):
        """
        Compute pixel-space affine transform and output geometry.
        """
        H, W_img = img_shape[:2]

        corners = np.array(
            [[0, 0, 1],
             [W_img, 0, 1],
             [W_img, H, 1],
             [0, H, 1]],
            dtype=float
        ).T

        warped = M @ corners
        xs, ys = warped[0], warped[1]

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        out_W = int(np.ceil((xmax - xmin) / self.gsd))
        out_H = int(np.ceil((ymax - ymin) / self.gsd))
        extent_world = (xmin, xmax, ymin, ymax)

        S = np.array(
            [[1 / self.gsd, 0, -xmin / self.gsd],
             [0, -1 / self.gsd, ymax / self.gsd],
             [0, 0, 1]],
            dtype=np.float32
        )

        M_aug = np.vstack([M, [0, 0, 1]])
        M_pixel = (S @ M_aug)[:2, :]

        return M_pixel, out_W, out_H, extent_world

    @staticmethod
    def _trim_nan_borders(image, extent_world, gsd):
        """
        Remove fully NaN or zero rows and columns and update world extent.
        """
        invalid = np.isnan(image) | np.isclose(image, 0)

        valid_rows = np.where(~np.all(invalid, axis=1))[0]
        valid_cols = np.where(~np.all(invalid, axis=0))[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return image, extent_world

        r0, r1 = valid_rows[0], valid_rows[-1] + 1
        c0, c1 = valid_cols[0], valid_cols[-1] + 1

        trimmed = image[r0:r1, c0:c1]

        xmin, xmax, ymin, ymax = extent_world
        new_xmin = xmin + c0 * gsd
        new_ymax = ymax - r0 * gsd

        new_W, new_H = trimmed.shape[1], trimmed.shape[0]
        new_xmax = new_xmin + new_W * gsd
        new_ymin = new_ymax - new_H * gsd

        new_extent = (new_xmin, new_xmax, new_ymin, new_ymax)
        return trimmed, new_extent


    def warp_batch(self, snapshots, verbose=True):
        """
        Apply frame-wise georeferencing to a batch of images.

        Returns
        -------
        rectified : list of ndarray
            Georeferenced images.
        extents : list of tuple
            World-coordinate extents for each image.
        """
        if len(snapshots) != self.num_snapshots:
            raise ValueError("Number of snapshots does not match GCP sets.")

        rectified = []
        extents = []
        i = 0
        for img, M in zip(snapshots, self.M_list):
            if np.all(M == 0):
                rectified.append(img)
                extents.append(None)
                continue
            
            if verbose:
                i += 1
                percent = 100.0 * (i + 1) / len(snapshots)
                print(
                    f"\r- Georeferencings: {percent:3.0f}% | "
                    f"Frame {i + 1}/{len(snapshots)}                       ",
                    end="",
                )

            M_pixel, out_W, out_H, extent = self._calculate_warp_params(img.shape, M)
            ortho = cv2.warpAffine(img, M_pixel, (out_W, out_H))
            
            del M_pixel

            ortho, extent = self._trim_nan_borders(ortho, extent, self.gsd)
            rectified.append(ortho)
            extents.append(extent)
        
        if verbose:
            print()
                    
        return rectified, extents

#%%
def Visualization(ortho_list,extent_list,params, fps=30):
    gcps = params["gcps"]
    basemap = params["basemap"]
    video_path = os.path.join(params["savedir"], "Visualization.mp4")

    T = len(ortho_list)
    sample = ortho_list[0]
    ext_arr = np.array(extent_list)
    xmin_glob = np.min(ext_arr[:, 0])
    xmax_glob = np.max(ext_arr[:, 1])
    ymin_glob = np.min(ext_arr[:, 2])
    ymax_glob = np.max(ext_arr[:, 3])
    
    print(
        "     Extent:\n"
        f"     xmin = {xmin_glob:.3f}, "
        f"     xmax = {xmax_glob:.3f},\n"
        f"     ymin = {ymin_glob:.3f}, "
        f"     ymax = {ymax_glob:.3f}"
    )
    
    shift = (np.floor(xmin_glob), np.floor(ymin_glob))

    fig = plt.figure(figsize=(6, 6), dpi=200, constrained_layout=True)
    ax = fig.add_subplot(111)
    # fig.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.90)
    
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    H_fig, W_fig = buf.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (W_fig, H_fig), isColor=True)
    
    if basemap is not None:
        extent_glob  = np.array([xmin_glob, xmax_glob, ymin_glob, ymax_glob])
        extent_glob -= np.array([shift[0], shift[0], shift[1], shift[1]])
        ax.imshow(basemap, extent = extent_glob)
        
    plt.plot(gcps[:,0] - shift[0], gcps[:,1] - shift[1], color = "tab:red", linestyle = "None", marker = '.', ms=5)

    ax.set_xlabel(f"Easting (m, +{shift[0]:.0f})", fontsize=14)
    ax.set_ylabel(f"Northing (m, +{shift[1]:.0f})", fontsize=14)
    ax.set_xlim(xmin_glob - shift[0], xmax_glob - shift[0])
    ax.set_ylim(ymin_glob - shift[1], ymax_glob - shift[1])
    ax.tick_params(labelsize = 14)
    im = ax.imshow(sample, origin="lower", extent=extent_list[0], aspect="equal", alpha = 0.75)

    for t in range(T):
        percent = 100.0 * (t + 1) / T
        print(
            f"\r- Visualization: {percent:3.0f} % | "
            f"Frame {t + 1}/{T}    ",
            end="",
        )

        ortho = ortho_list[t].copy()
        ortho = np.flipud(ortho)
        # if ortho.ndim == 2: 
        ortho = cv2.cvtColor(ortho, cv2.COLOR_GRAY2RGB)

        mask_nan = cv2.inRange(ortho, (0,0,0), (0,0,0)) > 0
        alpha = (~mask_nan).astype(np.uint8) * 255

        ortho_rgba = np.zeros((ortho.shape[0], ortho.shape[1], 4), dtype=np.uint8)
        ortho_rgba[..., :3] = ortho
        ortho_rgba[..., 3] = (alpha).astype(np.uint8)
        
        extent_ = np.array(extent_list[t], copy=True) - np.array(
            [shift[0], shift[0], shift[1], shift[1]]
        )

        im.set_data(ortho_rgba)
        im.set_extent(extent_)

        ax.set_title(f"Frame {t}", fontsize=14, loc="left")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        writer.write(frame)

    writer.release()
    plt.close(fig)

#%%
class CoRegistration:
    def __init__(self, vid, extent, params):
        self.vid = vid
        self.extent = extent
        self.T = len(vid)
        self.savedir = params["savedir"]
        self.fps = params["FPS"]
        self.gsd = params["GSD"]

    def write_extent_txt(self, extent):
        outpath = os.path.join(self.savedir, "extent.txt")
        np.savetxt(outpath, np.asarray(extent).reshape(1, 4), fmt="%.6f")

    def run(self, verbose=True):
        xmin = min(e[0] for e in self.extent)
        xmax = max(e[1] for e in self.extent)
        ymin = min(e[2] for e in self.extent)
        ymax = max(e[3] for e in self.extent)
        extent_out = (xmin, xmax, ymin, ymax)

        Nx = int(np.round((xmax - xmin) / self.gsd))
        Ny = int(np.round((ymax - ymin) / self.gsd))

        xg = xmin + self.gsd * (np.arange(Nx) + 0.5)
        yg = ymax - self.gsd * (np.arange(Ny) + 0.5)

        Yg, Xg = np.meshgrid(yg, xg, indexing="ij")
        query = np.stack((Yg.ravel(), Xg.ravel()), axis=-1).astype(np.float32)

        outpath = os.path.join(self.savedir, "Processed.mp4")
        writer = cv2.VideoWriter(
            outpath,
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(60/self.fps),
            (Nx, Ny),
            isColor=False,
        )
        if not writer.isOpened():
            raise IOError("Failed to open VideoWriter")

        frame_f = np.empty((Ny, Nx), dtype=np.float32)

        for t, (img, ex) in enumerate(zip(self.vid, self.extent)):
            H, W = img.shape

            x = ex[0] + self.gsd * (np.arange(W) + 0.5)
            y = ex[3] - self.gsd * (np.arange(H) + 0.5)

            interp = RegularGridInterpolator(
                (y, x),
                img.astype(np.float32),
                bounds_error=False,
                fill_value=np.nan,
            )

            frame_f[:] = interp(query).reshape(Ny, Nx)

            vmin, vmax = np.nanmin(frame_f), np.nanmax(frame_f)

            if vmax > vmin:
                frame_u8 = ((frame_f - vmin) / (vmax - vmin) * 255.0)
            else:
                frame_u8 = np.zeros_like(frame_f)

            writer.write(frame_u8.clip(0, 255).astype(np.uint8))

            if verbose:
                percent = 100.0 * (t + 1) / self.T
                print(
                    f"\r- Co-registering: {percent:3.0f} % | "
                    f"Frame {t + 1}/{self.T} |    ",
                    end="",
                )

        writer.release()
        self.write_extent_txt(extent_out)

        print("\n--- Co-registration report ---")
        print(f"Output video shape : ({self.T}, {Ny}, {Nx}) (frames, Ny, Nx)")
        print(f"gsd                : {self.gsd:.2f} m")
        print(f"fps                : {self.fps:.2f} Hz")
        print(
            f"Extent (xmin, xmax, ymin, ymax) : "
            f"({extent_out[0]:.2f}, {extent_out[1]:.2f}, "
            f"{extent_out[2]:.2f}, {extent_out[3]:.2f})"
        )




