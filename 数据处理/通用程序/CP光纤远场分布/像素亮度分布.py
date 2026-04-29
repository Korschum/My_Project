import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from scipy.optimize import curve_fit
import matplotlib.patches as patches
import os
import json

# --- Global Variables ---
img_fiber_path = '/Users/vassago/Desktop/资料/BY/02 动态范围/实验数据分析/20260419/DCIM/Pictures/1.jpg'
img_white_path = '/Users/vassago/Desktop/资料/BY/02 动态范围/实验数据分析/20260419/DCIM/Pictures/1l.jpg'
output_dir = '/Users/vassago/Desktop/资料/BY/02 动态范围/实验数据分析/20260419/picture_data'

# Image Data
img_fiber = None
img_white = None
img_fiber_gray = None

# Calibration Data
pixel_ratio = 1.0  # pixels / unit length

# Polygon Data
exclusion_polygons = [] 
current_polygon = []    

# Fitting Parameters Initial Guess
# [omega_0, z, alpha, beta, A, gamma, offset]
initial_params = [10.0, 0.0, 0.0, 0.0, 1000.0, 1.0, 10.0] 
param_bounds = ([0.1, -100, -1.57, -1.57, 1, 0.1, 0], [100, 100, 1.57, 1.57, 1e8, 5.0, 100])

# --- Image Loading & Preprocessing ---
def load_images():
    global img_fiber, img_white, img_fiber_gray
    try:
        img_fiber = cv2.imread(img_fiber_path)
        img_white = cv2.imread(img_white_path)
        if img_fiber is None or img_white is None:
            print("Error: Image not found. Please check the path.")
            return False
        
        img_fiber_gray = cv2.cvtColor(img_fiber, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return True
    except Exception as e:
        print(f"Error loading images: {e}")
        return False

# --- Geometric Model Calculation ---
def get_physical_coords(x, y, alpha, beta, z_center):
    xp = x * np.cos(beta) + y * np.sin(beta)
    yp = -x * np.sin(beta) + y * np.cos(beta)
    z = z_center - xp * np.sin(alpha)
    r = np.sqrt((xp * np.cos(alpha))**2 + yp**2)
    return r, z

def gaussian_beam_model(coords, omega_0, z_center, alpha, beta, A, gamma, offset):
    x, y = coords
    r, z = get_physical_coords(x, y, alpha, beta, z_center)
    
    lambda_val = 1.0 
    z_R = (np.pi * omega_0**2) / lambda_val
    w_z = omega_0 * np.sqrt(1 + (z / z_R)**2)
    w_z = np.clip(w_z, 1e-5, None)
    
    intensity = (1.0 / (w_z**2)) * np.exp(-2 * (r**2) / (w_z**2))
    
    # Core Correction: S = A * I^gamma + offset
    predicted_signal = A * (intensity ** gamma) + offset
    return predicted_signal

# --- Mask Generation ---
def create_mask(shape, polygons):
    mask = np.ones(shape, dtype=bool)
    h, w = shape
    from matplotlib.path import Path
    
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        path = Path(pts.reshape(-1, 2))
        points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        inside = path.contains_points(points).reshape(h, w)
        mask[inside] = False
    return mask

# --- Save Data Function ---
def save_results(popt, pcov, actual_vals, predicted_vals):
    # 1. Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # 2. Prepare data dictionary
    param_names = ['omega_0', 'z_center', 'alpha', 'beta', 'A', 'gamma', 'offset']
    results_dict = {
        "pixel_ratio": pixel_ratio,
        "parameters": {},
        "units": {
            "omega_0": "length_units",
            "z_center": "length_units",
            "alpha": "radians",
            "beta": "radians"
        }
    }
    
    # Extract parameters and errors
    for i, name in enumerate(param_names):
        results_dict["parameters"][name] = float(popt[i])
        # Calculate standard deviation (sqrt of variance diagonal)
        if pcov is not None:
            std_dev = np.sqrt(pcov[i, i])
            results_dict["parameters"][f"{name}_std"] = float(std_dev)

    # 3. Save JSON
    json_path = os.path.join(output_dir, 'fitting_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Parameters saved to: {json_path}")
    
    # 4. Save diagnostic image (Residual + Scatter plot)
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Residual
    plt.subplot(1, 2, 1)
    plt.scatter(predicted_vals, actual_vals - predicted_vals, alpha=0.5, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Intensity')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residual Distribution')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: Prediction vs Actual
    plt.subplot(1, 2, 2)
    plt.scatter(actual_vals, predicted_vals, alpha=0.5, color='blue')
    min_val = min(actual_vals.min(), predicted_vals.min())
    max_val = max(actual_vals.max(), predicted_vals.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')
    plt.xlabel('Measured Pixel Intensity')
    plt.ylabel('Model Predicted Intensity')
    plt.title('Goodness of Fit: Predicted vs Measured')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    img_path = os.path.join(output_dir, 'fitting_diagnostic.png')
    plt.savefig(img_path, dpi=300)
    print(f"Diagnostic image saved to: {img_path}")
    plt.close()

# --- Fitting Main Program ---
def run_fitting():
    global pixel_ratio, exclusion_polygons
    
    if pixel_ratio == 1.0:
        print("Warning: Calibration not performed. Using default pixel ratio.")
    
    h, w = img_fiber_gray.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    # Coordinate conversion
    x_phys = (x_grid - w/2) / pixel_ratio
    y_phys = (h/2 - y_grid) / pixel_ratio 
    
    # Apply mask
    mask = create_mask((h, w), exclusion_polygons)
    
    # Get valid data
    x_valid = x_grid[mask].ravel()
    y_valid = y_grid[mask].ravel()
    signal_valid = img_fiber_gray[mask].ravel()
    
    coords = (x_valid, y_valid)
    
    print(f"Starting fitting... Number of valid pixels: {len(x_valid)}")
    
    try:
        popt, pcov = curve_fit(
            gaussian_beam_model, 
            coords, 
            signal_valid, 
            p0=initial_params, 
            bounds=param_bounds,
            method='trf',
            max_nfev=5000
        )
        
        omega_0, z_center, alpha, beta, A, gamma, offset = popt
        
        print("-" * 30)
        print("Fitting Results (Geometric Parameters):")
        print(f"Beam Waist (omega_0): {omega_0:.4f}")
        print(f"Center Position (z_0): {z_center:.4f}")
        print(f"Tilt Angle (alpha):  {alpha:.4f} rad")
        print(f"Rotation Angle (beta): {beta:.4f} rad")
        print("-" * 30)
        print("System Response Parameters:")
        print(f"Response Exponent (Gamma): {gamma:.4f}")
        print(f"Background Offset:         {offset:.2f}")
        print("-" * 30)
        
        # 1. Visualize residual map
        visualize_fit(x_grid, y_grid, popt, mask)
        
        # 2. Prepare data for plotting and saving
        predicted_full = gaussian_beam_model((x_grid, y_grid), *popt)
        actual_vals = img_fiber_gray[mask].ravel()
        predicted_vals = predicted_full[mask].ravel()
        
        # 3. Plot and save
        plot_correlation(actual_vals, predicted_vals)
        save_results(popt, pcov, actual_vals, predicted_vals)
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        import traceback
        traceback.print_exc()

def visualize_fit(x_grid, y_grid, popt, mask):
    fitted_signal = gaussian_beam_model((x_grid, y_grid), *popt)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_fiber_gray, cmap='gray')
    plt.title("Original Image (with Masked Areas)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(fitted_signal, cmap='inferno')
    plt.title("Model Predicted Intensity Distribution")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    residual = np.zeros_like(img_fiber_gray)
    valid_residual = np.abs(img_fiber_gray[mask].ravel() - fitted_signal[mask].ravel())
    residual[mask] = valid_residual
    
    plt.imshow(residual, cmap='coolwarm')
    plt.title("Residual Analysis (Absolute Error)")
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_correlation(actual_data, predicted_data):
    plt.figure(figsize=(8, 8))
    
    # Random sampling
    num_points = len(actual_data)
    if num_points > 5000:
        indices = np.random.choice(num_points, 5000, replace=False)
        x_plot = actual_data[indices]
        y_plot = predicted_data[indices]
        alpha_val = 0.5
        s_val = 10
    else:
        x_plot = actual_data
        y_plot = predicted_data
        alpha_val = 0.6
        s_val = 20

    plt.scatter(x_plot, y_plot, alpha=alpha_val, s=s_val, color='blue', edgecolors='none')
    
    min_val = min(x_plot.min(), y_plot.min())
    max_val = max(x_plot.max(), y_plot.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)', linewidth=2)
    
    plt.xlabel('Measured Pixel Intensity')
    plt.ylabel('Model Predicted Intensity')
    plt.title("Goodness of Fit: Predicted vs Measured")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- Interactive Interface ---
class InteractiveApp:
    def __init__(self):
        self.fig, (self.ax_fiber, self.ax_white) = plt.subplots(1, 2, figsize=(15, 6))
        self.ax_fiber.set_title("Fiber Image: Draw polygons to exclude areas\n(Press Enter to finish drawing)")
        self.ax_white.set_title("White Image: Draw a line to calibrate length")
        
        self.img_fiber_plot = self.ax_fiber.imshow(img_fiber)
        self.img_white_plot = self.ax_white.imshow(img_white)
        
        self.poly_patch = None
        plt.subplots_adjust(bottom=0.2)
        
        ax_len = plt.axes([0.2, 0.05, 0.15, 0.05])
        self.txt_len = TextBox(ax_len, 'Real Length:', initial='10.0')
        
        ax_fit = plt.axes([0.4, 0.05, 0.1, 0.05])
        self.btn_fit = Button(ax_fit, 'Run Fitting')
        self.btn_fit.on_clicked(self.on_fit_clicked)
        
        ax_clear = plt.axes([0.55, 0.05, 0.1, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear Polygons')
        self.btn_clear.on_clicked(self.on_clear_clicked)
        
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.mode = 'white'
        self.temp_line_points = []

    def on_click(self, event):
        if event.inaxes == self.ax_white:
            self.mode = 'white'
            x, y = int(event.xdata), int(event.ydata)
            self.temp_line_points.append([x, y])
            if len(self.temp_line_points) == 2:
                for line in self.ax_white.lines:
                    line.remove()
                pts = np.array(self.temp_line_points)
                self.ax_white.plot(pts[:,0], pts[:,1], 'r-', linewidth=2)
                self.fig.canvas.draw_idle()
                try:
                    user_val = float(self.txt_len.text)
                    dist = np.sqrt(np.sum((pts[0]-pts[1])**2)) # Fixed calculation error
                    global pixel_ratio
                    pixel_ratio = dist / user_val
                    print(f"Calibration updated: {dist:.2f} px = {user_val} unit -> Ratio: {pixel_ratio:.2f}")
                except:
                    pass
                self.temp_line_points = []
        elif event.inaxes == self.ax_fiber:
            self.mode = 'fiber'
            x, y = int(event.xdata), int(event.ydata)
            current_polygon.append([x, y])
            if self.poly_patch:
                self.poly_patch.remove()
            pts = np.array(current_polygon)
            self.poly_patch = patches.Polygon(pts, closed=True, fill=False, edgecolor='yellow', linewidth=2)
            self.ax_fiber.add_patch(self.poly_patch)
            self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'enter' and self.mode == 'fiber' and len(current_polygon) > 2:
            exclusion_polygons.append(list(current_polygon))
            current_polygon.clear()
            if self.poly_patch:
                self.poly_patch.remove()
                self.poly_patch = None
            self.ax_fiber.imshow(img_fiber)
            for poly in exclusion_polygons:
                pts = np.array(poly)
                p = patches.Polygon(pts, closed=True, fill=False, edgecolor='red', linewidth=2)
                self.ax_fiber.add_patch(p)
            self.fig.canvas.draw_idle()
            print(f"Exclusion area saved. Total: {len(exclusion_polygons)}")

    def on_fit_clicked(self, event):
        run_fitting()

    def on_clear_clicked(self, event):
        exclusion_polygons.clear()
        self.ax_fiber.imshow(img_fiber)
        self.fig.canvas.draw_idle()
        print("All exclusion areas cleared.")

# --- Main Execution ---
if __name__ == "__main__":
    if load_images():
        app = InteractiveApp()
        plt.show()