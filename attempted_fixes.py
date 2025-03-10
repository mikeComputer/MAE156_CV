import cv2
import numpy as np
import os
import random
import pandas as pd
from tkinter import Tk, simpledialog, filedialog
import sys
from datetime import datetime

# Directories for saving data
base_dir = "clot_picture_save"
save_dir = os.path.join(base_dir, "permanent_images")
csv_dir = os.path.join(base_dir, "csv_data")
measurement_steps_dir = os.path.join(base_dir, "measurement_steps")
clot_objects_dir = os.path.join(base_dir, "clot_objects")
failed_macerations_dir = os.path.join(base_dir, "failed_macerations")
logs_dir = os.path.join(base_dir, "logs")

# Class to redirect output to multiple streams
class Tee:
    def __init__(self, *targets):
        self.targets = targets
    def write(self, obj):
        for target in self.targets:
            target.write(obj)
    def flush(self):
        for target in self.targets:
            target.flush()

def ensure_directories():
    """Create directories if they don't exist."""
    for directory in [base_dir, save_dir, csv_dir, measurement_steps_dir, clot_objects_dir,
                      failed_macerations_dir, logs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

def clean_temp_directories():
    """Clear temp directories at startup, including logs."""
    for directory in [csv_dir, measurement_steps_dir, clot_objects_dir, failed_macerations_dir, logs_dir]:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

def resize_image_for_display(image, max_dim=500):
    """Resize image for display, keep aspect ratio, return scale."""
    height, width = image.shape[:2]
    scale = max_dim / max(height, width) if max(height, width) > max_dim else 1.0
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size), scale

def capture_image():
    """Capture or import image."""
    root = Tk()
    root.withdraw()
    choice = simpledialog.askstring("Input", "Enter 'capture' for live feed or 'import' to choose an image file:")
    root.destroy()

    if choice == 'import':
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to load image: {file_path}")
                return None
            height, width, channels = image.shape
            print(f"Imported Image Resolution: {width} x {height} pixels")
            save_path = os.path.join(save_dir, f"import_{len(os.listdir(save_dir))}.png")
            cv2.imwrite(save_path, image)
            return save_path
        return None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    color_map = {}

    ret, frame = cap.read()
    if ret:
        height, width, channels = frame.shape
        print(f"Captured Frame Resolution: {width} x {height} pixels")
        

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       
        lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
       
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented = np.zeros_like(frame)

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid = (cX, cY)
                if centroid not in color_map:
                    color_map[centroid] = [random.randint(0, 255) for _ in range(3)]
                color = color_map[centroid]
                cv2.drawContours(segmented, [cnt], -1, color, -1)

        display = cv2.addWeighted(frame, 0.7, segmented, 0.3, 0)
        display_resized, _ = resize_image_for_display(display, max_dim=500)
        cv2.imshow("Live Segmentation (Press SPACE to capture, ESC to exit)", display_resized)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            save_path = os.path.join(save_dir, f"capture_{len(os.listdir(save_dir))}.png")
            cv2.imwrite(save_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            return save_path

    cap.release()
    cv2.destroyAllWindows()
    return None

def get_scale_factor(image):
    """Return hardcoded scale factor."""
    return 82.45 / 1058.15207  # very accurate value
    #return 8.9 / 100.91  # very accurate value

def apply_watershed_recursively(roi_image, roi_mask, contour_idx, recursive_area_threshold, depth=0, max_depth=10, prefix=""):
    """
    Recursively apply watershed to break apart large contours using a grayscale version of the ROI.

    Args:
        roi_image: The region of interest (ROI) image in BGR format.
        roi_mask: The binary mask for the ROI.
        contour_idx: Index of the contour being processed.
        recursive_area_threshold: Minimum area threshold for recursive processing.
        depth: Current recursion depth (default: 0).
        max_depth: Maximum recursion depth (default: 10).
        prefix: String prefix for debugging (default: "").

    Returns:
        List of contours after watershed segmentation.
    """
    # Base case: if max depth is reached, return the current contours
    if depth >= max_depth:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # Convert ROI image to grayscale
    roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # Create a 3-channel grayscale image for watershed
    roi_gray_3ch = cv2.merge([roi_gray, roi_gray, roi_gray])

    # Watershed processing
    kernel = np.ones((3, 3), np.uint8)
    dist_transform = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(roi_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Increment to avoid background conflict
    markers[unknown == 255] = 0  # Mark unknown regions

    # Apply watershed using the 3-channel grayscale image
    markers = cv2.watershed(roi_gray_3ch, markers)

    # Extract contours from watershed result
    result_contours = []
    for label in np.unique(markers):
        if label <= 1:  # Skip background (0) and boundaries (1)
            continue
        clot_mask = np.zeros_like(roi_mask, dtype=np.uint8)
        clot_mask[markers == label] = 255
        new_contours, _ = cv2.findContours(clot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for new_cnt in new_contours:
            area = cv2.contourArea(new_cnt)
            if area > recursive_area_threshold:
                # Recursively process large contours
                x, y, w, h = cv2.boundingRect(new_cnt)
                padding = 10
                sub_roi_y1 = max(0, y - padding)
                sub_roi_y2 = min(roi_mask.shape[0], y + h + padding)
                sub_roi_x1 = max(0, x - padding)
                sub_roi_x2 = min(roi_mask.shape[1], x + w + padding)
                sub_roi_mask = clot_mask[sub_roi_y1:sub_roi_y2, sub_roi_x1:sub_roi_x2]
                sub_roi_image = roi_image[sub_roi_y1:sub_roi_y2, sub_roi_x1:sub_roi_x2]
                sub_contours = apply_watershed_recursively(
                    sub_roi_image, sub_roi_mask, contour_idx, recursive_area_threshold, 
                    depth + 1, max_depth, prefix
                )
                for sub_cnt in sub_contours:
                    sub_cnt += [sub_roi_x1, sub_roi_y1]  # Adjust coordinates to ROI space
                    result_contours.append(sub_cnt)
            else:
                result_contours.append(new_cnt)

    return result_contours


def measure_clot_sizes(image_path, base_area_threshold=5, large_area_threshold=1000, recursive_area_threshold=50, red_mask_threshold=200):
    """Measure clot sizes with recursive watershed and red masking for contours above red_mask_threshold."""
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    height, width, channels = image.shape
    total_pixels = height * width
    print(f"Processing Image: {image_path}")
    print(f"Resolution: {width} x {height} pixels")
    print(f"Total Pixels: {total_pixels}")
    print(f"Channels: {channels}")
    print("-" * 40)

    scale_factor = get_scale_factor(image)
    reference_pixels = 480000
    area_threshold = base_area_threshold * (total_pixels / reference_pixels)
    print(f"Base Area Threshold: {area_threshold:.2f} pixels")
    print(f"Large Area Threshold: {large_area_threshold:.2f} pixels")
    print(f"Recursive Area Threshold: {recursive_area_threshold:.2f} pixels")
    print(f"Red Mask Threshold: {red_mask_threshold:.2f} pixels")

    # Crop the image to the center
    h, w = image.shape[:2]
    crop_x1 = int(w * 0.3)
    crop_x2 = int(w * 0.7)
    crop_y1 = int(h * 0.3)
    crop_y2 = int(h * 0.7)
    roi = image[crop_y1:crop_y2, crop_x1:crop_x2]
    cv2.imwrite(os.path.join(measurement_steps_dir, "cropped_image.png"), roi)

    # Convert to HSV for white mask (saturation-based)
    
    
    # white range: [23, 20, 56]
    
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]  # Saturation channel
    #saturation = hsv[23,20,56]
    _, color_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(measurement_steps_dir, "color_mask.png"), color_mask)

    # Clean the mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imwrite(os.path.join(measurement_steps_dir, "cleaned_mask.png"), cleaned_mask)

    # Initial contour detection
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []

    # First Pass: Process contours with selective recursive watershed
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue  # Skip contours below the base threshold
        
        if area < large_area_threshold:
            # Smaller contours: add directly
            filtered_contours.append(cnt)
        else:
            # Larger contours: apply recursive watershed
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 10
            roi_y1 = max(0, y - padding)
            roi_y2 = min(cleaned_mask.shape[0], y + h + padding)
            roi_x1 = max(0, x - padding)
            roi_x2 = min(cleaned_mask.shape[1], x + w + padding)
            
            roi_mask = cleaned_mask[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_image = roi[roi_y1:roi_y2, roi_x1:roi_x2]
            
            new_contours = apply_watershed_recursively(roi_image, roi_mask, i, recursive_area_threshold, prefix="")
            for new_cnt in new_contours:
                if cv2.contourArea(new_cnt) >= area_threshold:
                    new_cnt += [roi_x1, roi_y1]  # Adjust coordinates back to full ROI space
                    filtered_contours.append(new_cnt)

    # Second Pass: Reapply recursive watershed to large contours
    refined_contours = []
    for j, cnt in enumerate(filtered_contours):
        area = cv2.contourArea(cnt)
        if area <= recursive_area_threshold:
            # Contours small enough: keep as is
            refined_contours.append(cnt)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 10
            sub_roi_y1 = max(0, y - padding)
            sub_roi_y2 = min(roi.shape[0], y + h + padding)
            sub_roi_x1 = max(0, x - padding)
            sub_roi_x2 = min(roi.shape[1], x + w + padding)
            contour_mask = np.zeros_like(cleaned_mask)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            sub_roi_mask = contour_mask[sub_roi_y1:sub_roi_y2, sub_roi_x1:sub_roi_x2]
            sub_roi_image = roi[sub_roi_y1:sub_roi_y2, sub_roi_x1:sub_roi_x2]
            new_contours = apply_watershed_recursively(sub_roi_image, sub_roi_mask, j, recursive_area_threshold, prefix="second_pass_")
            for new_cnt in new_contours:
                if cv2.contourArea(new_cnt) >= area_threshold:
                    new_cnt += [sub_roi_x1, sub_roi_y1]  # Adjust coordinates back to ROI space
                    refined_contours.append(new_cnt)

    filtered_contours = refined_contours

    # Save pre-red masking contour overlay image
    pre_red_masking_image = roi.copy()
    cv2.drawContours(pre_red_masking_image, filtered_contours, -1, (0, 255, 0), 2)  # Green contours
    cv2.imwrite(os.path.join(measurement_steps_dir, "pre_red_masking_contour_overlay.png"), pre_red_masking_image)

    # Step 3: Apply red masking at the end, only once, to contours with area >= red_mask_threshold
    red_contours = []
    red_masking_image = roi.copy()  # Create a copy for visualizing red masking results
    for i, cnt in enumerate(filtered_contours):
        area = cv2.contourArea(cnt)
        if area >= red_mask_threshold:
            print(f"Processing contour {i} with area {area:.2f} pixels for red masking")
            # Define ROI for red masking (coordinates are relative to the cropped roi)
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 10
            roi_y1 = max(0, y - padding)
            roi_y2 = min(roi.shape[0], y + h + padding)
            roi_x1 = max(0, x - padding)
            roi_x2 = min(roi.shape[1], x + w + padding)

            # Map the ROI coordinates back to the original image
            orig_y1 = crop_y1 + roi_y1
            orig_y2 = crop_y1 + roi_y2
            orig_x1 = crop_x1 + roi_x1
            orig_x2 = crop_x1 + roi_x2

            # Extract the ROI directly from the original image
            red_masking_roi = image[orig_y1:orig_y2, orig_x1:orig_x2]
            # Save the ROI in BGR for debugging to confirm colors
            cv2.imwrite(os.path.join(measurement_steps_dir, f"red_masking_roi_bgr_{i}.png"), red_masking_roi)

            # Convert this ROI to HSV for red masking
            roi_hsv = cv2.cvtColor(red_masking_roi, cv2.COLOR_BGR2HSV)

            # Apply red mask to segment red objects
            lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
            lower_red4 = np.array([160, 12, 119])
            upper_red4 = np.array([179, 164, 174])
            red_mask = (cv2.inRange(roi_hsv, lower_red1, upper_red1) + 
                        cv2.inRange(roi_hsv, lower_red2, upper_red2) + 
                        cv2.inRange(roi_hsv, lower_red4, upper_red4))

            # Save raw red mask for debugging
            cv2.imwrite(os.path.join(measurement_steps_dir, f"raw_red_mask_contour_{i}.png"), red_mask)
            print(f"Raw red mask pixels detected for contour {i}: {np.sum(red_mask > 0)}")

            # Refine the mask
            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
            red_mask = cv2.erode(red_mask, kernel, iterations=2)
            cleaned_red_mask = cv2.dilate(red_mask, kernel, iterations=2)

            # Save cleaned red mask for debugging
            cv2.imwrite(os.path.join(measurement_steps_dir, f"red_mask_contour_{i}.png"), cleaned_red_mask)
            print(f"Cleaned red mask pixels detected for contour {i}: {np.sum(cleaned_red_mask > 0)}")

            # Detect contours from red-masked ROI
            red_contours_in_roi, _ = cv2.findContours(cleaned_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_red_contours = []
            for new_cnt in red_contours_in_roi:
                if cv2.contourArea(new_cnt) >= area_threshold:
                    # Adjust coordinates back to the cropped ROI space
                    new_cnt += [roi_x1, roi_y1]
                    valid_red_contours.append(new_cnt)
                    # Draw the red-masked contour on the visualization image
                    cv2.drawContours(red_masking_image, [new_cnt], -1, (0, 0, 255), 2)  # Red contours

            if valid_red_contours:  # If red contours are detected, replace the original contour
                # Remove the original contour from filtered_contours
                filtered_contours[i] = None  # Mark for removal (will filter out later)
                red_contours.extend(valid_red_contours)
            else:
                print(f"No valid red contours detected for contour {i}, keeping original contour.")

    # Filter out None values from filtered_contours after processing
    filtered_contours = [cnt for cnt in filtered_contours if cnt is not None]

    # Save the red masking contour overlay image
    cv2.imwrite(os.path.join(measurement_steps_dir, "red_masking_contour_overlay.png"), red_masking_image)

    # Combine red contours with filtered contours
    filtered_contours.extend(red_contours)

    # Visualize segmented clots
    segmented = np.zeros_like(roi)
    for cnt in filtered_contours:
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.drawContours(segmented, [cnt], -1, color, -1)
    cv2.imwrite(os.path.join(measurement_steps_dir, "segmented_image.png"), segmented)

    # Overlay contours on ROI
    contour_overlay_image = roi.copy()
    cv2.drawContours(contour_overlay_image, filtered_contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(measurement_steps_dir, "contour_overlay_image.png"), contour_overlay_image)

    # Measure clots
    clot_data = []
    measured_image = roi.copy()
    for i, cnt in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        pixel_length, pixel_width = w, h
        real_length, real_width = w * scale_factor, h * scale_factor
        cv2.rectangle(measured_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        centroid = (x + w // 2, y + h // 2)
        cv2.circle(measured_image, centroid, 5, (0, 255, 0), -1)
        clot_data.append({
            "Clot_ID": f"Clot_{i}",
            "Pixel length": pixel_length,
            "Pixel width": pixel_width,
            "Real life length (mm)": real_length,
            "Real life width (mm)": real_width
        })
    print(f"Clots measured: {len(clot_data)}")
    cv2.imwrite(os.path.join(measurement_steps_dir, "measured_image.png"), measured_image)

    # Save to CSV
    df = pd.DataFrame(clot_data, columns=["Clot_ID", "Pixel length", "Pixel width",
                                          "Real life length (mm)", "Real life width (mm)"])
    raw_csv_path = os.path.join(csv_dir, "clot_measurements.csv")
    df.to_csv(raw_csv_path, index=False)

    df_cleaned = df.round(5).drop_duplicates(subset=["Real life length (mm)", "Real life width (mm)"])
    cleaned_csv_path = os.path.join(csv_dir, "clot_measurements_cleaned.csv")
    df_cleaned.to_csv(cleaned_csv_path, index=False)

    if not df_cleaned.empty:
        lengths = df_cleaned["Real life length (mm)"].values
        widths = df_cleaned["Real life width (mm)"].values
        max_sizes = np.max([lengths, widths], axis=0)

        mean_length = np.mean(lengths)
        median_length = np.median(lengths)
        std_length = np.std(lengths)

        mean_width = np.mean(widths)
        median_width = np.median(widths)
        std_width = np.std(widths)

        mean_size = np.mean(max_sizes)
        median_size = np.median(max_sizes)
        std_size = np.std(max_sizes)
        min_size = np.min(max_sizes)
        max_size = np.max(max_sizes)

        print(f"Unique clots: {len(df_cleaned)}")
        print(f"Length Stats - Mean: {mean_length:.2f} mm, Median: {median_length:.2f} mm, Std: {std_length:.2f} mm")
        print(f"Width Stats - Mean: {mean_width:.2f} mm, Median: {median_width:.2f} mm, Std: {std_width:.2f} mm")
        print(f"Size Stats - Mean: {mean_size:.2f} mm, Median: {median_size:.2f} mm, Std: {std_size:.2f} mm")
        print(f"Min size: {min_size:.2f} mm")
        print(f"Max size: {max_size:.2f} mm")
    else:
        print("No clots detected.")

    large_clots = df_cleaned[(df_cleaned["Real life length (mm)"] >= 2) | (df_cleaned["Real life width (mm)"] >= 2)]
    small_clots = df_cleaned[(df_cleaned["Real life length (mm)"] < 2) & (df_cleaned["Real life width (mm)"] < 2)]

    num_large_clots = len(large_clots)
    total_clots = len(df_cleaned)
    num_small_clots = len(small_clots)
    percentage_small = (num_small_clots / total_clots) * 100 if total_clots > 0 else 0

    print(f"Large clots (>= 2mm): {num_large_clots}")
    print(f"Small clots (< 2mm): {num_small_clots} ( {percentage_small:.2f}%)")

    large_clots_csv_path = os.path.join(csv_dir, "large_clots.csv")
    large_clots.to_csv(large_clots_csv_path, index=False)

    for _, row in large_clots.iterrows():
        clot_id = row["Clot_ID"]
        idx = int(clot_id.split('_')[1])
        x, y, w, h = cv2.boundingRect(filtered_contours[idx])
        clot_mask = np.zeros_like(roi)
        cv2.drawContours(clot_mask, [filtered_contours[idx]], -1, (255, 255, 255), -1)
        clot_image = cv2.bitwise_and(roi, clot_mask)
        cv2.imwrite(os.path.join(failed_macerations_dir, f"{clot_id}.png"), clot_image)

    df_sorted = df_cleaned.sort_values(by=["Real life length (mm)", "Real life width (mm)"], ascending=False).head(10)
    for _, row in df_sorted.iterrows():
        clot_id = row["Clot_ID"]
        idx = int(clot_id.split('_')[1])
        x, y, w, h = cv2.boundingRect(filtered_contours[idx])
        clot_mask = np.zeros_like(roi)
        cv2.drawContours(clot_mask, [filtered_contours[idx]], -1, (255, 255, 255), -1)
        clot_image = cv2.bitwise_and(roi, clot_mask)
        cv2.imwrite(os.path.join(clot_objects_dir, f"{clot_id}.png"), clot_image)

    print(f"Raw data: {raw_csv_path}")
    print(f"Cleaned data: {cleaned_csv_path}")
    print(f"Large clots data: {large_clots_csv_path}")
    print(f"Step images: {measurement_steps_dir}")
    print(f"Clot images: {clot_objects_dir}")
    print(f"Failed macerations: {failed_macerations_dir}")

    display_final_image(resize_image_for_display(measured_image)[0])
    display_steps(measurement_steps_dir)


def display_final_image(measured_image):
    """Show final image, close on 'q' or window close."""
    cv2.imshow("Processed Image", measured_image)
    while True:
        if cv2.getWindowProperty("Processed Image", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def display_steps(measurement_steps_dir):
    """Show step images in grid, close on 'q' or window close."""
    steps = [
        ("cropped_image.png", "Cropped Image"),
        ("color_mask.png", "Color Mask"),
        ("cleaned_mask.png", "Cleaned Mask"),
        ("pre_red_masking_contour_overlay.png", "Pre Red Masking Overlay"),
        ("red_masking_contour_overlay.png", "Red Masking Overlay"),
        ("segmented_image.png", "Segmented Image"),
        ("contour_overlay_image.png", "Contour Overlay"),
        ("measured_image.png", "Measured Image")
    ]
    images = [(cv2.imread(os.path.join(measurement_steps_dir, step[0])), step[1]) for step in steps]
    images = [(img, label) for img, label in images if img is not None]

    if images:
        resized_images = [resize_image_for_display(img, max_dim=500) for img, _ in images]
        images_with_labels = []
        max_height = max(img.shape[0] for img, _ in resized_images)

        for (img, scale), (_, label) in zip(resized_images, images):
            label_height = 50
            labeled_img = np.zeros((max_height + label_height, img.shape[1], 3), dtype=np.uint8)
            labeled_img[label_height:, :img.shape[1]] = img
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = label_height // 2 + text_size[1] // 2
            cv2.putText(labeled_img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
            images_with_labels.append(labeled_img)

        num_images = len(images_with_labels)
        num_rows = (num_images + 2) // 3
        rows = []
        for i in range(num_rows):
            start_idx = i * 3
            end_idx = min((i + 1) * 3, num_images)
            row_images = images_with_labels[start_idx:end_idx]
            if row_images:
                while len(row_images) < 3:
                    blank_img = np.zeros_like(row_images[0])
                    row_images.append(blank_img)
                rows.append(np.hstack(row_images))

        if rows:
            combined = np.vstack(rows)
            cv2.imshow("Processing Steps", combined)
            while True:
                if cv2.getWindowProperty("Processing Steps", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        else:
            print("No step images to display.")
    else:
        print("No step images to display.")

# Main execution
if __name__ == "__main__":
    ensure_directories()
    
    clean_temp_directories()
    
    log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_output_log.txt"
    log_path = os.path.join(logs_dir, log_filename)
    log_file = open(log_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    
    image_path = capture_image()
    
    
    # if image_path:
    #     measure_clot_sizes(
    #         image_path,
    #         base_area_threshold=2,
    #         large_area_threshold=500,
    #         recursive_area_threshold=500,
    #         red_mask_threshold=500  # Adjust this value for iteration
    #     )

   
    if image_path:
        measure_clot_sizes(
            image_path,
            base_area_threshold=2,
            large_area_threshold=500,
            recursive_area_threshold=10,
            red_mask_threshold=50 
        )
    
    sys.stdout = original_stdout
    log_file.close()