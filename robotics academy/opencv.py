import GUI
import HAL
import cv2
import numpy as np

def gray():
    print("running grayscale...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    GUI.showImage(gray)

def morph(operation):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((7, 7), np.uint8)
    
    if operation == "eroded":                
        print("running erosion...")
        processed_mask = cv2.erode(red_mask, kernel, iterations=1)
    elif operation == "dilated":
        print("running dilation...")
        processed_mask = cv2.dilate(red_mask, kernel, iterations=2)
    elif operation == "opened":
        print("running opening...")
        processed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    elif operation == "closed":
        print("running closing...")
        processed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("Invalid operation. Use 'eroded', 'dilated', 'opened', or 'closed'.")

    # Create output: White for red regions, black otherwise
    output = cv2.bitwise_and(image, image, mask=processed_mask)
    output[processed_mask == 255] = [255, 255, 255]  # Force white color

    # (Optional) Draw bounding boxes
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box

    GUI.showImage(output)

def color_filter():
    print("running color filter...")

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color (you might need to adjust these)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges and combine them
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw on
    result = image.copy()
    
    # Draw bounding boxes
    for cnt in contours:
        # Ignore small areas that could be noise
        area = cv2.contourArea(cnt)
        if area > 500:  # Adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Show the result
    GUI.showImage(result)

def edge_filter():
    print("running edge filter...")
    image = GUI.getImage()
    if image is None:
        return

    # edge detection typically works on grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Canny Edge Detection ---
    canny_edges = cv2.Canny(gray, 15, 150)  # thresholds 

    # Apply Gaussian blur first to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplace_edges = cv2.convertScaleAbs(laplacian)


    GUI.showImage(canny_edges)


def convolutions(operation):
    # Get the image from the camera
    image = GUI.getImage()
    if image is None:
        return

    # Convert to grayscale (optional - some filters work better in grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define common convolution kernels
    kernels = {
        'blur': np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]], dtype=np.float32) / 9,  # Simple averaging
        
        'gaussian_blur': np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]], dtype=np.float32) / 16,  # Gaussian approximation
        
        'sharpen': np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]], dtype=np.float32),       # Sharpening
        
        'edge_enhance': np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]], dtype=np.float32),     # Strong edge enhancement
        
        'laplacian': np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]], dtype=np.float32)       # Edge detection
    }

    # Apply each convolution and display results
    results = {}
    for name, kernel in kernels.items():
        # Apply 2D filter (use -1 to maintain original depth)
        filtered = cv2.filter2D(image, -1, kernel)
        
        # For edge detection kernels, convert to absolute values
        if 'edge' in name or 'laplacian' in name:
            filtered = cv2.convertScaleAbs(filtered)
        
        results[name] = filtered

    if operation =="blur":
        print("running convulitions (blur)...")
        GUI.showImage(results['blur'])
    if operation =="sharpen":
        print("running convulitions (sharpen)...")
        GUI.showImage(results['sharpen'])
    if operation =="edge_enhance":
        print("running convulitions (edge enhance)...")
        GUI.showImage(results['edge_enhance'])




# Global variables to store previous frame
prev_gray = None
color_wheel = None

def init_color_wheel():
    """Initialize color wheel for optical flow visualization"""
    global color_wheel
    color_wheel = np.zeros((255, 255, 3), dtype=np.uint8)
    cv2.circle(color_wheel, (127, 127), 127, (255, 255, 255), -1)
    for i in range(360):
        angle = np.radians(i)
        x = int(127 + 120 * np.cos(angle))
        y = int(127 + 120 * np.sin(angle))
        bgr = cv2.cvtColor(np.uint8([[[i, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        cv2.line(color_wheel, (127, 127), (x, y), (int(bgr[0]), int(bgr[1]), int(bgr[2])), 2)

def optical_flow():
    print("running optical flow...")

    global prev_gray
    
    frame = GUI.getImage()
    if frame is None:
        return
    
    # Initialize color wheel on first run
    if color_wheel is None:
        init_color_wheel()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow if we have a previous frame
    if prev_gray is not None:
        # Parameters for Farneback optical flow:
        # pyr_scale = 0.5: image scale (<1) to build pyramids
        # levels = 3: number of pyramid layers
        # winsize = 15: averaging window size
        # iterations = 3: number of iterations at each pyramid level
        # poly_n = 5: size of pixel neighborhood
        # poly_sigma = 1.1: standard deviation of Gaussian
        # flags = 0: operation flags
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )
        
        # Convert flow to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Normalize magnitude for visualization
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Create HSV image for visualization
        hsv = np.zeros_like(frame)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue (direction)
        hsv[..., 1] = 255                       # Saturation
        hsv[..., 2] = magnitude                 # Value (magnitude)
        
        # Convert to BGR for display
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Overlay flow on original image
        overlay = cv2.addWeighted(frame, 0.7, flow_bgr, 0.3, 0)
        
        # Show result
        GUI.showImage(overlay)
    
    # Store current frame for next iteration
    prev_gray = gray.copy()

def corner_detector():
    print("running corner detector...")
    image = GUI.getImage()
    if image is None:
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Harris corner detection parameters
    block_size = 2      # Neighborhood size
    ksize = 3           # Aperture parameter for Sobel operator
    k = 0.04            # Harris detector free parameter
    threshold = 0.01    # Threshold for corner detection
    
    # Detect corners
    corner_response = cv2.cornerHarris(gray, block_size, ksize, k)
    
    # Create output image
    output = image.copy()
    
    # Threshold and mark only strong corners
    output[corner_response > threshold * corner_response.max()] = [0, 255, 0]  # Mark only corner pixels green
    
    # Show the result
    GUI.showImage(output)


def hough_circles():
    print("running hough circles...")
    image = GUI.getImage()
    if image is None:
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(blurred, 
                              cv2.HOUGH_GRADIENT,
                              dp=1,            # Inverse ratio of accumulator resolution
                              minDist=20,      # Minimum distance between centers
                              param1=100,       # Upper threshold for edge detection
                              param2=50,       # Threshold for center detection
                              minRadius=0,     # Minimum radius
                              maxRadius=0)     # Maximum radius (0 means unlimited)
    
    # Create output image
    output = image.copy()
    
    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(output, center, radius, (0, 255, 0), 2)  # Green circle
            cv2.circle(output, center, 2, (0, 0, 255), 3)       # Red center
    
    # Show the result
    GUI.showImage(output)

while True:
    # Enter iterative code!
    image = GUI.getImage()
    if image is None:
        continue

    # gray()
    
    #morphology
    # morph("eroded")
    # morph("dilated")
    # morph("opened")
    # morph("closed")

    # color_filter()
    # edge_filter()

    #convolutions
    # convolutions("blur")
    # convolutions("sharpen")
    # convolutions("edge_enhance")
   
    # optical_flow()

    # corner_detector()

    hough_circles()

