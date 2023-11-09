from flask import Flask, request, jsonify
import cv2
import numpy as np
from win32api import GetSystemMetrics
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
import base64

app = Flask(__name__)

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    # Receive image data (base64 encoded) and number of colors from the request body
    image_base64_1 = request.json['image_base64_1']
    image_base64_2 = request.json['image_base64_2']
    num_colors = 10

    # Extract colors from the images
    colors1 = extract_colors_from_image_base64(image_base64_1, num_colors)
    colors2 = extract_colors_from_image_base64(image_base64_2, num_colors)

    # Calculate color similarity
    emd_distance = calculate_color_similarity_from_base64(image_base64_1, image_base64_2, num_colors)
    similarity_score = emd_distance

    # Calculate similarity percentage
    similarity_percentage = calculate_similarity_percentage(similarity_score, 10.0)

    # Prepare the response data
    response_data = {
        "similarity_score": similarity_score,
        "similarity_percentage": similarity_percentage
    }

    # Return the response
    return jsonify(response_data), 200

# Implement the functions for extracting colors, calculating similarity, and calculating percentage
def extract_colors_from_image_base64(image_base64, num_colors):
    # Decode the base64 image data
    image_data = base64.b64decode(image_base64)
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Reshape the image to be a list of pixels
    pixels = hsv_image.reshape(-1, 3)

    # Use K-Means clustering to group similar colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Calculate the percentage of each color cluster in the image
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Count the number of pixels in each cluster
    counts = np.bincount(labels)

    # Get the percentage of each color
    percentages = (counts / counts.sum()) * 100

    # Sort the colors by percentage
    color_data = [(percent, color) for percent, color in zip(percentages, cluster_centers)]
    color_data.sort(reverse=True)

    return color_data

def calculate_color_similarity_from_base64(image_base641, image_base642, num_colors):
    colors1 = extract_colors_from_image_base64(image_base641, num_colors)
    colors2 = extract_colors_from_image_base64(image_base642, num_colors)

    # Extract color histograms
    hist1 = [color[0] for color in colors1]
    hist2 = [color[0] for color in colors2]

    # Calculate Earth Mover's Distance (EMD) between the histograms
    emd_distance = wasserstein_distance(hist1, hist2)

    # The smaller the EMD distance, the more similar the images are
    return emd_distance

def calculate_similarity_percentage(emd_distance, max_distance):
    # Calculate the similarity percentage
    similarity_percentage = 100 * (1 - (emd_distance / max_distance))

    return similarity_percentage

@app.route('/compare_images', methods=['POST'])
def compare_images():
    # Receive base64 encoded image data for both images from the request body
    image_base64_1 = request.json['image_base64_1']
    image_base64_2 = request.json['image_base64_2']

    base64_bytes = base64.b64decode(image_base64_1)
    image_n = np.frombuffer(base64_bytes, dtype=np.uint8)
    image1 = cv2.imdecode(image_n, cv2.IMREAD_COLOR)

    base64_bytes = base64.b64decode(image_base64_2)
    image_n2 = np.frombuffer(base64_bytes, dtype=np.uint8)
    image2 = cv2.imdecode(image_n2, cv2.IMREAD_COLOR)


    # Perform image comparison using SIFT and homography
    reslut = process_images(image1, image2)

    # Prepare the response data
    response_data = {
        "inlier_percentage": reslut[1],
        "similarity_result": reslut[0]
    }

    # Return the response
    return jsonify(response_data), 200

def decode_base64_image(base64_string):
    base64_bytes = base64.b64decode(base64_string)
    image = np.frombuffer(base64_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def process_images(image1, image2):
    # Resize images to a consistent size
    screen_height = GetSystemMetrics(1)
    max_height = int(0.8 * screen_height)
    scaling_factor = max_height / max(image1.shape[0], image2.shape[0])
    image1 = cv2.resize(image1, (0, 0), fx=scaling_factor, fy=scaling_factor)
    image2 = cv2.resize(image2, (0, 0), fx=scaling_factor, fy=scaling_factor)

    # Apply SIFT feature detection and descriptor calculation
    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)
    keypoints2, descriptors2 = sift.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)

    # Match features using the BFMatcher
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Check for sufficient matches to determine similarity
    if len(good_matches) > 4:
        # Find homography matrix using RANSAC
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Calculate inlier percentage
        inlier_percentage = (np.sum(mask) / len(good_matches)) * 100

        # Determine similarity based on inlier percentage threshold
        similarity_threshold = 30.0
        if inlier_percentage >= similarity_threshold:
            similarity_result = "Images are similar with a {:.2f}% similarity percentage".format(inlier_percentage)
        else:
            similarity_result = "Images are not similar"
    else:
        similarity_result = "Not enough matches to determine similarity"
    return similarity_result, inlier_percentage


# Run the Flask application
if __name__ == '__main__':
    app.run()