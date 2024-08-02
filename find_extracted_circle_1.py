import cv2
import numpy as np

def detect_blobs(image, params):
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    return keypoints

def auto_adjust_params(image, target_blob_count=12):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.filterByInertia = True

    min_area_range = [200, 300, 400, 500, 600, 700]
    max_area_range = [1000, 2000, 3000, 4000, 5000]
    min_circularity_range = [0.1, 0.3, 0.5, 0.7, 0.9]
    min_convexity_range = [0.6, 0.7, 0.8, 0.9]
    min_inertia_range = [0.4, 0.5, 0.6, 0.7]

    for min_area in min_area_range:
        for max_area in max_area_range:
            for min_circularity in min_circularity_range:
                for min_convexity in min_convexity_range:
                    for min_inertia in min_inertia_range:
                        params.minArea = min_area
                        params.maxArea = max_area
                        params.minCircularity = min_circularity
                        params.minConvexity = min_convexity
                        params.minInertiaRatio = min_inertia

                        keypoints = detect_blobs(image, params)
                        if len(keypoints) == target_blob_count:
                            print(f"Optimal parameters found: minArea={min_area}, maxArea={max_area}, "
                                  f"minCircularity={min_circularity}, minConvexity={min_convexity}, minInertiaRatio={min_inertia}")
                            return keypoints, params

    print("Optimal parameters not found.")
    return [], params

def find_blob_boundary_points(blob_center, blob_radius, image_shape):
    boundary_points = []
    x_center, y_center = int(blob_center[0]), int(blob_center[1])
    for angle in range(0, 360):
        theta = np.deg2rad(angle)
        x = x_center + int(blob_radius * np.cos(theta))
        y = y_center + int(blob_radius * np.sin(theta))
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            boundary_points.append((x, y))
    return boundary_points

# 이미지 파일 읽기
image_file = 'reflec_image.png'
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"이미지 파일 '{image_file}'을(를) 찾을 수 없습니다.")

# 원본 이미지에서 블랍 감지
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 0
params.filterByArea = True
params.minArea = 5
params.maxArea = 2000
params.filterByCircularity = True
params.minCircularity = 0.7
params.filterByConvexity = True
params.minConvexity = 0.8
params.filterByInertia = True
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image)

# 가로로 200 픽셀 구간에서 블랍 개수 세기
height, width = image.shape
num_sections = width // 200
blob_count = np.zeros(num_sections)

for keypoint in keypoints:
    x = int(keypoint.pt[0])
    blob_count[x // 200] += 1

# 블랍이 가장 많은 구간 찾기
max_blob_index = np.argmax(blob_count)
start_x = max_blob_index * 200
end_x = start_x + 200

# 해당 구간에서 가운데에 블랍이 분포하는 부분 선택
cropped_image = image[:, start_x:end_x]

# 픽셀을 16개의 픽셀로 쪼개기
height, width = cropped_image.shape
enlarged_image = cv2.resize(cropped_image, (width * 4, height * 4), interpolation=cv2.INTER_NEAREST)

# 자동으로 파라미터를 조정하여 블랍 감지
keypoints_enlarged, optimal_params = auto_adjust_params(enlarged_image)

# 블랍 경계 점 찾기
all_boundary_points = []
for keypoint in keypoints_enlarged:
    center = keypoint.pt
    radius = keypoint.size / 2
    boundary_points = find_blob_boundary_points(center, radius, enlarged_image.shape)
    all_boundary_points.append(boundary_points)

# 각 블랍의 경계 점을 그리기
for points in all_boundary_points:
    for point in points:
        enlarged_image[point[1], point[0]] = 255

# 블랍을 이미지에 그리기
im_with_keypoints = cv2.drawKeypoints(enlarged_image, keypoints_enlarged, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 이미지를 저장
output_file_with_blobs = 'output_image_with_enlarged_blobs_1.png'
cv2.imwrite(output_file_with_blobs, im_with_keypoints)

# 원래 이미지의 크기를 4배로 확대
original_image_with_enlarged = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4), interpolation=cv2.INTER_NEAREST)

# 확대된 이미지를 원래 이미지에 포함시키기
original_image_with_enlarged[:, start_x * 4:end_x * 4] = enlarged_image

# 결과 이미지를 원래 이미지에 포함시켜서 저장
output_file_combined = 'output_image_combined.png'
cv2.imwrite(output_file_combined, original_image_with_enlarged)

print(f'확대된 블랍을 포함한 이미지를 {output_file_with_blobs} 파일로 저장했습니다.')
print(f'확대된 구간을 포함한 전체 이미지를 {output_file_combined} 파일로 저장했습니다.')

