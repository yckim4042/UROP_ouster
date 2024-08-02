import cv2
import numpy as np

# 이미지 파일 읽기
image_file = 'reflec_image.png'
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"이미지 파일 '{image_file}'을(를) 찾을 수 없습니다.")

# 블랍 감지기 설정
params = cv2.SimpleBlobDetector_Params()

# 블랍의 색상 필터링 설정
params.filterByColor = True
params.blobColor = 0  # 검은색 블랍을 찾음

# 블랍의 크기 필터링 설정
params.filterByArea = True
params.minArea = 30  # 최소 면적
params.maxArea = 200  # 최대 면적

# 블랍의 원형도 필터링 설정
params.filterByCircularity = True
params.minCircularity = 0.1  # 최소 원형도

# 블랍 감지기 생성
detector = cv2.SimpleBlobDetector_create(params)

# 이미지에서 블랍 감지
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

# 블랍 감지기 설정 재조정 (확대된 이미지에 대해)
params_enlarged = cv2.SimpleBlobDetector_Params()

params_enlarged.filterByColor = True
params_enlarged.blobColor = 0

params_enlarged.filterByArea = True
params_enlarged.minArea = 200
params_enlarged.maxArea = 2000

# 확대된 이미지에 대한 블랍 감지기 생성
detector_enlarged = cv2.SimpleBlobDetector_create(params_enlarged)

# 확대된 이미지에서 블랍 감지
keypoints_enlarged = detector_enlarged.detect(enlarged_image)
im_with_keypoints = cv2.drawKeypoints(enlarged_image, keypoints_enlarged, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 이미지를 저장
output_file_with_blobs = 'output_image_with_enlarged_blobs.png'
cv2.imwrite(output_file_with_blobs, im_with_keypoints)

# 원래 이미지의 크기를 4배로 확대
original_image_with_enlarged = cv2.resize(image, (width * 4 * num_sections, height * 4), interpolation=cv2.INTER_NEAREST)

# 확대된 이미지를 원래 이미지에 포함시키기
original_image_with_enlarged[:, start_x * 4:end_x * 4] = enlarged_image

# 결과 이미지를 원래 이미지에 포함시켜서 저장
output_file_combined = 'output_image_combined.png'
cv2.imwrite(output_file_combined, original_image_with_enlarged)

print(f'확대된 블랍을 포함한 이미지를 {output_file_with_blobs} 파일로 저장했습니다.')
print(f'확대된 구간을 포함한 전체 이미지를 {output_file_combined} 파일로 저장했습니다.')

