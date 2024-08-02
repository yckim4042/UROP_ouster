import cv2
import numpy as np

# 이미지 파일 읽기
image_file = 'reflec_image.png'
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"이미지 파일 '{image_file}'을(를) 찾을 수 없습니다.")

# 블랍 감지기 설정
params = cv2.SimpleBlobDetector_Params()

# 검은색 원을 찾기 위한 설정
params.filterByColor = True
params.blobColor = 0

# 블랍의 크기를 제한하기 위한 설정
params.filterByArea = True
params.minArea = 30  # 최소 면적
params.maxArea = 200  # 최대 면적

# 원형 블랍을 찾기 위한 설정
params.filterByCircularity = True
params.minCircularity = 0.7

# 블랍 감지기 생성
detector = cv2.SimpleBlobDetector_create(params)

# 이미지에서 블랍 감지
keypoints = detector.detect(image)

# 원을 이미지에 그리기
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 이미지를 저장
output_file_with_blobs = 'output_image_with_blobs.png'
cv2.imwrite(output_file_with_blobs, im_with_keypoints)

print(f'작은 원형 블랍을 포함한 이미지를 {output_file_with_blobs} 파일로 저장했습니다.')

