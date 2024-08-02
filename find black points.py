import cv2
import numpy as np

# 이미지 파일 읽기
image_file = 'reflec_image.png'
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"이미지 파일 '{image_file}'을(를) 찾을 수 없습니다.")

# 검은색 픽셀의 위치를 찾기 (임계값 10을 기준으로)
low_threshold = 100  # 반사도가 낮은 값의 임계값
too_low_threshold = 10  # 반사도가 너무 낮은 값의 임계값

# 흰색 배경으로 새로운 이미지 생성
new_image = np.ones_like(image) * 255  # 모든 픽셀을 흰색(255)으로 설정

# 검은 포인트만 새로운 이미지에 복사
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] < low_threshold:
            if image[i, j] < too_low_threshold:
                new_image[i, j] = 255  # 너무 낮은 반사도 포인트도 흰색으로 설정
            else:
                new_image[i, j] = image[i, j]  # 원래 값을 유지

# 허프 변환을 사용하여 원 검출
circles = cv2.HoughCircles(new_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

# 원이 검출되었는지 확인
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 원의 중심과 반지름을 가져오기
        center = (i[0], i[1])
        radius = i[2]
        
        # 원의 경계 그리기
        cv2.circle(new_image, center, radius, (0, 0, 0), 2)
        # 원의 중심 그리기
        cv2.circle(new_image, center, 2, (0, 0, 0), 3)

    # 결과 이미지를 저장
    output_file_with_circles = 'output_image_with_circles.png'
    cv2.imwrite(output_file_with_circles, new_image)

    print(f'검출된 원을 포함한 이미지를 {output_file_with_circles} 파일로 저장했습니다.')
else:
    print('이미지에서 원을 찾을 수 없습니다.')

