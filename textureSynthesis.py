import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st # 가우시안 커널을 위한 라이브러리
import scipy.misc
import os

from random import randint, gauss
from math import floor
from skimage import io, feature, transform 
from IPython.display import clear_output

# GIF 만들기 위한 라이브러리
import imageio 
from PIL import Image

def textureSynthesis(exampleMapPath, outputSize, searchKernelSize, savePath, attenuation = 80, truncation = 0.8, snapshots = True):
    
    # 파라미터 설정
    PARM_attenuation = attenuation
    PARM_truncation = truncation
    # 파라미터 저장
    text_file = open(savePath + 'params.txt', "w")
    text_file.write("Attenuation: %d \n Truncation: %f \n KernelSize: %d" % (PARM_attenuation, PARM_truncation, searchKernelSize))
    text_file.close()
    
    # searchKernelSize가 홀수인지 확인
    if searchKernelSize % 2 == 0:
        searchKernelSize = searchKernelSize + 1
        
    # 예제 맵 이미지 로드
    exampleMap = loadExampleMap(exampleMapPath)
    imgRows, imgCols, imgChs = np.shape(exampleMap) 
    
    # 캔버스 초기화: 랜덤한 3x3 패치를 캔버스 중앙에 배치
    canvas, filledMap = initCanvas(exampleMap, outputSize)
    
    # 예제 맵의 패치 배열 미리 계산
    examplePatches = prepareExamplePatches(exampleMap, searchKernelSize)
    
    # 해상도가 필요한 픽셀 찾기 (해결된 이웃의 개수로 가중치 부여)
    resolved_pixels = 3 * 3
    pixels_to_resolve = outputSize[0]*outputSize[1]
    
    # 메인 루프-------------------------------------------------------
    
    # 해결이 필요한 후보 맵 초기화 (정보를 재사용하고자 함)
    bestCandidateMap = np.zeros(np.shape(filledMap))
    
    while resolved_pixels < pixels_to_resolve:
        
        # 후보 맵 업데이트
        updateCandidateMap(bestCandidateMap, filledMap, 5)

        # 최상의 후보 좌표 가져오기
        candidate_row, candidate_col = getBestCandidateCoord(bestCandidateMap, outputSize)

        # 후보 패치 비교를 위한 패치 가져오기
        candidatePatch = getNeighbourhood(canvas, searchKernelSize, candidate_row, candidate_col)

        # 마스크 맵 가져오기
        candidatePatchMask = getNeighbourhood(filledMap, searchKernelSize, candidate_row, candidate_col)
        # 가우시안으로 가중치 부여
        candidatePatchMask *= gkern(np.shape(candidatePatchMask)[0], np.shape(candidatePatchMask)[1])
        # 3D 배열로 변환
        candidatePatchMask = np.repeat(candidatePatchMask[:, :, np.newaxis], 3, axis=2)

        # 이제 예제 패치와 비교하여 거리 메트릭 구성
        # 예제 패치의 차원에 맞게 복사
        examplePatches_num = np.shape(examplePatches)[0]
        candidatePatchMask = np.repeat(candidatePatchMask[np.newaxis, :, :, :, ], examplePatches_num, axis=0)
        candidatePatch = np.repeat(candidatePatch[np.newaxis, :, :, :, ], examplePatches_num, axis=0)

        distances = candidatePatchMask * pow(examplePatches - candidatePatch, 2)
        distances = np.sum(np.sum(np.sum(distances, axis=3), axis=2), axis=1) # 패치의 모든 픽셀을 단일 숫자로 합산

        # 거리를 확률로 변환
        probabilities = distances2probability(distances, PARM_truncation, PARM_attenuation)
        
        # 생성된 PMF에서 샘플링하고 적절한 픽셀 값 가져오기
        sample = np.random.choice(np.arange(examplePatches_num), 1, p=probabilities)
        chosenPatch = examplePatches[sample]
        halfKernel = floor(searchKernelSize / 2)
        chosenPixel = np.copy(chosenPatch[0, halfKernel, halfKernel])

        # 픽셀 해결
        canvas[candidate_row, candidate_col, :] = chosenPixel
        filledMap[candidate_row, candidate_col] = 1

        # 실시간 업데이트 표시
        plt.imshow(canvas)
        clear_output(wait=True)
        display(plt.show())

        resolved_pixels = resolved_pixels + 1
        
        # 이미지 저장
        if snapshots:
            img = Image.fromarray(np.uint8(canvas*255))
            img = img.resize((300, 300), resample=0, box=None)
            img.save(savePath + 'out' + str(resolved_pixels-9) + '.jpg')

    # 이미지 저장
    if snapshots == False:
        img = Image.fromarray(np.uint8(canvas*255))
        img = img.resize((300, 300), resample=0, box=None)
        img.save(savePath + 'out.jpg')

def distances2probability(distances, PARM_truncation, PARM_attenuation):
    
    probabilities = 1 - distances / np.max(distances)  
    probabilities *= (probabilities > PARM_truncation)
    probabilities = pow(probabilities, PARM_attenuation) # 값 감쇠
    # 모든 값을 잘라내지 않았는지 확인
    if np.sum(probabilities) == 0:
        # 그렇다면 되돌리기
        probabilities = 1 - distances / np.max(distances) 
        probabilities *= (probabilities > PARM_truncation * np.max(probabilities)) # 값을 잘라냄 (상위 %를 원함)
        probabilities = pow(probabilities, PARM_attenuation)
    probabilities /= np.sum(probabilities) # 더해서 1이 되도록 정규화  
    
    return probabilities

def getBestCandidateCoord(bestCandidateMap, outputSize):
    
    candidate_row = floor(np.argmax(bestCandidateMap) / outputSize[0])
    candidate_col = np.argmax(bestCandidateMap) - candidate_row * outputSize[1]
    
    return candidate_row, candidate_col

def loadExampleMap(exampleMapPath):
    exampleMap = io.imread(exampleMapPath) # MxNx3 배열 반환
    exampleMap = exampleMap / 255.0 # 정규화
    # 3채널 RGB인지 확인
    if (np.shape(exampleMap)[-1] > 3): 
        exampleMap = exampleMap[:, :, :3] # 알파 채널 제거
    elif (len(np.shape(exampleMap)) == 2):
        exampleMap = np.repeat(exampleMap[np.newaxis, :, :], 3, axis=0) # 그레이스케일에서 RGB로 변환
    return exampleMap

def getNeighbourhood(mapToGetNeighbourhoodFrom, kernelSize, row, col):
    
    halfKernel = floor(kernelSize / 2)
    
    if mapToGetNeighbourhoodFrom.ndim == 3:
        npad = ((halfKernel, halfKernel), (halfKernel, halfKernel), (0, 0))
    elif mapToGetNeighbourhoodFrom.ndim == 2:
        npad = ((halfKernel, halfKernel), (halfKernel, halfKernel))
    else:
        print('ERROR: getNeighbourhood 함수가 잘못된 차원의 맵을 받았습니다!')
        
    paddedMap = np.lib.pad(mapToGetNeighbourhoodFrom, npad, 'constant', constant_values=0)
    
    shifted_row = row + halfKernel
    shifted_col = col + halfKernel
    
    row_start = shifted_row - halfKernel
    row_end = shifted_row + halfKernel + 1
    col_start = shifted_col - halfKernel
    col_end = shifted_col + halfKernel + 1
    
    return paddedMap[row_start:row_end, col_start:col_end]

def updateCandidateMap(bestCandidateMap, filledMap, kernelSize):
    bestCandidateMap *= 1 - filledMap # 해결된 픽셀을 맵에서 제거
    # bestCandidateMap이 비어있는지 확인
    if np.argmax(bestCandidateMap) == 0:
        # 처음부터 채우기
        for r in range(np.shape(bestCandidateMap)[0]):
            for c in range(np.shape(bestCandidateMap)[1]):
                bestCandidateMap[r, c] = np.sum(getNeighbourhood(filledMap, kernelSize, r, c))

def initCanvas(exampleMap, size):
    
    # 예제 맵 차원 가져오기
    imgRows, imgCols, imgChs = np.shape(exampleMap)
    
    # 빈 캔버스 생성 
    canvas = np.zeros((size[0], size[1], imgChs)) # 예제 맵의 채널 수를 상속
    filledMap = np.zeros((size[0], size[1])) # 어떤 픽셀이 해결되었는지 표시하는 맵
    
    # 랜덤한 3x3 블록 초기화
    margin = 1
    rand_row = randint(margin, imgRows - margin - 1)
    rand_col = randint(margin, imgCols - margin - 1)
    exampleMap_patch = exampleMap[rand_row-margin:rand_row+margin+1, rand_col-margin:rand_col+margin+1] # 마지막 요소가 포함되지 않도록 +1 필요

    # 캔버스 중앙에 패치 배치
    center_row = floor(size[0] / 2)
    center_col = floor(size[1] / 2)
    canvas[center_row-margin:center_row+margin+1, center_col-margin:center_col+margin+1] = exampleMap_patch
    filledMap[center_row-margin:center_row+margin+1, center_col-margin:center_col+margin+1] = 1 # 이 픽셀들을 해결된 것으로 표시

    return canvas, filledMap

def prepareExamplePatches(exampleMap, searchKernelSize):
    
    # 예제 맵 차원 가져오기
    imgRows, imgCols, imgChs = np.shape(exampleMap)
    
    # 이미지에서 슬라이드할 수 있는 검색 창의 가능한 단계 찾기
    num_horiz_patches = imgRows - (searchKernelSize - 1)
    num_vert_patches = imgCols - (searchKernelSize - 1)
    
    # 후보 배열 초기화
    examplePatches = np.zeros((num_horiz_patches * num_vert_patches, searchKernelSize, searchKernelSize, imgChs))
    
    # 배열 채우기
    for r in range(num_horiz_patches):
        for c in range(num_vert_patches):
            examplePatches[r * num_vert_patches + c] = exampleMap[r:r + searchKernelSize, c:c + searchKernelSize]
            
    return examplePatches

def gkern(kern_x, kern_y, nsig=3):
    """2D 가우시안 커널 배열을 반환합니다."""
    """https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy에서 수정된 코드입니다."""

    # X
    interval = (2 * nsig + 1.) / (kern_x)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_x + 1)
    kern1d_x = np.diff(st.norm.cdf(x))
    # Y
    interval = (2 * nsig + 1.) / (kern_y)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_y + 1)
    kern1d_y = np.diff(st.norm.cdf(x))
    
    kernel_raw = np.sqrt(np.outer(kern1d_x, kern1d_y))
    kernel = kernel_raw / kernel_raw.sum()
    
    return kernel

