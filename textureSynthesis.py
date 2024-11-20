import numpy as np
import matplotlib as plt
import scipy.stats as st #가우시안 커널을 위한 라이브러리
import scipy.misc
import os

from random import randint, gauss
from math import floor
from skimage import io, feature, transform 
from IPython.display import clear_output

# gif 만들기 위한 라이브러리
import imageio 
from PIL import Image

def textureSynthesis(exampleMapPath, outputSize, searchKernelSize, savePath, attenuation = 80, truncation = 0.8, snapshots = True):
    
    # 매개변수 설정
    PARM_attenuation = attenuation
    PARM_truncation = truncation
    # 파라미터를 파일에 기록
    text_file = open(savePath + 'params.txt', "w")
    text_file.write("Attenuation: %d \n Truncation: %f \n KernelSize: %d" % (PARM_attenuation, PARM_truncation, searchKernelSize))
    text_file.close()
    
    # searchKernelSize가 홀수인지 확인
    if searchKernelSize % 2 == 0:
        searchKernelSize = searchKernelSize + 1
        
    # 예시 맵 이미지를 로드
    exampleMap = loadExampleMap(exampleMapPath)
    imgRows, imgCols, imgChs = np.shape(exampleMap) 
    
    # 생성할 이미지를 초기화 = 캔버스; + 랜덤한 3x3 패치를 캔버스 중앙에 배치
    canvas, filledMap = initCanvas(exampleMap, outputSize)
    
    # 예시 맵에서 예시 패치 배열을 미리 계산
    examplePatches = prepareExamplePatches(exampleMap, searchKernelSize)
    
    # 해상도가 필요한 픽셀 찾기 (해상된 이웃의 개수에 가중치 부여)
    resolved_pixels = 3 * 3
    pixels_to_resolve = outputSize[0]*outputSize[1]
    
    # 메인 루프-------------------------------------------------------
    
    # 해결해야 할 최고의 후보 맵 초기화 (정보 재사용을 원함)
    bestCandidateMap = np.zeros(np.shape(filledMap))
    
    while resolved_pixels < pixels_to_resolve:
        
        # 후보 맵 업데이트
        updateCandidateMap(bestCandidateMap, filledMap, 5)

        # 최고의 후보 좌표 가져오기
        candidate_row, candidate_col = getBestCandidateCoord(bestCandidateMap, outputSize)

        # 비교할 후보 패치 가져오기
        candidatePatch = getNeighbourhood(canvas, searchKernelSize, candidate_row, candidate_col)

        # 마스크 맵 가져오기
        candidatePatchMask = getNeighbourhood(filledMap, searchKernelSize, candidate_row, candidate_col)
        # 가우시안으로 가중치 부여
        candidatePatchMask *= gkern(np.shape(candidatePatchMask)[0], np.shape(candidatePatchMask)[1])
        # 3D 배열로 변환
        candidatePatchMask = np.repeat(candidatePatchMask[:, :, np.newaxis], 3, axis=2)

        # 이제 모든 예시 패치와 비교하고 거리 측정값을 구성해야 함
        # 예시 패치의 차원에 맞게 복사
        examplePatches_num = np.shape(examplePatches)[0]
        candidatePatchMask = np.repeat(candidatePatchMask[np.newaxis, :, :, :, ], examplePatches_num, axis=0)
        candidatePatch = np.repeat(candidatePatch[np.newaxis, :, :, :, ], examplePatches_num, axis=0)

        distances = candidatePatchMask * pow(examplePatches - candidatePatch, 2)
        distances = np.sum(np.sum(np.sum(distances, axis=3), axis=2), axis=1) # 패치의 모든 픽스를 단일 숫자로 합산

        # 거리를 확률로 변환 
        probabilities = distances2probability(distances, PARM_truncation, PARM_attenuation)
        
        # 구성된 PMF를 샘플링하고 적절한 픽셀 값을 가져오기
        sample = np.random.choice(np.arange(examplePatches_num), 1, p=probabilities)
        chosenPatch = examplePatches[sample]
        halfKernel = floor(searchKernelSize / 2)
        chosenPixel = np.copy(chosenPatch[0, halfKernel, halfKernel])

        # 픽셀 해상화
        canvas[candidate_row, candidate_col, :] = chosenPixel
        filledMap[candidate_row, candidate_col] = 1

        # 실시간 업데이트 표시
        plt.pyplot.imshow(canvas)
        clear_output(wait=True)
        display(plt.pyplot.show())

        resolved_pixels = resolved_pixels+1
        
        # 이미지 저장
        if snapshots:
            img = Image.fromarray(np.uint8(canvas*255))
            img = img.resize((300, 300), resample=0, box=None)
            img.save(savePath + 'out' + str(resolved_pixels-9) + '.jpg')

    # 이미지 저장
    if snapshots==False:
        img = Image.fromarray(np.uint8(canvas*255))
        img = img.resize((300, 300), resample=0, box=None)
        img.save(savePath + 'out.jpg')

def distances2probability(distances, PARM_truncation, PARM_attenuation):
    
    probabilities = 1 - distances / np.max(distances)  
    probabilities *= (probabilities > PARM_truncation)
    probabilities = pow(probabilities, PARM_attenuation) # 값 감쇄
    # 모든 값을 잘라내지 않았는지 확인
    if np.sum(probabilities) == 0:
        # 그렇다면 원래대로 되돌리기
        probabilities = 1 - distances / np.max(distances) 
        probabilities *= (probabilities > PARM_truncation*np.max(probabilities)) # 값을 잘라내기 (상위 %를 원함)
        probabilities = pow(probabilities, PARM_attenuation)
    probabilities /= np.sum(probabilities) # 합이 1이 되도록 정규화  
    
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
        exampleMap = exampleMap[:,:,:3] # 알파 채널 제거
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
    # bestCandidateMap이 비어 있는지 확인
    if np.argmax(bestCandidateMap) == 0:
        # 처음부터 채우기
        for r in range(np.shape(bestCandidateMap)[0]):
            for c in range(np.shape(bestCandidateMap)[1]):
                bestCandidateMap[r, c] = np.sum(getNeighbourhood(filledMap, kernelSize, r, c))

def initCanvas(exampleMap, size):
    
    # 예시 맵의 차원 가져오기
    imgRows, imgCols, imgChs = np.shape(exampleMap)
    
    # 빈 캔버스 생성 
    canvas = np.zeros((size[0], size[1], imgChs)) # 예시 맵에서 채널 수를 상속받음
    filledMap = np.zeros((size[0], size[1])) # 어떤 픽셀이 해결되었는지를 나타내는 맵
    
    # 랜덤한 3x3 블록 초기화
