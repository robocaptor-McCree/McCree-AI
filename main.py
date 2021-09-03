# -*- coding: utf-8 -*-

#import torch
#import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import json

import matplotlib.pyplot as plt#시각화를 위한 package

import argparse

from glob import glob
from pathlib import Path


def parse_args(args):

    parser = argparse.ArgumentParser(description='Robocaptor MacCree')

    #TODO: model list 작성하기
    parser.add_argument('--model-help', help="", type=str)
    parser.add_argument('--mode', help='single-image or directory', default="directory", type=str)
    parser.add_argument('--single-img', help='single image path', type=str)
    parser.add_argument('--dir', help='directory path', default="data/origin_img", type=str)
    parser.add_argument('--save', help='save directory path',  default="data/perturbed_img/", type=str)
    parser.add_argument('--annotation', help='annotation file', default="data/imagenet_classes.json", type=str)

    #TODO: attack에 사용할 모델
    parser.add_argument('--model', help='', default="", type=str)

    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def fgsm_attack(img, epsilon, gradient):
    # 기울기값의 원소의 sign 값을 구함
    sign_gradient = gradient.sign()
    # 이미지 각 픽셀의 값을 sign_gradient 방향으로 epsilon 만큼 조절
    perturbed_img = img + epsilon * sign_gradient
    # [0,1] 범위를 벗어나는 값을 조절
    perturbed_img = torch.clamp(perturbed_img, 0, 1)
    return perturbed_img


def attack(tensor_img, model):
    # 이미지의 기울기값을 구하도록 설정
    tensor_img.requires_grad_(True)

    # 이미지를 모델에 통과시킴
    output = model(tensor_img)

    # 오차값 구하기 (레이블 263은 웰시코기)
    loss = F.nll_loss(output, torch.tensor([263]))

    # 기울기값 구하기
    model.zero_grad()
    loss.backward()
    # 미분값을 저장하여, gradient 값 추출

    # 이미지의 기울기값을 추출
    gradient = tensor_img.grad.data

    # FGSM 공격으로 적대적 예제 생성
    epsilon = 0.03
    perturbed_img = fgsm_attack(tensor_img, epsilon, gradient)

    return perturbed_img



def image2tensor(img):
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
    ])

    tensor_img = img_transforms(img)
    tensor_img = tensor_img.unsqueeze(0)

    print("이미지 텐서 모양:", tensor_img.size())
    return tensor_img


def main(args=None):

    args = parse_args(args)

    model = models.resnet101(pretrained=True)  # Model로 ResNet-101버전 로딩
    model.eval()
    print(model)  # 모델 구조 프린트

    CLASSES = json.load(open(args.annotation))
    # 학습을 위한 데이터셋(클래스 Json)파일 로딩
    idx2class = [CLASSES[str(i)] for i in range(1000)]
    # Class 별 ID 셋팅

    # 이미지 불러오기
    if args.mode == 'single-image':
        img = Image.open(args.single_img)
        img_name = args.single_img

        # 이미지를 텐서로 변환하기
        tensor_img = image2tensor(img)
        perturbed_img = attack(tensor_img, model)

        # 텐서를 넘파이행렬로 변환
        perturbed_img_view = perturbed_img.squeeze(0).detach()
        perturbed_img_view = perturbed_img_view.transpose(0, 2).transpose(0, 1).numpy()

        # 이미지 하나 저장
        plt.imsave(args.save + img_name, perturbed_img_view)


    else:
        imgs = glob(args.dir + '/*')

        for img in imgs:
            img_name = img
            img = Image.open(img)
            img_name = img_name

            # 이미지를 텐서로 변환하기
            tensor_img = image2tensor(img)
            perturbed_img = attack(tensor_img, model)

            # 텐서를 넘파이행렬로 변환
            perturbed_img_view = perturbed_img.squeeze(0).detach()
            perturbed_img_view = perturbed_img_view.transpose(0, 2).transpose(0, 1).numpy()

            # 이미지 저
            plt.imsave(args.save + img_name, perturbed_img_view)



    """
    오리지널 이미지 시각화 코드 
    
    # 시각화를 위해 넘파이 행렬 변환
    original_img_view = img_tensor.squeeze(0).detach()  # [1, 3, 244, 244] -> [3, 244, 244]
    original_img_view = original_img_view.transpose(0, 2).transpose(0, 1).numpy()

    # 텐서 시각화
    plt.imshow(original_img_view)
    """

    """
    original data classification 테스트 코드
    
    output = model(img_tensor)
    prediction = output.max(1, keepdim=False)[1]
    # 가장 확률이 높은 예측 클래스(prediction)

    prediction_idx = prediction.item()
    prediction_name = idx2class[prediction_idx]

    print("예측된 레이블 번호:", prediction_idx)
    print("레이블 이름:", prediction_name)
    """

    """
    perturbed data classification 테스트 코드
    
        # 생성된 적대적 예제를 모델에 통과시킴
    output = model(perturbed_data)



    perturbed_prediction = output.max(1, keepdim=True)[1]

    perturbed_prediction_idx = perturbed_prediction.item()
    perturbed_prediction_name = idx2class[perturbed_prediction_idx]

    print("예측된 레이블 번호:", perturbed_prediction_idx)
    print("레이블 이름:", perturbed_prediction_name)

    # 시각화를 위해 넘파이 행렬 변환
    perturbed_data_view = perturbed_data.squeeze(0).detach()
    perturbed_data_view = perturbed_data_view.transpose(0, 2).transpose(0, 1).numpy()

    plt.imshow(perturbed_data_view)

    f, a = plt.subplots(1, 2, figsize=(10, 10))

    # 원본
    a[0].set_title(prediction_name)
    a[0].imshow(original_img_view)

    # 적대적 예제
    a[1].set_title(perturbed_prediction_name)
    a[1].imshow(perturbed_data_view)
    """


if __name__ == '__main__':
    main()