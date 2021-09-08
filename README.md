# McCree-AI

[![license](https://img.shields.io/github/license/robocaptor-McCree/McCree-Web.svg)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

McCree의 Adversarial image 생성 프로그램이다.

FGSM(fast gradient sign method)를 사용하여 adversarial image를 생성한다. 

입력 이미지에 대한 손실한수의 Gradient를 계산하여 그 손실을 최대화하는 이미지를 생성한다. 

이 과정은 다음과 같은 수식으로 정리할 수 있다.

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/51123268/132456623-4418ba4a-f88d-46b1-aea0-f7cbf3a26c04.gif)

![image](https://user-images.githubusercontent.com/51123268/132457049-8fe160a1-b6a6-4dc7-8257-3a04ae824cf5.png)

![image](https://user-images.githubusercontent.com/51123268/132457060-6a712f7f-a526-413e-87a4-82b6605908fa.png)

![image](https://user-images.githubusercontent.com/51123268/132457063-73701f99-d38e-46c9-ab1a-c1731532c1ba.png)

![image](https://user-images.githubusercontent.com/51123268/132457071-c569e027-cffa-4161-aa75-491a1196b665.png)

![image](https://user-images.githubusercontent.com/51123268/132457081-4e677e6c-5954-419b-89ca-177db873e434.png)

![image](https://user-images.githubusercontent.com/51123268/132457085-eb73d6b9-c9a2-4786-9b29-e39dec0cdde1.png)

FGSM은 원본 이미지의 손실을 최대화하는 것을 목표로 하기 때문에, Gradient를 사용한다. 

원본 이미지의 각 픽셀의 손실에 대한 기여도를 Gradient를 통해 계산한 후, 그 기여도에 따라 픽셀값에 왜곡을 추가하여 adversarial image를 생성한다. 

각 픽셀의 기여도는 연쇄 법칙(chain rule)을 이용해 Gradient를 계산하는 것으로 빠르게 파악할 수 있다. 

매크로에 사용될 대상 모델은 학습이 종료된 상태이기 때문에, 신경망의 가중치에 대한 Gradient는 필요하지 않다. 
 
따라서 FGSM은 원본 이미지만으로도 학습이 종료된 상태의 모델(매크로)을 혼란시킬 수 있다.

![image](https://user-images.githubusercontent.com/51123268/132466980-5e9a01c5-58f9-4acd-bd6b-7c2c42b40382.png)


## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Install

```
$ git clone https://github.com/robocaptor-McCree/McCree-AI
$ cd McCree-AI
```
## Usage
 * 원본 이미지를 'McCree-AI/data/origin_img/'에 넣고 main.py를 실행시키면 'McCree-AI/data/perturbed_img/'에 adversarial noise가 적용된 image가 저장된다.
```
$ python main.py
```

* adversarial noise의 강도는 epsilon의 값으로 조절할 수 있다.

  epsilon을 줄이면 noise가 줄어 image의 왜곡이 줄지만, 컴퓨터가 제대로 분류해낼 가능성이 커진다. 

  반대로 epsilon을 크게하면, noise가 늘어나 image의 왜곡이 많아진다. 

  epsilon의 기본값은 0.03으로 설정했다.
  
```python
def fgsm(img, epsilon, gradient):
    sign_gradient = gradient.sign()
    perturbed_img = img + epsilon * sign_gradient
    perturbed_img = torch.clamp(perturbed_img, 0, 1)
    return perturbed_img
```
* gradient는 원본 image에서 추출한다.
```python
def attack(tensor_img, model):
    tensor_img.requires_grad_(True)

    output = model(tensor_img)

    loss = F.nll_loss(output, torch.tensor([263]))

    model.zero_grad()
    loss.backward()
    
    gradient = img_tensor.grad.data

    epsilon = 0.03
    perturbed_img = fgsm_attack(img_tensor, epsilon, gradient)

    return perturbed_img
```

## Contributing

This project exists thanks to all the people who contribute. 

![GitHub Contributors Image](https://contrib.rocks/image?repo=robocaptor-McCree/McCree-AI)


## License

[MIT © robocaptor-McCree.](../LICENSE)
