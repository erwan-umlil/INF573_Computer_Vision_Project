# INF573_Computer_Vision_Project
Detection of some object, removal, inpainting.

## Project Structure
```
.
├── README.md
├── requirements.txt

├── segmentation
        ├── segmentation.py
            
├── diffusion
    ├── img
    ├── results
    ├── cv2_inpainting.py
    ├── heat.py
    ├── naive.py

├── SinGAN
    ├── ...

├── edge_connect
    ├── ...

├── images
    ├── ...

├── output
    ├── edge_connect
    ├── heat
    ├── segmentation
    ├── SinGAN

├── main_edge_connect.py
├── main_singan_editing.py
├── main_singan_paint2image.py

```
## Installation
In order to have the good environnement to run this code you need to :
- Create an virtual environnement (optional)
```
python3 -m venv venv
source venv/bin/activate
```

- Install all the needed dependencies
```
pip install -r requirements.txt
```

## Usage
The different scripts have only been tested on PNG images.
### Whole pipeline using SinGAN editing and heat naive colouring
#### CPU
```
python3 main_singan_editing.py --input_name car.png --remove 7 --heat_epochs 10  --ref_name car_heat.png --editing_start_scale 2 --not_cuda
```
#### GPU
```
python3 main_singan_editing.py --input_name car.png --remove 7 --heat_epochs 10  --ref_name car_heat.png --editing_start_scale 2
```

### Whole pipeline using SinGAN paint2image and heat naive colouring
#### CPU
```
python3 main_singan_paint2image.py --input_name car.png --remove 7 --heat_epochs 10  --ref_name car_heat.png --paint_start_scale 2 --not_cuda
```
#### GPU
```
python3 main_singan_paint2image.py --input_name car.png --remove 7 --heat_epochs 10  --ref_name car_heat.png --paint_start_scale 2
```

We provide the possibility to test the different modules one by one.

### Segmentation alone
```
python3 segmentation.py --image images/car.png --output output/segmentation --remove 7
```

### cv2 colouring alone
```
python3 diffusion/cv2_inpainting.py
```

### Naive colouring alone
```
python3 diffusion/naive.py
```

### Heat equation colouring alone
```
python3 diffusion/heat.py --image output/segmentation/s_car.png --mask output/segmentation/s_car_mask.png
```

### SinGAN alone
Please follow the instructions to install and use SinGAN there: https://github.com/tamarott/SinGAN

### edge-connect alone
Please follow the instructions to install and use edge-connect there: https://github.com/knazeri/edge-connect


## List of supported objects to remove and their code number
0: background
1: aeroplane
2: bicycle
3: bird
4: boat
5: bottle
6: bus
7: car
8: cat
9: chair
10: cow
11: dining table
12: dog
13: horse
14: motorbike
15: person
16: potted plant
17: sheep
18: sofa
19: train
20: tv/monitor
