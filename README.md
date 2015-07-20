# OpenCVAR
Computational photography final project

## Record Video

```
python video.py -o nick_dancing.mp4
```

## Showing Matched Keypoints

```
python video.py -d -k matches -i buzz.bmp -o keypoints.mp4
```

![Matched Keypoints](http://i.imgur.com/8NnY5j2.gif)
<iframe width="420" height="315" src="https://www.youtube.com/embed/BKH8z2Yr1S4" frameborder="0" allowfullscreen></iframe>

## Bounding Box

```
python video.py -d -k matches -r -i buzz.bmp -o box.mp4
```

![Bounding Box](http://i.imgur.com/HzONwXh.gif)
<iframe width="420" height="315" src="https://www.youtube.com/embed/g28XrxfDJ80" frameborder="0" allowfullscreen></iframe>

## Blur

```
python video.py -d -r -b -i bulldog.png -o blur.mp4
```

![Blur](http://i.imgur.com/HKTmpQH.gif)
<iframe width="420" height="315" src="https://www.youtube.com/embed/SEqr9iqpNN4" frameborder="0" allowfullscreen></iframe>

## Image overlay

```
python video.py -d -i bulldog.png -p buzz.bmp -o image.mp4
```

![Image Overlay](http://i.imgur.com/4vVGegP.gif)
<iframe width="420" height="315" src="https://www.youtube.com/embed/Vpcl604ui0o" frameborder="0" allowfullscreen></iframe>

## Video Overlay

```
python video.py -d -i qr_code.png -v nick_dancing.mp4 -o video.mp4
```
![Video Overlay](http://i.imgur.com/ADGH8h2.gif)
<iframe width="420" height="315" src="https://www.youtube.com/embed/bWGAgWVjzYI" frameborder="0" allowfullscreen></iframe>

## Dependencies
* Python 2.7.10
* OpenCV 2.4.11

## Usage
```
usage: video.py [-h] [-i INPUT] [-p PASTE] [-o OUTPUT] [-b] [-r]
                [-k {keypoints,matches}] [-n NUMMATCHES]

Augmented reality using OpenCV.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Image to find (default=match.png)
  -p PASTE, --paste PASTE
                        Image to paste over found feature
  -o OUTPUT, --output OUTPUT
                        Output video file name (default=output.mp4)
  -b, --blur            Blur found feature (default=False)
  -r, --rectangle       Draw rectange around feature (default=False)
  -k {keypoints,matches}, --keypoints {keypoints,matches}
                        Draw all feature keypoints or just matches
  -n NUMMATCHES, --nummatches NUMMATCHES
                        Minimum number of matches (default=20)
```

## Examples

```
python video.py -b -r
```
[YouTube]()
