# OpenCVAR
Computational photography final project

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
