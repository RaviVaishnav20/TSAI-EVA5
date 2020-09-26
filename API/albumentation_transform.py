from utils import *
from albumentations import *
from albumentations.pytorch.transforms import ToTensor

class AlbumentationTransformations(object):
    """
## Pixel-level transforms
Pixel-level transforms will change just an input image and will leave any additional targets such as masks, bounding boxes, and keypoints unchanged. The list of pixel-level transforms:

- Blur
- CLAHE
- ChannelDropout
- ChannelShuffle
- Downscale
- Equalize
- FDA
- FancyPCA
- FromFloat
- GaussNoise
- GaussianBlur
- GlassBlur
- HistogramMatching
- HueSaturationValue
- IAAAdditiveGaussianNoise
- IAAEmboss
- IAASharpen
- IAASuperpixels
- ISONoise
- ImageCompression
- InvertImg
- MedianBlur
- MotionBlur
- MultiplicativeNoise
- Normalize
- Posterize
- RGBShift
- RandomBrightnessContrast
- RandomFog
- RandomGamma
- RandomRain
- RandomShadow
- RandomSnow
- RandomSunFlare
- Solarize
- ToFloat
- ToGray
- ToSepia

## Spatial-level transforms
Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. The following table shows which additional targets are supported by each transform.

Transform	Image	Masks	BBoxes	Keypoints
CenterCrop	✓	✓	✓	✓
CoarseDropout	✓	✓
Crop	✓	✓	✓	✓
CropNonEmptyMaskIfExists	✓	✓	✓	✓
ElasticTransform	✓	✓
Flip	✓	✓	✓	✓
GridDistortion	✓	✓
GridDropout	✓	✓
HorizontalFlip	✓	✓	✓	✓
IAAAffine	✓	✓	✓	✓
IAACropAndPad	✓	✓	✓	✓
IAAFliplr	✓	✓	✓	✓
IAAFlipud	✓	✓	✓	✓
IAAPerspective	✓	✓	✓	✓
IAAPiecewiseAffine	✓	✓	✓	✓
Lambda	✓	✓	✓	✓
LongestMaxSize	✓	✓	✓	✓
MaskDropout	✓	✓
NoOp	✓	✓	✓	✓
OpticalDistortion	✓	✓
PadIfNeeded	✓	✓	✓	✓
RandomCrop	✓	✓	✓	✓
RandomCropNearBBox	✓	✓	✓	✓
RandomGridShuffle	✓	✓
RandomResizedCrop	✓	✓	✓	✓
RandomRotate90	✓	✓	✓	✓
RandomScale	✓	✓	✓	✓
RandomSizedBBoxSafeCrop	✓	✓	✓
RandomSizedCrop	✓	✓	✓	✓
Resize	✓	✓	✓	✓
Rotate	✓	✓	✓	✓
ShiftScaleRotate	✓	✓	✓	✓
SmallestMaxSize	✓	✓	✓	✓
Transpose	✓	✓	✓	✓
VerticalFlip	✓	✓	✓
    """
    def __init__(self, trans_list):
      self.album_transforms = Compose(trans_list)

    def __call__(self, img):
        img = np.array(img)
        img = self.album_transforms(image=img)['image']
        return img