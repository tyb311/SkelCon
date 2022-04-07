
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

from albumentations import (Compose, CLAHE, RandomGamma) 
IMAGE_ENHANCE = Compose([CLAHE(p=1), RandomGamma(p=1)])

onnx_seg='onnx/drive.onnx'
# onnx_seg='chase.onnx'
# onnx_seg='stare.onnx'
# onnx_seg='hrf.onnx'

path_img='onnx/01_test.tif'

def demo():
	img = np.array(Image.open(path_img))[:512,:512]
	img = IMAGE_ENHANCE(image=img)['image']
	raw = img

	# img = img[:,:,1]
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	print(img.shape)

	img = img.astype(np.float32)
	h,w = img.shape
	img = img.reshape(1,1,h,w)/255
	out = onnx_inference(img).squeeze()
	out = (out>0.5).astype(np.uint8)*255

	plt.subplot(121),plt.imshow(raw)
	plt.subplot(122),plt.imshow(out)
	plt.show()


def onnx_inference(input=np.random.randn(1, 1, 64, 128).astype(np.float32)):
	ort_session = ort.InferenceSession(onnx_seg)
	outputs = ort_session.run(
		None,
		{"input": input},
	)
	# print(outputs[0].shape)
	return outputs[0]

demo()