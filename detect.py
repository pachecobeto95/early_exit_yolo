import os, argparse, tqdm, random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable


def _create_data_loader(img_path: str, batch_size: int, img_size: int, n_cpu: int):
	"""Creates a DataLoader for inferencing.

	img_path: Path to file containing all paths to validation images.
	batch_size: Size of each image batch
	img_size: Size of each image dimension for yolo
	n_cpu: Number of cpu threads to use during batch generation
	return: Returns DataLoader
	"""

	transformations = transforms.Compose([
		transforms.Resize(img_size), 
		transforms.ToTensor()])

	dataset = ImageFolder(img_path,transform=transformations)    
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu, pin_memory=True)

	return dataloader

def detect_dir(img_path, output_path, batch_size=8, img_size=416, n_cpu=8, conf_thres=0.5, nms_thres=0.5):
    """Detects objects on all images in specified directory and saves output images with drawn detections.

    :type img_path: str
    :param output_path: Path to output directory
    :type output_path: str
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    """

	dataloader = create_data_loader(img_path, batch_size, img_size, n_cpu)

	model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

	img_detections, imgs = detect(model, dataloader, output_path, conf_thres, nms_thres)


def detect(model, dataloader: Dataloader, output_path: str, conf_thres: float, nms_thres: float):
	"""Inferences images with model.

	model: Model for inference
	dataloader: Dataloader provides the batches of images to inference
	output_path: Path to output directory
	conf_thres: Object confidence threshold, defaults to 0.5
	nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
	
	return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
		Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
		List of input image paths
	rtype: [Tensor], [str]
	"""

	# Create output directory, if missing
	os.makedirs(output_path, exist_ok=True)

	Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

	img_detections = []  # Stores detections for each image index
	imgs = []  # Stores image paths

	model.eval()  # Set model to evaluation mode

	with torch.no_grad():

	for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
		# Configure input
		input_imgs = Variable(input_imgs.type(Tensor))

		# Get detections
		detections = model(input_imgs)
		detections = non_max_suppression(detections, conf_thres, nms_thres)

		# Store image and detections
		img_detections.extend(detections)
		imgs.extend(img_paths)

	return img_detections, imgs




def run(args)

	detect_dir(args.images_dir, args.output, batch_size=args.batch_size, img_size=args.img_size, 
		n_cpu=args.n_cpu, conf_thres=args.conf_thres, nms_thres=args.nms_thres)

def parse_args()

	parser = argparse.ArgumentParser(description="Detect objects on images.")
	#parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
	#parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
	parser.add_argument("-i", "--images_dir", type=str, default="./img_examples", help="Path to directory with images to inference")
	#parser.add_argument("-c", "--classes", type=str, default="data/coco.names", help="Path to classes label file (.names)")
	parser.add_argument("-o", "--output_dir", type=str, default="./output_img", help="Path to output directory")
	parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
	parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
	parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
	parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
	parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
	
	args = parser.parse_args()

	return args


if (__name__ == '__main__'):

	args = parse_args()
	run(args)
