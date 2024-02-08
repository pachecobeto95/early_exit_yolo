import torch
from PIL import Image

def parse_args()

	parser = argparse.ArgumentParser(description="Detect objects on images.")
	parser.add_argument("-i", "--images_dir", type=str, default="./img_examples", help="Path to directory with images to inference")
	parser.add_argument("-o", "--output_dir", type=str, default="./output_img", help="Path to output directory")
	
	args = parser.parse_args()

	return args

def detect(args)

	# Model
	model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

	img_list = []

	# Images
	for f in ['zidane.jpg', 'bus.jpg']:

		img_path = args.images_dir +"/%s" %(f)
		
		torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, img_path)  # download 2 images

		img_list.append(Image.open(img_path))  # PIL image

	# Inference
	results = model(img_list, size=640)  # batch of images

	results.save()



if (__name__ == '__main__'):

	args = parse_args()
	run(args)
