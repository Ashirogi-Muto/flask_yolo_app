from flask import Flask
from flask import request
from flask import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

confThreshold = 0.1
executor = PoolExecutor(5)

app = Flask(__name__)

# For testing
@app.route('/')
def hello_world():
	return 'Hey, we have Flask in a Docker container!'

# the main route method
# extracts image from the request
@app.route('/detect', methods=['POST'])
def detect():
	image_data = request.files['image'].read()
	executor_thread = executor.submit(detect_object_in_image, image_data)
	objects_in_image =  executor_thread.result()
	response = app.response_class(
		response=json.dumps(objects_in_image),
		status=200,
		mimetype='application/json'
	)
	print('detection done')
	return response

def detect_object_in_image(image_data):
	nparr = np.frombuffer(image_data, np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	image_blob = cv2.dnn.blobFromImage(img, 0.00392,(416,416),(0,0,0),True,False)
	return getOutputForImages(image_blob)


def getOutputForImages(image_blob):
	net = getModelNetwork()
	net.setInput(image_blob)
	detections = net.forward(getOutputsNames(net))
	objects_in_image = extract_data_from_detections(detections)
	return objects_in_image

def getClasses():
	classes = []
	with open('coco.names', 'r') as f:
		classes = [line.strip() for line in f.readlines()]
	return classes

def getModelNetwork():
	return cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

def getOutputsNames(network):
	layer_names = network.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in network.getUnconnectedOutLayers()]
	return output_layers


def extract_data_from_detections(detections):
	objects_dict = {}
	classes = getClasses()
	for item in detections:
		for detection in item:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > confThreshold:
				if str(classes[class_id]) not in objects_dict:
					objects_dict[str(classes[class_id])] = float(confidence) * 100
				else:
					if (objects_dict[str(classes[class_id])] <  float(confidence) * 100):
						objects_dict[str(classes[class_id])] <  float(confidence) * 100

	objects_list = []
	for key, value in objects_dict.items():
		objects_list.append({ 'label': key, 'accuracy': round(value, 2) })
	return objects_list

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', threaded=True)