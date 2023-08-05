import websockets
import asyncio
import tensorflow as tf
import numpy as np
import cv2, base64
import json
import datetime

# Kieran Chao | ZneT Local
#
# This file is responsible for running the server and handling the video processing and streaming. The server uses
# a websocket connection to communicate with the client, and uses asyncio to handle multiple tasks at once. The video processing
# is done using OpenCV and TensorFlow Lite. 
#
# Your device must have a webcam in order to run this program, and it must be connected to the internet. The server also must
# be activated before the client interface is opened, otherwise the client will receive an error.
#
# Both MoveNet models are included in the models folder. By default, the model uses the Thunder model, which is the most accurate 
# out of the two models. However, it is also the slowest. If you want to improve performance by using the Lightning model, then you must:
#
#   1. Change the model path in line 25 to 'models\lite-model_movenet_singlepose_lightning_3.tflite'
#   2. Change the 'resize_with_pad' arguments on lines 171, 237 from 256 to 192 (or uncomment the neighboring lines)



# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(
            model_path='models\lite-model_movenet_singlepose_thunder_3.tflite')
interpreter.allocate_tensors()

# Define the edges between keypoints
EDGES = {
    (0, 1): 'm', # m - magenta, c - cyan, y - yellow (arbitrary values are assigned to each edge)
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


# Function to draw the keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# Function to draw the connections between the keypoints
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)),
                        (int(x2), int(y2)), (0, 0, 255), 2)

# Function to reset state
def reset_state():
    global state
    state = {
        'isCameraToggle': False,
        'isVirtualCanvasToggle': False, 
        'isRecording': False, 
        'fileSelectedPath': '', 
        'projectPath': ''
        }

# Function to update state
def update_state(result):
    global state, videosPath, dataPath
    state.update(result)
    videosPath = fr'{state["projectPath"]}\Videos'
    dataPath = fr'{state["projectPath"]}\Data'

# Initialise global variables and constants
state = {
    'isCameraToggle': False, 
    'isVirtualCanvasToggle': False, 
    'isRecording': False, 
    'fileSelectedPath': '', 
    'projectPath': ''
    }
port = 5000
videosPath = fr'{state["projectPath"]}\Videos'
dataPath = fr'{state["projectPath"]}\Data'

print("Started server on port : ", port)

# This function is called whenever a new client connects to the server. It creates a listener, a stream handler and a video handler for each client.
async def main(websocket):    
    tasks = [
        asyncio.create_task(listener(websocket)),
        asyncio.create_task(streamHandler(websocket)),
        asyncio.create_task(videoHandler(websocket))        
        ]
    try:
        await asyncio.gather(*tasks)
    # When the client disconnects, the state is reset and all the tasks are cancelled.
    except websockets.exceptions.ConnectionClosed:
        print("Client Disconnected !")
        
        reset_state()
        for task in tasks:
            print(task)
            task.cancel()

# This function receives state from the client and updates local state accordingly.
async def listener(websocket):
    print("Client Connected !")          
    while True:       
        print("Awaiting Message") 
        result = await websocket.recv()
        print(f"Received Message: {result}")
        result = json.loads(result)    
        
        update_state(result)
        print(state)

# This function is responsible for handling the video processing when a video is selected.
async def videoHandler(websocket):
    print("Video Listener is listening !")
    currentFile = state['fileSelectedPath']
    while True:
        # If only the virtual canvas is activated and a file has been selected, then the video is processed and the keypoints are sent to the client.            
        if state['isCameraToggle'] == False and state['isVirtualCanvasToggle'] and state['fileSelectedPath'] != '' and currentFile != state['fileSelectedPath']:
            print("Virtual Canvas Activated !")
            # Initialise variables
            keypoints = []
            currentFile = state['fileSelectedPath']
            cap = cv2.VideoCapture(currentFile)
                
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
                    
            size = (frame_width, frame_height)            
                
            vcwriter = cv2.VideoWriter(fr'{videosPath}\vc-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, size)
            
            # Process video and run model
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                        break

                img = frame.copy()
                # img = tf.image.resize_with_pad(
                #     np.expand_dims(img, axis=0), 192, 192)
                img = tf.image.resize_with_pad(
                            np.expand_dims(img, axis=0), 256, 256)
                input_image = tf.cast(img, dtype=tf.float32)

                # Setup input and output
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Make predictions
                interpreter.set_tensor(
                            input_details[0]['index'], np.array(input_image))
                interpreter.invoke()
                keypoints_with_scores = interpreter.get_tensor(
                            output_details[0]['index'])
                keypoints.append(np.squeeze(np.multiply(keypoints_with_scores, [frame.shape[0], frame.shape[1] , 1])).tolist())
                
                virtualcanvas = np.zeros((frame_height, frame_width, 3), np.uint8)

                # Rendering
                draw_connections(virtualcanvas, keypoints_with_scores, EDGES, 0.4)
                draw_keypoints(virtualcanvas, keypoints_with_scores, 0.4)

                vcwriter.write(virtualcanvas) 

            # Create .txt file
            with open(fr'{dataPath}\data-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt', 'w') as file:
                file.write(json.dumps({"keypoints": keypoints}))
                file.close()

            # Send keypoints to client
            await websocket.send(            
                                json.dumps({"keypoints": keypoints})
                            )                                         
            cap.release()
            vcwriter.release()
            print("Processing Successful !")
        else:
            await asyncio.sleep(1)    

# This function is responsible for handling live video processing and streaming.
async def streamHandler(websocket):
    
    print("Stream Listener is listening !")    
    while True:
        await asyncio.sleep(3)
        # If the camera is activated, then both the camera and the virtual canvas are processed and streamed to the client.
        if state['isCameraToggle']:
            print("Camera and Virtual Canvas is Activated !")
            # Initialise variables
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)            
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))                    
            size = (frame_width, frame_height)       
            
            camwriter = None
            vcwriter = None
            
           # Process video and run model
            while cap.isOpened() and state['isCameraToggle']:
                ret, frame = cap.read()
                
            
                if not ret or frame is None:
                    break
                
                img = frame.copy()
                img = tf.image.resize_with_pad(
                            np.expand_dims(img, axis=0), 256, 256)
                input_image = tf.cast(img, dtype=tf.float32)

                # Setup input and output
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                 # Make predictions
                interpreter.set_tensor(
                            input_details[0]['index'], np.array(input_image))
                interpreter.invoke()
                keypoints_with_scores = interpreter.get_tensor(
                            output_details[0]['index'])
                        
                virtualcanvas = np.zeros((frame_height, frame_width, 3), np.uint8)
                # Rendering
                draw_connections(virtualcanvas, keypoints_with_scores, EDGES, 0.4)
                draw_keypoints(virtualcanvas, keypoints_with_scores, 0.4)

                # When the client is recording, the frames are written to a video file.
                if state['isRecording']:
                    if camwriter is None:
                        camwriter = cv2.VideoWriter(fr'{videosPath}\cam-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, size)
                        print("Camera is Recording !")
                    if vcwriter is None:
                        vcwriter = cv2.VideoWriter(fr'{videosPath}\vc-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, size)
                        print("Virtual Canvas is Recording !")
                    camwriter.write(frame)                    
                    vcwriter.write(virtualcanvas)
                else:
                    if camwriter is not None:
                        print("Camera has finished recording !")
                        camwriter.release()
                        camwriter = None
                    if vcwriter is not None:
                        print("Virtual Canvas has finished recording !")
                        vcwriter.release()
                        vcwriter = None
                        
                vcencoded = cv2.imencode('.jpg', virtualcanvas)[1]

                vcdata = str(base64.b64encode(vcencoded))
                vcdata = vcdata[2:len(vcdata)-1]

                camencoded = cv2.imencode('.jpg', frame)[1]

                camdata = str(base64.b64encode(camencoded))
                camdata = camdata[2:len(camdata)-1]
                        
                # Send the frames to the client in real time       
                await websocket.send(            
                            json.dumps({"camdata": camdata, "vcdata": vcdata, "keypoints": np.squeeze(np.multiply(keypoints_with_scores, [frame.shape[0], frame.shape[1] , 1])).tolist()})
                        )             
        
            print("Camera and Virtual Canvas is Deactivated !")
            cap.release()         

# Start the server                  
start_server = websockets.serve(main, port=port)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
asyncio.get_event_loop().set_debug()