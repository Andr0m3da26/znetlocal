import websockets
import asyncio
import tensorflow as tf
import numpy as np
import cv2, base64
import json
import datetime

interpreter = tf.lite.Interpreter(
            model_path='models\lite-model_movenet_singlepose_thunder_3.tflite')
interpreter.allocate_tensors()

EDGES = {
    (0, 1): 'm',
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

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    # print(shaped.tolist()[11])

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)),
                        (int(x2), int(y2)), (0, 0, 255), 2)

def reset_state():
    global state
    state = {
        'isCameraToggle': False,
        'isVirtualCanvasToggle': False, 
        'isRecording': False, 
        'fileSelectedPath': '', 
        'projectPath': ''
        }

state = {
    'isCameraToggle': False, 
    'isVirtualCanvasToggle': False, 
    'isRecording': False, 
    'fileSelectedPath': '', 
    'projectPath': ''
    }
port = 5000
print("Started server on port : ", port)

# if iscameratoggle is true
#   start stream

# stream:
# initialise camera variables
# initialise camera video writer
# initialise virtual canvas variables 
# initialise virtual canvas video writer

# while iscameratoggle is true
#   read camera frame
#   if isvirtualcanvastoggle is true
#       process frame
#   if isvirtualcanvastoggle and isrecording is true
#       write frame to virtual canvas video writer
#   if isvirtualcanvastoggle is true 
#   send camera frame and/ virtual canvas frame to client   
#   
#   if isrecording is false
#       stop camera recording
#       if virtualcanvastoggle is true
#           stop virtual canvas recording
# stop recording

async def main(websocket):
    
    tasks = [
        asyncio.create_task(listener(websocket)),
        asyncio.create_task(streamHandler(websocket)),
        asyncio.create_task(videoHandler(websocket))
        
        # videolistener(websocket)
        ]
    
    try:
        await asyncio.gather(*tasks)
    except websockets.exceptions.ConnectionClosed:
        print("Client Disconnected !")
        
        reset_state()
        for task in tasks:
            print(task)
            task.cancel()

# receives state from client and updates local state
async def listener(websocket):
    print("Client Connected !")    
    # websocket.send("Connected")
    # result = await websocket.recv()
    # while not websocket.closed:
    #     print(result)
    #     result = json.loads(result)    
    #     asyncio.create_task(onStateChanged(websocket, result))
    #     result = await websocket.recv()
    
        
    while True:       
        print("Awaiting Message") 
        result = await websocket.recv()
        print(f"Received Message: {result}")
        result = json.loads(result)    
        state.update(result)
        print(state)
        # await onStateChanged(websocket, state)
    

# async def onStateChanged(websocket, state):
#     if state['isCameraToggle'] == False and state['isVirtualCanvasToggle'] and state['fileSelectedPath'] != '':
#         await video(websocket)
#     if state['isCameraToggle'] == True:
#         await stream(websocket)

# async def video(websocket):
#     keypoints = []
#     try:
#         cap = cv2.VideoCapture(state['fileSelectedPath'])
            
#         frame_width = int(cap.get(3))
#         frame_height = int(cap.get(4))
                
#         size = (frame_width, frame_height)            
            
#         result = cv2.VideoWriter(f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
#                                         cv2.VideoWriter_fourcc(*'MJPG'),
#                                         10, size)
       

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                     break

#             img = frame.copy()
#                     # img = tf.image.resize_with_pad(
#                     #     np.expand_dims(img, axis=0), 192, 192)
#             img = tf.image.resize_with_pad(
#                         np.expand_dims(img, axis=0), 256, 256)
#             input_image = tf.cast(img, dtype=tf.float32)

#                     # Setup input and output
#             input_details = interpreter.get_input_details()
#             output_details = interpreter.get_output_details()

#                     # Make predictions
#             interpreter.set_tensor(
#                         input_details[0]['index'], np.array(input_image))
#             interpreter.invoke()
#             keypoints_with_scores = interpreter.get_tensor(
#                         output_details[0]['index'])
#             keypoints.append(np.squeeze(np.multiply(keypoints_with_scores, [frame.shape[0], frame.shape[1] , 1])).tolist())
            
#             virtualcanvas = np.zeros((frame_height, frame_width, 3), np.uint8)

#                     # Rendering
#             draw_connections(virtualcanvas, keypoints_with_scores, EDGES, 0.4)
#             draw_keypoints(virtualcanvas, keypoints_with_scores, 0.4)

#             result.write(virtualcanvas) 

#         await websocket.send(            
#                             json.dumps({"keypoints": keypoints})
#                         )                 
            
                    
#         cap.release()
#         result.release()
#     except Exception as e:
#         cap.release()
#         result.release()
        
#     except websockets.connection.ConnectionClosed as e:
#         # print(e)
#         cap.release()
#         result.release()

async def videoHandler(websocket):
    print("Video Listener is listening !")
    currentFile = state['fileSelectedPath']
    while True:
        
            
        if state['isCameraToggle'] == False and state['isVirtualCanvasToggle'] and state['fileSelectedPath'] != '' and currentFile != state['fileSelectedPath']:
            print("Virtual Canvas Activated !")
            keypoints = []
            currentFile = state['fileSelectedPath']
            cap = cv2.VideoCapture(currentFile)
                
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
                    
            size = (frame_width, frame_height)            
                
            vcwriter = cv2.VideoWriter(fr'{state["projectPath"]}\Videos\vc-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, size)
            

            while cap.isOpened():
                # print("Virtual Canvas is Recording !")
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

            await websocket.send(            
                                json.dumps({"keypoints": keypoints})
                            )                 
            
                        
            cap.release()
            vcwriter.release()
            print("Processing Successful !")
        else:
        
            await asyncio.sleep(1)    
    



async def streamHandler(websocket):
    
    print("Stream Listener is listening !")    
    while True:
        await asyncio.sleep(3)
        # print("Waiting for Stream Listener to be activated !")
        if state['isCameraToggle']:
            print("Camera and Virtual Canvas is Activated !")
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)            
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))                    
            size = (frame_width, frame_height)       
            
            camwriter = None
            vcwriter = None
            
            # camwriter = cv2.VideoWriter(fr'{state["projectPath"]}\Videos\cam-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
            #                         cv2.VideoWriter_fourcc(*'MJPG'),
            #                         10, size)
            
            # vcwriter = cv2.VideoWriter(fr'{state["projectPath"]}\Videos\vc-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
            #                     cv2.VideoWriter_fourcc(*'MJPG'),
            #                     10, size)
            # try:
            while cap.isOpened() and state['isCameraToggle']:
                ret, frame = cap.read()
                
            
                if not ret or frame is None:
                    break
                # await websocket.send(            
                #             json.dumps({"image": frame.tolist()}))
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
                        
                virtualcanvas = np.zeros((frame_height, frame_width, 3), np.uint8)
                        # Rendering
                draw_connections(virtualcanvas, keypoints_with_scores, EDGES, 0.4)
                draw_keypoints(virtualcanvas, keypoints_with_scores, 0.4)

                if state['isRecording']:
                    if camwriter is None:
                        camwriter = cv2.VideoWriter(fr'{state["projectPath"]}\Videos\cam-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, size)
                        print("Camera is Recording !")
                    if vcwriter is None:
                        vcwriter = cv2.VideoWriter(fr'{state["projectPath"]}\Videos\vc-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
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
                        
                        # await websocket.send(data)
                await websocket.send(            
                            json.dumps({"camdata": camdata, "vcdata": vcdata, "keypoints": np.squeeze(np.multiply(keypoints_with_scores, [frame.shape[0], frame.shape[1] , 1])).tolist()})
                        )             
    
                        
                        # cv2.imshow("Transimission", frame)
                        
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
            # except websockets.exceptions.ConnectionClosed:
            #     print("Client Disconnected !")
                
            #     reset_state()
            #     for task in asyncio.all_tasks():
            #         print(task)
            #         task.cancel()
        
            print("Camera and Virtual Canvas is Deactivated !")
            cap.release()
            # camwriter.release()
            # vcwriter.release()
        
            

# async def videolistener(websocket):
#     while True:
#         if state['isCameraToggle'] == True:
#             await stream(websocket)

# async def stream(websocket):
#     try:
#         cap = cv2.VideoCapture(0)
            
#         frame_width = int(cap.get(3))
#         frame_height = int(cap.get(4))
                
#         size = (frame_width, frame_height)            
            
#         result = cv2.VideoWriter(f'output\{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
#                                         cv2.VideoWriter_fourcc(*'MJPG'),
#                                         10, size)
        
    
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 break
#             await websocket.send(            
#                         json.dumps({"image": frame}))
#             img = frame.copy()
#                     # img = tf.image.resize_with_pad(
#                     #     np.expand_dims(img, axis=0), 192, 192)
#             img = tf.image.resize_with_pad(
#                         np.expand_dims(img, axis=0), 256, 256)
#             input_image = tf.cast(img, dtype=tf.float32)

#                     # Setup input and output
#             input_details = interpreter.get_input_details()
#             output_details = interpreter.get_output_details()

#                     # Make predictions
#             interpreter.set_tensor(
#                         input_details[0]['index'], np.array(input_image))
#             interpreter.invoke()
#             keypoints_with_scores = interpreter.get_tensor(
#                         output_details[0]['index'])
                    
#             virtualcanvas = np.zeros((frame_height, frame_width, 3), np.uint8)
#                     # Rendering
#             draw_connections(virtualcanvas, keypoints_with_scores, EDGES, 0.4)
#             draw_keypoints(virtualcanvas, keypoints_with_scores, 0.4)

#             result.write(virtualcanvas)
                    
#             encoded = cv2.imencode('.jpg', virtualcanvas)[1]

#             data = str(base64.b64encode(encoded))
#             data = data[2:len(data)-1]
                    
#                     # await websocket.send(data)
#             await websocket.send(            
#                         json.dumps({"image": data, "keypoints": np.squeeze(np.multiply(keypoints_with_scores, [frame.shape[0], frame.shape[1] , 1])).tolist()})
#                     )
                    
#                     # cv2.imshow("Transimission", frame)
                    
#                     # if cv2.waitKey(1) & 0xFF == ord('q'):
#                     #     break
#         cap.release()
#         result.release()
        
#     except websockets.connection.ConnectionClosed as e:
#         # print(e)
#         cap.release()
#         result.release()
    

start_server = websockets.serve(main, port=port)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
asyncio.get_event_loop().set_debug()