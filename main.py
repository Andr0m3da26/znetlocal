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

record = False
port = 5000
print("Started server on port : ", port)

async def transmit(websocket, path):
    print("Client Connected !")    
    # result = await websocket.recv()
    # while not websocket.closed:
    #     print(result)
    #     result = json.loads(result)    
    #     asyncio.create_task(commandHandler(websocket, result))
    #     result = await websocket.recv()

    try:        
        while True:        
            result = await websocket.recv()
            print(result)
            result = json.loads(result)    
            asyncio.create_task(commandHandler(websocket, result))
    except websockets.exceptions.ConnectionClosed:
        print("Client Disconnected !")

async def commandHandler(websocket, result):
    if 'command' not in result:
        print("Expected command (string) in json")
        return None
    match result['command']:
        case "stream":
            if 'virtualcanvas' not in result:
                print("Expected virtualcanvas (bool) in json")
        
            if type(result['virtualcanvas']) != bool:
                print(f"virtualcanvas is of an invalid type {type(result['virtualcanvas'])} in json")     

            await stream(websocket, result['virtualcanvas'])
        case "video":
            if 'path' not in result:
                print("Expected path (string) in json")
            if type(result['path']) != str:
                print(f"path is of an invalid type {type(result['path'])} in json")    

            await video(websocket, result['path'])    
        case "record":
            if type(result['record']) != bool:
                print(f"record is of an invalid type {type(result['path'])} in json") 
            global record
            record = result['record'] 
            print(record)            
        case _:
            print("Invalid Command !")
    # if result['command'] == "stream":
    #     if 'virtualcanvas' not in result:
    #         print("Expected virtualcanvas (bool) in json")
       
    #     if type(result['virtualcanvas']) != bool:
    #         print(f"virtualcanvas is of an invalid type {type(result['virtualcanvas'])} in json")     

    #     await stream(websocket, result['virtualcanvas'])
        
        
    # elif result['command'] == "video":
    #     if 'path' not in result:
    #         print("Expected path (string) in json")
    #     if type(result['path']) != str:
    #         print(f"path is of an invalid type {type(result['path'])} in json")    

    #     await video(websocket, result['path'])          
      
    # else:
    #     print("Invalid Command !")

async def video(websocket, path):
    keypoints = []
    try:
        cap = cv2.VideoCapture(path)
            
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
                
        size = (frame_width, frame_height)            
            
        result = cv2.VideoWriter(f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
                                        cv2.VideoWriter_fourcc(*'MJPG'),
                                        10, size)
       

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

            result.write(virtualcanvas) 

        await websocket.send(            
                            json.dumps({"keypoints": keypoints})
                        )                 
            
                    
        cap.release()
        result.release()
    except Exception as e:
        cap.release()
        result.release()
        
    except websockets.connection.ConnectionClosed as e:
        # print(e)
        cap.release()
        result.release()

async def stream(websocket, virtualcanvas):
    try:
        cap = cv2.VideoCapture(0)
            
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
                
        size = (frame_width, frame_height)            
            
        result = cv2.VideoWriter(f'output\{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.avi', 
                                        cv2.VideoWriter_fourcc(*'MJPG'),
                                        10, size)
        
        if virtualcanvas == True:

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
                        
                virtualcanvas = np.zeros((frame_height, frame_width, 3), np.uint8)
                        # Rendering
                draw_connections(virtualcanvas, keypoints_with_scores, EDGES, 0.4)
                draw_keypoints(virtualcanvas, keypoints_with_scores, 0.4)

                result.write(virtualcanvas)
                        
                encoded = cv2.imencode('.jpg', virtualcanvas)[1]

                data = str(base64.b64encode(encoded))
                data = data[2:len(data)-1]
                        
                        # await websocket.send(data)
                await websocket.send(            
                            json.dumps({"image": data, "keypoints": np.squeeze(np.multiply(keypoints_with_scores, [frame.shape[0], frame.shape[1] , 1])).tolist()})
                        )
                        
                        # cv2.imshow("Transimission", frame)
                        
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
            cap.release()
            result.release()
        else:
            while cap.isOpened():
                _, frame = cap.read()
                result.write(frame)
            
                        
            cap.release()
            result.release()
    except websockets.connection.ConnectionClosed as e:
        # print(e)
        cap.release()
        result.release()
    

start_server = websockets.serve(transmit, port=port)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()