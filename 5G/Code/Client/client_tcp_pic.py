import socket
import time
#mport drone_take_pic as pic
import cv2

def client(server_ip, server_port, iterations):
    round_trip_times = []
    print("Opening camera")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, server_port))
        print("Connected to the server.")

        for i in range(iterations):
            print(f"\n--- Iteration {i + 1}/{iterations} ---")
            start_time = time.time()
            file_path = f"lower_res_image_{i}.jpg"
            #pic.takePic(file_path)
            
            #take pic
            file_path = f"lower_res_image_{i}.jpg"
            #pic.takePic(file_path)
            print("Capturing frame")
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
            print("Scaling frame")
            new_width = 640  # Set your desired width
            new_height = 480  # Set your desired height
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print("Writing frame")
            cv2.imwrite(file_path, resized_frame)

            try:
                print(f"Sending {file_path}...")
                with open(file_path, "rb") as file:
                    while (data:= file.read(1028)):
                        client_socket.send(data)
                print("sending EOF")
                client_socket.send(b"EOF")
                print(f"{file_path} sent successfully.")
            except Exception as e:
                print(f"Error during file transmission: {e}")
                round_trip_times.append(None)
                continue

            try:
                print("Waiting for response from server...")
                response = client_socket.recv(1028).decode('utf-8')
                print("Response received from server:", response)
            except Exception as e:
                print(f"Error during response reception: {e}")
                round_trip_times.append(None)
                continue

            round_trip_time = time.time() - start_time
            round_trip_times.append(round_trip_time)
            print(f"Round-trip time for iteration {i + 1}: {round_trip_time:.6f} seconds")
        
        print("\n--- Timing Statistics ---")
        valid_times = [t for t in round_trip_times if t is not None]
        for idx, rt_time in enumerate(round_trip_times, start=1):
            print(f"Iteration {idx}: {rt_time:.6f} seconds" if rt_time else f"Iteration {idx}: Failed")
        
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            print(f"\nAverage round-trip time: {avg_time:.6f} seconds.")
        else:
            print("No successful iterations.")    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        print("Connection closed.")
    print("Closing camera")
    cap.release()

if __name__ == "__main__":
    #client('localhost', 16325, 3)
    client('168.6.245.108', 16325, 5)
    #client('168.5.45.117', 16325, 3)
