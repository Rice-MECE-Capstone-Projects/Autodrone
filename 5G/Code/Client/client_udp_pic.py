import socket
import time
#import drone_take_pic as pic
import cv2

def udp_client(server_ip, port, file_path, iterations):
    """UDP client to send a file multiple times and measure round-trip time."""
    round_trip_times = []  # List to store round-trip times for each iteration
    print("Opening camera")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()


    for i in range(iterations):
        print(f"\n--- Iteration {i + 1}/{iterations} ---")

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
            try:
                # Start timing the round-trip
                start_time = time.time()

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

                # Sending file
                print("Sending file...")
                with open(file_path, "rb") as file:
                    while (data := file.read(1028)):
                        client_socket.sendto(data, (server_ip, port))
                client_socket.sendto(b"EOF", (server_ip, port))  # End of file marker
                print("File sent.")

                # Receiving response
                print("Waiting for response from server...")
                response, _ = client_socket.recvfrom(1028)
                print("Response received from server:", response.decode('utf-8'))

                # Calculate round-trip time
                round_trip_time = time.time() - start_time
                round_trip_times.append(round_trip_time)
                print(f"Round-trip time for iteration {i + 1}: {round_trip_time:.6f} seconds")

            except Exception as e:
                print(f"An error occurred during iteration {i + 1}: {e}")
                round_trip_times.append(None)  # Append None for failed iteration

    # Calculate and display statistics
    print("\n--- Timing Statistics ---")
    valid_times = [t for t in round_trip_times if t is not None]
    for idx, rt_time in enumerate(round_trip_times, start=1):
        print(f"Iteration {idx}: {rt_time:.6f} seconds" if rt_time else f"Iteration {idx}: Failed")
    
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        print(f"\nAverage round-trip time: {avg_time:.6f} seconds.")
    else:
        print("No successful iterations.")
    print("Closing camera")
    cap.release()

if __name__ == "__main__":
    # Replace with your server's IP address and file path
    udp_client('168.6.245.108', 16325, "lower_res_image_0.jpg", iterations=20)
    #udp_client('localhost', 16325, "ryon.jpeg", iterations=3)
    #udp_client('168.5.45.117', 16325, "lower_res_image_0.jpg", iterations=20)
