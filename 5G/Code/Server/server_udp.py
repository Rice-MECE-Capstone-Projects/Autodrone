import socket
import time
#import Depth_Estimator_594.DispAnything as DispAnything
import argparse

def parseDepth(args):
    parser = argparse.ArgumentParser(description="Depth map inference and visualization.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'], help="Model encoder type.")

    depthargs = parser.parse_args(args)
    return depthargs

def udp_server(server_ip, port, iterations, save_path="received_file.jpg"):
    """UDP server to receive a file and send a response for a fixed number of iterations."""
    print(f"Server starting on {server_ip}:{port}, set for {iterations} iterations...")
    
    iteration_count = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_socket:
        server_socket.bind((server_ip, port))

        while iteration_count < iterations:
            print(f"\n--- Iteration {iteration_count + 1}/{iterations} ---")
            try:
                # Receive file data
                save_path = f"recv_lower_res_image_{iteration_count}.jpg"
                with open(save_path, "wb") as file:
                    print("Waiting for file...")
                    while True:
                        data, client_addr = server_socket.recvfrom(1028)
                        if data.endswith(b"EOF"):  # End of file marker
                            file.write(data[:-3])
                            print("end of pic")
                            print(f"File reception complete from {client_addr}.")
                            break
                        file.write(data)
                
                args = ['--image_path', f"recv_lower_res_image_{iteration_count}.jpg",'--encode', 'vits']
                depthargs = parseDepth(args)

                #DispAnything.main(depthargs)

                # Send response
                response = "File received successfully!"
                server_socket.sendto(response.encode('utf-8'), client_addr)
                print(f"Response sent to {client_addr}.")

                # Increment iteration count
                iteration_count += 1
            except Exception as e:
                print(f"Error during iteration {iteration_count + 1}: {e}")

    print("\n--- Server shut down after completing all iterations ---")

if __name__ == "__main__":
    # Replace with your server's IP address
    #udp_server('localhost', 16325, iterations=3)
    #udp_server('168.6.245.108', 16325, iterations=10)
    udp_server('168.5.45.117', 16325, iterations=20)