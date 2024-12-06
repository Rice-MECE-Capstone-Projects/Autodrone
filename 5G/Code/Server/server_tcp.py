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

def handle_client(client_socket, iterations):
    for i in range(iterations):
        print(f"\n--- Handling Iteration {i + 1}/{iterations} ---")

        try:
            start_time = time.time()

            file_path = f"received_image_{i}.jpg"
            print(f"Receiving pic for iteration {i + 1}...")
            with open(file_path, "wb") as file:
                while True:
                    data = client_socket.recv(1028)
                    if not data:
                        print("Connection closed by client.")
                        break
                    if data.endswith(b"EOF"):
                        file.write(data[:-3])
                        print("end of pic")
                        break
                    file.write(data)
            print(f"File saved as {file_path}.")

            args = ['--image_path', f"received_image_{i}.jpg",'--encode', 'vits']
            depthargs = parseDepth(args)

            #DispAnything.main(depthargs)

            response = f"File received successfully for iteration {i + 1}."
            client_socket.send(response.encode('utf-8'))
            print("Response sent to client")

            elapsed_time = time.time() - start_time
            print(f"Time taken for iteration {i+1}: {elapsed_time:.6f} seconds.")

        except Exception as e:
            print(f"Error during iteration {i+1}: {e}")
            break
    
    print("All iterations completed. Closing connection.")
    client_socket.close()

def server(server_ip, server_port, iterations):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(1)

    print(f"Server listening on {server_ip}:{server_port}...")

    try: 
        while True:
            print("\nWaiting for a client to connect...")
            client_socket, client_address = server_socket.accept()
            print(f"Connection accepted from {client_address}")

            handle_client(client_socket, iterations)
    
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server_socket.close()
        print("Server socket closed")

if __name__ == "__main__":
    #server('localhost', 16325, 3)
    #server('168.6.245.108', 16325, 5)
    server('168.5.45.117', 16325, 3)