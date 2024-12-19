import socket

# Define the server address and port
server_address = ('localhost', 5001)

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the address and port
sock.bind(server_address)

print(f"Server listening on {server_address[0]}:{server_address[1]}")

while True:
    # Receive data from the client
    data, client_address = sock.recvfrom(4096)
    if data:
        print(f"Received from {client_address}: {data.decode('utf-8')}")
