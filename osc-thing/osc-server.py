from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client

def handle_message(address, *args):
    print(f"Received message at {address} with arguments {args}")
    return "Hello, World!"

if __name__ == "__main__":
    # Create a dispatcher to handle incoming messages
    disp = dispatcher.Dispatcher()
    disp.map("/*", handle_message)  # Use wildcard to match any address

    # Set up the server
    ip = "127.0.0.1"
    port = 5000
    server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
    print(f"Serving on {server.server_address}")

    # Run the server
    server.serve_forever()