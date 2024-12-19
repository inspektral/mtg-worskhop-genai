from pythonosc import udp_client

if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5001
    client = udp_client.SimpleUDPClient(ip, port)
    client.send_message("/hello", ["wowowow"])