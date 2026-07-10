import socket
import time
import json
import os

# Global configuration (messy)
TIMEOUT = 2
VULN_DB = {
    "OpenSSH_7.2p2": "CVE-2016-6210",
    "Apache/2.4.49": "CVE-2021-41773",
    "nginx/1.14.0": "CVE-2018-16843"
}

def grab_banner(ip, port):
    # Blocking network I/O
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(TIMEOUT)
        s.connect((ip, port))
        
        # Send a dummy HTTP request if port is 80
        if port == 80:
            s.sendall(b"HEAD / HTTP/1.0\r\n\r\n")
            
        banner = s.recv(1024).decode('utf-8', errors='ignore').strip()
        s.close()
        return banner
    except Exception as e:
        # Catch-all exception, silent failure
        print(f"Failed connecting to {ip}:{port} - {e}")
        return None

def process_targets(filepath):
    print("Starting network sweep...")
    start_time = time.time()
    results = []

    if not os.path.exists(filepath):
        print("Target file missing!")
        return

    # Blocking file I/O
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        # Dangerous split, assuming perfect input
        parts = line.split(":")
        ip = parts[0]
        ports = [22, 80] if len(parts) == 1 else [int(parts[1])]

        for port in ports:
            print(f"Scanning {ip} on port {port}...")
            banner = grab_banner(ip, port)
            
            if banner:
                # Raw dictionary assembly
                hit = {
                    "target_ip": ip,
                    "target_port": port,
                    "raw_banner": banner[:50], # Truncate messy banners
                    "vulnerability": "None",
                    "flagged": False
                }
                
                # Inline business logic
                for known_sig, cve in VULN_DB.items():
                    if known_sig in banner:
                        hit["vulnerability"] = cve
                        hit["flagged"] = True
                        print(f"CRITICAL: {ip}:{port} matches {cve}!")
                
                results.append(hit)

    # Output generation
    with open("scan_results.json", 'w') as out:
        json.dump(results, out, indent=4)
        
    print(f"Sweep complete in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Expects a file with lines like "192.168.1.50" or "10.0.0.5:8080"
    process_targets("targets.txt")