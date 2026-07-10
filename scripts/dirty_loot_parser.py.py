import json
import os
import time

def process_loot(folder_path):
    results = {}
    error_count = 0
    start = time.time()

    print("Starting loot processing...")

    if not os.path.exists(folder_path):
        print("Folder not found!")
        return None

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            try:
                # Blocking synchronous I/O
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # Unsafe dictionary access and no validation
                    ip = data.get('source_ip', 'unknown')
                    payload = data.get('payload_data', '')
                    timestamp = data.get('timestamp')
                    
                    is_flagged = False
                    
                    # Messy inline business logic
                    if 'eval(' in payload or '<script>' in payload or 'DROP TABLE' in payload:
                        is_flagged = True
                        
                    if ip not in results:
                        results[ip] = {'total_requests': 0, 'flagged_requests': 0, 'last_seen': None}
                        
                    results[ip]['total_requests'] += 1
                    
                    if is_flagged:
                        results[ip]['flagged_requests'] += 1
                        
                    results[ip]['last_seen'] = timestamp
                    
            except Exception as e:
                # Catch-all exception and basic print logging
                print(f"Error reading file {filename}: {e}")
                error_count += 1

    print("--- LOOT ANALYSIS COMPLETE ---")
    for ip, stats in results.items():
        if stats['flagged_requests'] > 0:
            print(f"WARNING: IP {ip} has {stats['flagged_requests']} flagged payloads out of {stats['total_requests']} total.")

    print(f"Processed in {time.time() - start} seconds with {error_count} errors.")
    return results

if __name__ == "__main__":
    process_loot("./captured_loot")