import os

def split_file(file_path, chunk_size=90*1024*1024): # 90MB chunks
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        part_num = 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            part_name = f"{file_path}.{part_num:03d}"
            with open(part_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            
            print(f"Created {part_name}")
            part_num += 1

if __name__ == "__main__":
    split_file("chroma_db.zip")
