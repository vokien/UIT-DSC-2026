import os
import glob
import json
import numpy as np

class CheckpointManager:
    def __init__(self, base_dir="checkpoints"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_task_dir(self, task_name: str) -> str:
        task_dir = os.path.join(self.base_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        return task_dir

    def _cleanup_old_checkpoints(self, task_dir: str, prefix: str, keep_last: int = 2):
        json_pattern = os.path.join(task_dir, f"{prefix}_*.json")
        npy_pattern = os.path.join(task_dir, f"{prefix}_*.npy")
        
        all_files = glob.glob(json_pattern) + glob.glob(npy_pattern)
        
        # Sắp xếp file theo thời gian sửa đổi cũ nhất -> mới nhất
        all_files.sort(key=os.path.getmtime)
        
        # Nếu vượt quá số lượng cho phép, xóa các file cũ
        if len(all_files) > keep_last:
            files_to_delete = all_files[:-keep_last]
            for f in files_to_delete:
                try:
                    os.remove(f)
                    print(f"[Dọn dẹp] Đã xóa checkpoint cũ: {f}")
                except Exception as e:
                    print(f"Lỗi khi xóa file checkpoint cũ {f}: {e}")

    def save_checkpoint(self, task_name: str, prefix: str, version: int, data, is_numpy: bool = False, keep_last: int = 2):
        task_dir = self._get_task_dir(task_name)
        ext = ".npy" if is_numpy else ".json"
        
        filename = f"{prefix}_v{version}{ext}"
        filepath = os.path.join(task_dir, filename)
        
        if is_numpy:
            np.save(filepath, data)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
                
        self._cleanup_old_checkpoints(task_dir, prefix, keep_last)
        return filepath

    def load_latest_checkpoint(self, task_name: str, prefix: str, is_numpy: bool = False):
        task_dir = self._get_task_dir(task_name)
        ext = ".npy" if is_numpy else ".json"
        pattern = os.path.join(task_dir, f"{prefix}_*{ext}")
        files = glob.glob(pattern)
        
        if not files:
            return None, 0
            
        # Sắp xếp để lấy file mới nhất
        files.sort(key=os.path.getmtime)
        latest_file = files[-1]
        
        print(f"\n[!] Phát hiện checkpoint mới nhất: {latest_file}. Đang load...")
        
        # Trích xuất số version
        version = 0
        try:
            basename = os.path.basename(latest_file)
            name_part = basename.replace(ext, "")
            version_str = name_part.split("_v")[-1]
            version = int(version_str)
        except:
            pass
            
        if is_numpy:
            data = np.load(latest_file)
        else:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
        return data, version
