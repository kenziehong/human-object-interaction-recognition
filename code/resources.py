"""Đo tài nguyên khi chạy: RAM đỉnh, thời gian CPU, dung lượng file, thông tin máy.

RAM đỉnh lấy bằng 1 luồng nền, cứ INTERVAL giây cộng RSS của tiến trình chính và mọi tiến trình con
(worker joblib/loky của Parallel, GridSearchCV). Cộng RSS sẽ đếm trùng phần bộ nhớ dùng chung (mảng memmap
của joblib) nên là cận trên; macOS nén bộ nhớ nên RSS có thể thấp hơn lượng đã cấp phát; đỉnh ngắn hơn
INTERVAL có thể bị bỏ lỡ.

Dùng:
    mon = ResourceMonitor().start()
    m = mon.mark()            # mốc bắt đầu một đoạn cần đo (lồng nhau được)
    ...
    mon.since(m)              # {"peak_rss_mb", "start_rss_mb", "peak_delta_mb", "cpu_s", "wall_s", "cpu_util"}
    mon.stop()

peak_rss_mb gồm cả phần nền: worker loky còn sống từ lần Parallel/GridSearchCV trước (mỗi worker đã nạp
numpy/sklearn, ~100-200 MB). peak_delta_mb = peak_rss_mb - RSS lúc đặt mốc: phần RAM tăng thêm do chính đoạn đó;
đoạn đầu tiên dùng Parallel/GridSearchCV còn tính cả RAM khởi động worker.
"""
import platform
import threading
import time
from importlib import metadata

import psutil

INTERVAL = 0.2
MB = 1e6


class _Mark:
    def __init__(self, rss, cpu):
        self.t = time.time()
        self.cpu0 = cpu
        self.start = rss
        self.peak = rss


class ResourceMonitor:
    def __init__(self, interval=INTERVAL):
        self.interval = interval
        self.proc = psutil.Process()
        self.cpu_seen = {}            # pid -> thời gian CPU (user + system) lần cuối thấy, giữ lại cả worker đã thoát
        self.marks = []
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        """Trả về (tổng RSS hiện tại, tổng thời gian CPU tích luỹ) của cả cây tiến trình."""
        rss = 0
        for p in [self.proc, *self.proc.children(recursive=True)]:
            try:
                with p.oneshot():
                    rss += p.memory_info().rss
                    c = p.cpu_times()
                    self.cpu_seen[p.pid] = c.user + c.system
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return rss, sum(self.cpu_seen.values())

    def _update(self):
        rss, cpu = self._sample()
        for m in self.marks:
            m.peak = max(m.peak, rss)
        return rss, cpu

    def _loop(self):
        while not self._stop.wait(self.interval):
            with self.lock:
                self._update()

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()

    def mark(self):
        with self.lock:
            m = _Mark(*self._update())
            self.marks.append(m)
        return m

    def since(self, m, drop=True):
        """Tài nguyên từ mốc m tới giờ. cpu_util = thời gian CPU / thời gian thực (≈ số nhân dùng trung bình)."""
        with self.lock:
            _, cpu = self._update()
            if drop and m in self.marks:
                self.marks.remove(m)
        wall = time.time() - m.t
        cpu_s = cpu - m.cpu0
        return {"peak_rss_mb": round(m.peak / MB, 1), "start_rss_mb": round(m.start / MB, 1),
                "peak_delta_mb": round((m.peak - m.start) / MB, 1), "cpu_s": round(cpu_s, 1), "wall_s": round(wall, 1),
                "cpu_util": round(cpu_s / wall, 2) if wall > 0 else None}


def file_mb(path):
    return round(path.stat().st_size / MB, 3) if path.exists() else None


def dir_mb(path, pattern="**/*"):
    return round(sum(f.stat().st_size for f in path.glob(pattern) if f.is_file()) / MB, 1)


def machine_info():
    vm = psutil.virtual_memory()
    pkgs = {}
    for name in ["numpy", "scikit-learn", "xgboost", "opencv-python", "psutil"]:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu_logical": psutil.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(vm.total / 1e9, 1), "python": platform.python_version(), "packages": pkgs}
