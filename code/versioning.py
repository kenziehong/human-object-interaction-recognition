"""Quản lý phiên bản kết quả: features/<version>/ và results/<version>/.

Quy tắc: không bao giờ xoá hay ghi đè một phiên bản đã có. Muốn bổ sung vào phiên bản cũ
(ví dụ chạy thêm vài cấu hình) thì phải truyền --update một cách chủ động.
"""
import json
import platform
import re
import sys
from datetime import datetime

import config as C

_VALID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*(\.[A-Za-z0-9_-]+)*$")   # không cho "..", "/"


def _check(version):
    if not _VALID.match(version):
        sys.exit(f"Tên phiên bản không hợp lệ: {version!r} (chỉ dùng chữ, số, _ . -; không chứa / hay ..)")
    return version


def feat_dir(version):
    return C.FEAT_ROOT / _check(version)


def result_dir(version):
    return C.RESULT_ROOT / _check(version)


def prepare_dir(path, marker, update=False):
    """Tạo thư mục phiên bản. Nếu file `marker` đã có mà không --update thì dừng (không ghi đè)."""
    if (path / marker).exists() and not update:
        sys.exit(f"Phiên bản {path} đã có {marker}. Không ghi đè kết quả cũ: hãy đặt tên phiên bản mới "
                 f"(ví dụ thêm hậu tố), hoặc truyền --update nếu thật sự muốn bổ sung vào phiên bản này.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_run_info(path, step, args, **extra):
    """Ghi/ghép thông tin lần chạy vào <path>/run_info.json (mỗi bước 1 khoá, lần --update được nối thêm)."""
    f = path / "run_info.json"
    info = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    entry = {"time": datetime.now().isoformat(timespec="seconds"),
             "command": " ".join([sys.argv[0].split("/")[-1], *sys.argv[1:]]),
             "args": vars(args), "machine": f"{platform.machine()} {platform.platform()}", **extra}
    info.setdefault(step, []).append(entry)
    f.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_run_info(path):
    f = path / "run_info.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def feature_config():
    """Ảnh chụp các tham số đặc trưng trong config.py, lưu kèm phiên bản để biết đã trích với tham số nào."""
    return {k: getattr(C, k) for k in dir(C)
            if k.isupper() and k not in {"ROOT", "DATA_DIR", "LABELS_CSV", "FEAT_ROOT", "RESULT_ROOT"}}
