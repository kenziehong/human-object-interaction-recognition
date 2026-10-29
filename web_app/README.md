# web_app: demo web nhận diện hành động người–vật

Web chạy cục bộ, 3 bước: **Tải ảnh lên → Xử lý ảnh → Dự đoán**.

```
ảnh gốc ──DP-HOI──► khung người + vật ──cắt crop──► CLIP ViT-B/32 + SVM ──► 1 trong 6 lớp
```

6 lớp: `hold`, `ride`, `eat/drink`, `talk_on_phone`, `work_on_computer`, `unknown`.

## Cấu trúc

```
web_infer/
├── app.py               server (thư viện chuẩn http.server) + logic cắt crop
├── dphoi_detector.py    dựng DETR của DP-HOI, chỉ dùng nhánh detection, chạy được CPU / Apple MPS
├── clip_svm.py          embedding CLIP + SVM (tiền xử lý giống lúc train)
├── static/index.html    giao diện
├── weights/
│   ├── clip_svm.joblib              mô hình phân loại (có sẵn trong repo, 23 MB)
│   └── dphoi_res50_3layers.pth      checkpoint DP-HOI (tải bằng setup.sh, không đưa lên repo)
├── third_party/DP-HOI/  mã nguồn DP-HOI (clone bằng setup.sh, không đưa lên repo)
├── scripts/setup.sh
└── requirements.txt
```

## Cài đặt và chạy

```bash
cd web_infer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup.sh        # clone DP-HOI + tải checkpoint ResNet50 (161 MB, Google Drive)
python app.py                # mở http://localhost:8000   (--port 8080, --cpu)
```

`setup.sh` làm 2 việc (có thể tự làm tay):

1. `git clone https://github.com/xingaoli/DP-HOI.git third_party/DP-HOI`
2. Tải [checkpoint pre-train DP-HOI ResNet50](https://drive.google.com/file/d/1DYBY8wHDGDPh9pyvxBewTqmpbztAzazb/view) về `weights/dphoi_res50_3layers.pth`

Không cần cài `clip` của OpenAI mà DP-HOI yêu cầu: nhánh caption chỉ dùng lúc pre-train, `dphoi_detector.py` thay module đó bằng module rỗng.

## Cách cắt crop

1. DP-HOI tìm người và vật (80 lớp COCO), giữ khung có độ tin cậy ≥ 0.7.
2. Vật **chạm** khung người thì thuộc về người đó; chạm nhiều người thì thuộc về người có tâm gần vật nhất. Bỏ vật:
   - to hơn 4 lần người (bàn, giường…): cảnh nền;
   - có khung chứa tâm của người khác: vật dùng chung, hoặc detector gộp nhiều vật thành 1 khung.
3. Khung cắt:
   - **có vật**: khung nhỏ nhất bao người + vật của người đó;
   - **chỉ có người**: khung người nới 20% mỗi phía (giống `data_preparation/tao_dataset_6class.py`), để lấy được vật detector bỏ sót.

## Nguồn

- DP-HOI: Li et al., *Disentangled Pre-training for Human-Object Interaction Detection*, CVPR 2024, <https://github.com/xingaoli/DP-HOI>.
- CLIP ViT-B/32 qua [open_clip](https://github.com/mlfoundations/open_clip) (trọng số `openai`). Dữ liệu train của CLIP không công bố và có thể chứa ảnh COCO, nên kết quả CLIP + SVM chỉ là mốc so sánh.
