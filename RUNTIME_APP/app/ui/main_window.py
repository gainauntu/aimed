from __future__ import annotations
import uuid
from pathlib import Path
import requests
from PySide6 import QtWidgets, QtGui, QtCore

from app.ui.pin_dialog import PinDialog
from app.ui.demo_gate import popup_future, popup_error

DEFAULT_URL = "http://127.0.0.1:9000"

DARK_QSS = """
QMainWindow { background: #1f1f1f; }
QWidget { color: #e6e6e6; font-size: 12px; }
QTabWidget::pane { border: 1px solid #2c2c2c; background: #232323; border-radius: 8px; }
QTabBar::tab { background: #2a2a2a; padding: 8px 14px; margin-right: 6px; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:selected { background: #353535; font-weight: 700; }
QLineEdit, QPlainTextEdit, QTextEdit { background: #1c1c1c; border: 1px solid #3a3a3a; border-radius: 8px; padding: 6px; }
QGroupBox { border: 1px solid #3a3a3a; border-radius: 10px; margin-top: 10px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #cfcfcf; }
QPushButton { background: #3b3b3b; border: 1px solid #4a4a4a; border-radius: 10px; padding: 8px 12px; }
QPushButton:hover { background: #474747; }
QPushButton:disabled { background: #2b2b2b; color:#8a8a8a; border: 1px solid #333; }
QTableWidget { background: #1c1c1c; border: 1px solid #3a3a3a; border-radius: 10px; gridline-color: #333; }
QHeaderView::section { background: #2a2a2a; border: none; padding: 6px; font-weight: 700; }
QComboBox { background: #1c1c1c; border: 1px solid #3a3a3a; border-radius: 8px; padding: 6px; }
QSpinBox, QDoubleSpinBox { background: #1c1c1c; border: 1px solid #3a3a3a; border-radius: 8px; padding: 6px; }
QCheckBox::indicator { width: 18px; height: 18px; }
QProgressBar { border: 1px solid #3a3a3a; border-radius: 8px; text-align: right; padding: 2px; background: #1c1c1c; }
QProgressBar::chunk { background: #4c8bf5; border-radius: 8px; }
"""

def _pix(path: str, mw=520, mh=360):
    p = QtGui.QPixmap(path)
    if p.isNull():
        return p
    return p.scaled(mw, mh, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)

def json_dumps_pretty(obj):
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("알약 식별 AI 도구")
        self.resize(1400, 900)

        self._a_path = ""
        self._b_path = ""
        self._dataset_root = ""

        self._log_buffer = []

        # demo gating
        self.demo_mode = True
        self.admin_pin = "1234"

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self.setStyleSheet(DARK_QSS)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        self.tab_data = QtWidgets.QWidget()
        self.tab_model = QtWidgets.QWidget()
        self.tab_test = QtWidgets.QWidget()
        self.tab_settings = QtWidgets.QWidget()

        self.tabs.addTab(self.tab_data, "데이터")
        self.tabs.addTab(self.tab_model, "모델 관리")
        self.tabs.addTab(self.tab_test, "간단 테스트")
        self.tabs.addTab(self.tab_settings, "설정")

        self._build_data_tab()
        self._build_model_tab()
        self._build_test_tab()
        self._build_settings_tab()

        # bottom area
        bottom = QtWidgets.QVBoxLayout()
        root.addLayout(bottom)
        pbar_row = QtWidgets.QHBoxLayout()
        bottom.addLayout(pbar_row)
        self.pbar = QtWidgets.QProgressBar()
        self.pbar.setValue(0)
        self.lbl_pbar = QtWidgets.QLabel("0%  준비됨")
        pbar_row.addWidget(self.pbar, 1)
        pbar_row.addWidget(self.lbl_pbar)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setMinimumHeight(150)
        bottom.addWidget(self.log)

        self._log("[앱 시작]")

    def _log(self, s: str):
        # During startup, tabs may log before the bottom log widget is created.
        if not hasattr(self, "log") or self.log is None:
            self._log_buffer.append(s)
            return
        self.log.appendPlainText(s)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _require_pin_then(self, action_fn):
        dlg = PinDialog(self, title="관리자 PIN", prompt="해당 작업을 위해 관리자 PIN을 입력하세요.")
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        if dlg.pin() != self.admin_pin:
            popup_error(self, "PIN이 올바르지 않습니다.")
            return
        action_fn()

    # ---------------- 데이터 탭 ----------------
    def _build_data_tab(self):
        lay = QtWidgets.QVBoxLayout(self.tab_data)

        row = QtWidgets.QHBoxLayout()
        lay.addLayout(row)
        self.ed_dataset = QtWidgets.QLineEdit()
        self.ed_dataset.setPlaceholderText("데이터셋 폴더 경로")
        row.addWidget(self.ed_dataset, 1)

        self.btn_dataset_pick = QtWidgets.QPushButton("폴더 선택")
        self.btn_scan = QtWidgets.QPushButton("스캔")
        self.btn_model_update = QtWidgets.QPushButton("모델 업데이트(학습/적용)")
        row.addWidget(self.btn_dataset_pick)
        row.addWidget(self.btn_scan)
        row.addWidget(self.btn_model_update)

        split = QtWidgets.QHBoxLayout()
        lay.addLayout(split, 1)

        left_box = QtWidgets.QGroupBox("약 목록(클래스) — 더블클릭: 이름변경")
        left_v = QtWidgets.QVBoxLayout(left_box)
        self.tbl_classes = QtWidgets.QTableWidget(0, 2)
        self.tbl_classes.setHorizontalHeaderLabels(["약 이름", "쌍 개수(UP/DOWN)"])
        self.tbl_classes.horizontalHeader().setStretchLastSection(True)
        self.tbl_classes.verticalHeader().setVisible(False)
        self.tbl_classes.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_classes.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        left_v.addWidget(self.tbl_classes, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_class = QtWidgets.QPushButton("새 약 추가(폴더 가져오기)")
        self.btn_del_class = QtWidgets.QPushButton("선택 약 삭제")
        btn_row.addWidget(self.btn_add_class)
        btn_row.addWidget(self.btn_del_class)
        left_v.addLayout(btn_row)

        split.addWidget(left_box, 1)

        right_box = QtWidgets.QGroupBox("선택 약 데이터(미리보기)")
        right_v = QtWidgets.QVBoxLayout(right_box)

        prev_row = QtWidgets.QHBoxLayout()
        right_v.addLayout(prev_row)

        prevA = QtWidgets.QVBoxLayout()
        prevA.addWidget(QtWidgets.QLabel("앞면(UP)"))
        self.lbl_up = QtWidgets.QLabel()
        self.lbl_up.setMinimumHeight(340)
        self.lbl_up.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_up.setStyleSheet("border: 1px solid #3a3a3a; border-radius: 10px;")
        prevA.addWidget(self.lbl_up, 1)

        prevB = QtWidgets.QVBoxLayout()
        prevB.addWidget(QtWidgets.QLabel("뒷면(DOWN)"))
        self.lbl_dn = QtWidgets.QLabel()
        self.lbl_dn.setMinimumHeight(340)
        self.lbl_dn.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_dn.setStyleSheet("border: 1px solid #3a3a3a; border-radius: 10px;")
        prevB.addWidget(self.lbl_dn, 1)

        prev_row.addLayout(prevA, 1)
        prev_row.addSpacing(10)
        prev_row.addLayout(prevB, 1)

        pair_row = QtWidgets.QHBoxLayout()
        right_v.addLayout(pair_row)
        self.tbl_pairs = QtWidgets.QTableWidget(0, 2)
        self.tbl_pairs.setHorizontalHeaderLabels(["UP 파일", "DOWN 파일"])
        self.tbl_pairs.horizontalHeader().setStretchLastSection(True)
        self.tbl_pairs.verticalHeader().setVisible(False)
        self.tbl_pairs.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_pairs.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        pair_row.addWidget(self.tbl_pairs, 1)

        side_btns = QtWidgets.QVBoxLayout()
        self.btn_del_pair = QtWidgets.QPushButton("선택 쌍 삭제")
        self.btn_open_folder = QtWidgets.QPushButton("폴더 열기")
        side_btns.addWidget(self.btn_del_pair)
        side_btns.addWidget(self.btn_open_folder)
        side_btns.addStretch(1)
        pair_row.addLayout(side_btns)

        split.addWidget(right_box, 2)

        self.btn_dataset_pick.clicked.connect(self._pick_dataset)
        self.btn_scan.clicked.connect(lambda: popup_future(self))
        self.btn_model_update.clicked.connect(lambda: self._require_pin_then(lambda: popup_future(self)))
        self.btn_add_class.clicked.connect(lambda: popup_future(self))
        self.btn_del_class.clicked.connect(lambda: popup_future(self))
        self.btn_del_pair.clicked.connect(lambda: popup_future(self))
        self.btn_open_folder.clicked.connect(lambda: popup_future(self))

        self._fill_demo_dataset()
        self._log("[미리보기] 데이터 탭 UI 로드 완료")

    def _fill_demo_dataset(self):
        demo = [
            ("C 55", 20), ("C51", 20), ("DSC L60", 20), ("E228", 20), ("EL", 20),
            ("F하양이", 26), ("H90", 20), ("MI 5", 23), ("MMA", 14), ("SHP 50", 20),
            ("SP-LOX", 26), ("SPP RM", 24), ("SPT 10", 20),
        ]
        self.tbl_classes.setRowCount(len(demo))
        for r,(name,cnt) in enumerate(demo):
            self.tbl_classes.setItem(r,0,QtWidgets.QTableWidgetItem(name))
            self.tbl_classes.setItem(r,1,QtWidgets.QTableWidgetItem(str(cnt)))

        pairs = [("U_TEST1.png","D_TEST1.png"),("U_TEST10.png","D_TEST10.png"),("U_TEST11.png","D_TEST11.png")]
        self.tbl_pairs.setRowCount(len(pairs))
        for r,(u,d) in enumerate(pairs):
            self.tbl_pairs.setItem(r,0,QtWidgets.QTableWidgetItem(u))
            self.tbl_pairs.setItem(r,1,QtWidgets.QTableWidgetItem(d))

    def _pick_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "데이터셋 폴더 선택", self._dataset_root or str(Path.home()))
        if not path:
            return
        self._dataset_root = path
        self.ed_dataset.setText(path)
        self._log(f"[데이터] 데이터셋 폴더 선택: {path}")

    # ---------------- 모델 관리 탭 ----------------
    def _build_model_tab(self):
        lay = QtWidgets.QVBoxLayout(self.tab_model)
        row = QtWidgets.QHBoxLayout()
        lay.addLayout(row)
        self.ed_models_dir = QtWidgets.QLineEdit("models/versions")
        row.addWidget(self.ed_models_dir, 1)
        self.btn_models_pick = QtWidgets.QPushButton("폴더 선택")
        self.btn_models_refresh = QtWidgets.QPushButton("새로고침")
        self.btn_apply_model = QtWidgets.QPushButton("선택 모델 적용")
        row.addWidget(self.btn_models_pick)
        row.addWidget(self.btn_models_refresh)
        row.addWidget(self.btn_apply_model)

        main = QtWidgets.QHBoxLayout()
        lay.addLayout(main, 1)

        left = QtWidgets.QGroupBox("모델 버전 목록")
        lv = QtWidgets.QVBoxLayout(left)
        self.list_versions = QtWidgets.QListWidget()
        lv.addWidget(self.list_versions, 1)
        self.btn_del_model = QtWidgets.QPushButton("선택 모델 삭제")
        lv.addWidget(self.btn_del_model)

        right = QtWidgets.QVBoxLayout()
        info = QtWidgets.QGroupBox("모델 정보")
        iv = QtWidgets.QVBoxLayout(info)
        self.txt_model_info = QtWidgets.QPlainTextEdit()
        self.txt_model_info.setReadOnly(True)
        iv.addWidget(self.txt_model_info, 1)

        summary = QtWidgets.QGroupBox("포함 클래스(요약)")
        sv = QtWidgets.QVBoxLayout(summary)
        self.tbl_model_classes = QtWidgets.QTableWidget(0,2)
        self.tbl_model_classes.setHorizontalHeaderLabels(["약 이름","쌍 수"])
        self.tbl_model_classes.horizontalHeader().setStretchLastSection(True)
        self.tbl_model_classes.verticalHeader().setVisible(False)
        self.tbl_model_classes.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        sv.addWidget(self.tbl_model_classes, 1)

        right.addWidget(info, 1)
        right.addWidget(summary, 1)

        main.addWidget(left, 1)
        main.addLayout(right, 2)

        self.btn_models_pick.clicked.connect(lambda: popup_future(self))
        self.btn_models_refresh.clicked.connect(self._fill_demo_model_info)
        self.btn_apply_model.clicked.connect(lambda: self._require_pin_then(lambda: popup_future(self)))
        self.btn_del_model.clicked.connect(lambda: self._require_pin_then(lambda: popup_future(self)))

        self._fill_demo_model_info()
        self._log("[미리보기] 모델 관리 탭 UI 로드 완료")

    def _fill_demo_model_info(self):
        self.list_versions.clear()
        self.list_versions.addItem("v_20260124_194732")
        self.list_versions.setCurrentRow(0)
        info = {
            "version": "v_20260124_194732",
            "created_at": "2026-01-24 19:47:32",
            "dataset_root": r"C:\Users\...",
            "classes": [{"name":"C 55","pairs":20},{"name":"C51","pairs":20}]
        }
        self.txt_model_info.setPlainText(json_dumps_pretty(info))
        demo = [("C 55",20),("C51",20),("DSC L60",20),("E228",20),("EL",20),("F하양이",26)]
        self.tbl_model_classes.setRowCount(len(demo))
        for i,(n,c) in enumerate(demo):
            self.tbl_model_classes.setItem(i,0,QtWidgets.QTableWidgetItem(n))
            self.tbl_model_classes.setItem(i,1,QtWidgets.QTableWidgetItem(str(c)))

    # ---------------- 설정 탭 ----------------
    def _build_settings_tab(self):
        lay = QtWidgets.QVBoxLayout(self.tab_settings)
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)

        self.sp_topk = QtWidgets.QSpinBox(); self.sp_topk.setRange(1,10); self.sp_topk.setValue(5)
        self.sp_margin = QtWidgets.QDoubleSpinBox(); self.sp_margin.setDecimals(3); self.sp_margin.setRange(0.0, 1.0); self.sp_margin.setValue(0.050)
        self.sp_patch = QtWidgets.QDoubleSpinBox(); self.sp_patch.setDecimals(3); self.sp_patch.setRange(0.0, 1.0); self.sp_patch.setValue(0.700)

        self.chk_ocr = QtWidgets.QCheckBox("OCR 사용")
        self.chk_ocr.setChecked(True)

        self.cmb_device = QtWidgets.QComboBox()
        self.cmb_device.addItems(["자동(권장)", "DirectML", "CPU"])

        form.addRow("후보 Top-K", self.sp_topk)
        form.addRow("확신 여유(margin)", self.sp_margin)
        form.addRow("패치 검증 임계값", self.sp_patch)
        form.addRow("", self.chk_ocr)
        form.addRow("추론 장치", self.cmb_device)

        btns = QtWidgets.QHBoxLayout()
        lay.addLayout(btns)
        self.btn_adv = QtWidgets.QPushButton("고급 설정")
        self.btn_save = QtWidgets.QPushButton("설정 저장")
        self.btn_default = QtWidgets.QPushButton("기본값")
        btns.addWidget(self.btn_adv)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_default)
        btns.addStretch(1)

        space = QtWidgets.QFrame()
        space.setStyleSheet("border: 1px solid #2c2c2c; border-radius: 10px; background: #202020;")
        lay.addWidget(space, 1)

        self.btn_adv.clicked.connect(lambda: self._require_pin_then(lambda: popup_future(self)))
        self.btn_save.clicked.connect(lambda: popup_future(self))
        self.btn_default.clicked.connect(lambda: popup_future(self))
        self._log("[미리보기] 설정 탭 UI 로드 완료")

    # ---------------- 간단 테스트 탭 ----------------
    def _build_test_tab(self):
        lay = QtWidgets.QVBoxLayout(self.tab_test)

        top = QtWidgets.QHBoxLayout()
        lay.addLayout(top)
        top.addWidget(QtWidgets.QLabel("서버 URL:"))
        self.ed_url = QtWidgets.QLineEdit(DEFAULT_URL)
        self.ed_url.setMinimumWidth(280)
        top.addWidget(self.ed_url)
        top.addWidget(QtWidgets.QLabel("timeout(s):"))
        self.sp_to = QtWidgets.QSpinBox(); self.sp_to.setRange(1,180); self.sp_to.setValue(60)
        top.addWidget(self.sp_to)
        top.addStretch(1)
        self.btn_run = QtWidgets.QPushButton("판정 실행")
        top.addWidget(self.btn_run)

        pick = QtWidgets.QHBoxLayout()
        lay.addLayout(pick)
        pick.addWidget(QtWidgets.QLabel("앞면(UP)"))
        self.btn_pick_a = QtWidgets.QPushButton("파일 선택")
        self.ed_a = QtWidgets.QLineEdit(); self.ed_a.setReadOnly(True)
        pick.addWidget(self.btn_pick_a)
        pick.addWidget(self.ed_a, 1)

        pick.addSpacing(8)
        pick.addWidget(QtWidgets.QLabel("뒷면(DOWN)"))
        self.btn_pick_b = QtWidgets.QPushButton("파일 선택")
        self.ed_b = QtWidgets.QLineEdit(); self.ed_b.setReadOnly(True)
        pick.addWidget(self.btn_pick_b)
        pick.addWidget(self.ed_b, 1)

        prev = QtWidgets.QHBoxLayout()
        lay.addLayout(prev, 1)
        self.lbl_a_prev = QtWidgets.QLabel()
        self.lbl_b_prev = QtWidgets.QLabel()
        for lb in (self.lbl_a_prev, self.lbl_b_prev):
            lb.setMinimumHeight(420)
            lb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lb.setStyleSheet("border: 1px solid #3a3a3a; border-radius: 10px;")
        prev.addWidget(self.lbl_a_prev, 1)
        prev.addWidget(self.lbl_b_prev, 1)

        res = QtWidgets.QHBoxLayout()
        lay.addLayout(res)

        self.lbl_result_big = QtWidgets.QLabel("OK: -")
        f = self.lbl_result_big.font(); f.setPointSize(24); f.setBold(True); self.lbl_result_big.setFont(f)
        self.lbl_result_big.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_result_big.setMinimumHeight(120)
        self.lbl_result_big.setStyleSheet("border: 1px solid #3a3a3a; border-radius: 10px; background:#1c1c1c;")
        res.addWidget(self.lbl_result_big, 1)

        self.tbl_top = QtWidgets.QTableWidget(3, 2)
        self.tbl_top.setHorizontalHeaderLabels(["클래스", "점수(%)"])
        self.tbl_top.verticalHeader().setVisible(False)
        self.tbl_top.horizontalHeader().setStretchLastSection(True)
        self.tbl_top.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        res.addWidget(self.tbl_top, 1)

        self.btn_pick_a.clicked.connect(lambda: self._pick_img("A"))
        self.btn_pick_b.clicked.connect(lambda: self._pick_img("B"))
        self.btn_run.clicked.connect(self._run_predict)
        self._log("[미리보기] 간단 테스트 탭 UI 로드 완료")

    def _pick_img(self, which: str):
        label = "앞면(UP)" if which == "A" else "뒷면(DOWN)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"{label} 이미지 선택", str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All (*.*)"
        )
        if not path:
            return
        if which == "A":
            self._a_path = path
            self.ed_a.setText(path)
            self.lbl_a_prev.setPixmap(_pix(path))
        else:
            self._b_path = path
            self.ed_b.setText(path)
            self.lbl_b_prev.setPixmap(_pix(path))
        self._log(f"[테스트] {label} 선택: {path}")

    def _run_predict(self):
        if not self._a_path or not self._b_path:
            popup_error(self, "이미지 2개를 모두 선택하세요.")
            return
        url = self.ed_url.text().strip().rstrip("/")
        to = int(self.sp_to.value())
        req = {"request_id": str(uuid.uuid4()), "side_a_path": self._a_path, "side_b_path": self._b_path}
        self._log(f"POST {url}/predict (timeout={to}s)")
        try:
            r = requests.post(url + "/predict", json=req, timeout=(5, to))
            data = r.json()
        except Exception as e:
            popup_error(self, str(e))
            self._log(f"[ERROR] {e}")
            return

        st = data.get("status")
        if st == "OK":
            name = data.get("class_name","")
            conf = float(data.get("confidence_percent",0.0))
            self.lbl_result_big.setText(f"OK: {name} ({conf:.1f}%)")
        elif st == "UNDECIDED":
            top3 = data.get("top3") or []
            if top3:
                name = top3[0].get("name","")
                conf = float(top3[0].get("confidence",0.0))
                self.lbl_result_big.setText(f"미확정: {name} ({conf:.1f}%)")
            else:
                self.lbl_result_big.setText("미확정")
        else:
            self.lbl_result_big.setText("오류")

        top3 = data.get("top3") or []
        for i in range(3):
            if i < len(top3):
                self.tbl_top.setItem(i,0,QtWidgets.QTableWidgetItem(str(top3[i].get("name",""))))
                self.tbl_top.setItem(i,1,QtWidgets.QTableWidgetItem(f"{float(top3[i].get('confidence',0.0)):.1f}%"))
            else:
                self.tbl_top.setItem(i,0,QtWidgets.QTableWidgetItem(""))
                self.tbl_top.setItem(i,1,QtWidgets.QTableWidgetItem(""))

        self._log(f"RESP status={st} reason={data.get('reason','')}")
