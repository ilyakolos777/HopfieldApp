import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QTabWidget, QTextBrowser,
    QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea
)

from patterns import (
    GRID_SIZE, N_NEURONS, NOISE_LEVELS,
    get_bipolar_patterns, apply_noise, hamming_distance
)
from hopfield import HopfieldNetwork


class PixelGridWidget(QWidget):
    def __init__(self, grid_size: int = GRID_SIZE, editable: bool = True, cell_px: int = 28):
        super().__init__()
        self.grid_size = grid_size
        self.editable = editable
        self.state = np.full(grid_size * grid_size, -1, dtype=int)
        self.on_change_callback = None
        self._draw_val = 1

        total_px = grid_size * cell_px + 2
        self.setFixedSize(total_px, total_px)

    def set_state(self, vector: np.ndarray) -> None:
        self.state = vector.copy().flatten()
        self.update()

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def _toggle_at_pos(self, x: int, y: int, first_click: bool = False) -> None:
        if not self.editable:
            return
        cell_w = self.width() / self.grid_size
        cell_h = self.height() / self.grid_size
        col = int(x // cell_w)
        row = int(y // cell_h)
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            idx = row * self.grid_size + col
            if first_click:
                self._draw_val = -1 if self.state[idx] == 1 else 1
            if self.state[idx] != self._draw_val:
                self.state[idx] = self._draw_val
                self.update()
                if self.on_change_callback:
                    self.on_change_callback()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._toggle_at_pos(int(pos.x()), int(pos.y()), first_click=True)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            pos = event.position()
            self._toggle_at_pos(int(pos.x()), int(pos.y()), first_click=False)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = self.width()
        h = self.height()
        cell_w = w / self.grid_size
        cell_h = h / self.grid_size

        active_brush = QBrush(QColor("#3b82f6"))
        inactive_brush = QBrush(QColor("#121216"))
        grid_pen = QPen(QColor("#27272a"), 1)

        painter.setPen(grid_pen)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = self.state[r * self.grid_size + c]
                rect = QRectF(c * cell_w, r * cell_h, cell_w, cell_h)
                painter.fillRect(rect, active_brush if val == 1 else inactive_brush)
                painter.drawRect(rect)


class HopfieldApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Нейронная сеть Хопфилда — Распознавание образов (К, О, Л)")
        self.resize(1420, 840)

        self.patterns = get_bipolar_patterns()
        self.network = HopfieldNetwork(n_neurons=N_NEURONS)
        self.network.fit(self.patterns)

        self.current_base_label = "К"
        self.current_noise_percent = 0
        self.rng = np.random.default_rng(42)

        self._setup_styles()
        self._build_ui()

        self._load_reference("К")
        self._run_statistical_study()
        self._update_samples_view()

    def _setup_styles(self) -> None:
        pg.setConfigOption("background", "#0f0f11")
        pg.setConfigOption("foreground", "#a1a1aa")
        pg.setConfigOptions(antialias=True)

        self.setStyleSheet("""
            QMainWindow { background-color: #09090b; }
            QLabel { color: #e4e4e7; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
            QLabel#title { font-size: 15px; font-weight: bold; color: #60a5fa; }
            QLabel#subtitle { font-size: 12px; color: #a1a1aa; }
            QFrame#card { background-color: #18181b; border-radius: 8px; border: 1px solid #27272a; padding: 10px; }
            QPushButton {
                background-color: #27272a; color: #f4f4f5; border: 1px solid #3f3f46;
                border-radius: 5px; padding: 7px 12px; font-weight: 600; font-size: 12px;
            }
            QPushButton:hover { background-color: #3f3f46; border-color: #60a5fa; }
            QPushButton#primary { background-color: #2563eb; border-color: #3b82f6; color: #ffffff; }
            QPushButton#primary:hover { background-color: #1d4ed8; }
            QComboBox, QSpinBox {
                background-color: #0f0f11; color: #f4f4f5; border: 1px solid #3f3f46;
                border-radius: 4px; padding: 5px 8px; font-size: 13px;
            }
            QTabWidget::pane { border: 1px solid #27272a; background: #18181b; border-radius: 6px; }
            QTabBar::tab { background: #0f0f11; color: #a1a1aa; padding: 10px 18px; font-weight: bold; }
            QTabBar::tab:selected { background: #18181b; color: #60a5fa; border-bottom: 2px solid #3b82f6; }
            QTextBrowser { background-color: #0f0f11; color: #e4e4e7; border: 1px solid #27272a; padding: 12px; border-radius: 6px; }
            QTableWidget {
                background-color: #0f0f11; color: #e4e4e7; gridline-color: #27272a;
                border: 1px solid #27272a; border-radius: 6px; font-size: 12px;
            }
            QHeaderView::section {
                background-color: #18181b; color: #60a5fa; font-weight: bold;
                border: 1px solid #27272a; padding: 6px;
            }
        """)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self.tab_interactive = QWidget()
        self._build_interactive_tab(self.tab_interactive)
        self.tabs.addTab(self.tab_interactive, "1. Интерактивное распознавание (К, О, Л)")

        self.tab_study = QWidget()
        self._build_study_tab(self.tab_study)
        self.tabs.addTab(self.tab_study, "2. Статистический анализ шума (10% – 100%)")

        self.tab_samples = QWidget()
        self._build_samples_tab(self.tab_samples)
        self.tabs.addTab(self.tab_samples, "3. Визуальный ряд искажений и весовые коэффициенты")

    def _build_interactive_tab(self, parent: QWidget) -> None:
        layout = QHBoxLayout(parent)
        layout.setSpacing(15)

        left_card = QFrame()
        left_card.setObjectName("card")
        left_card.setFixedWidth(330)
        left_layout = QVBoxLayout(left_card)
        left_layout.setSpacing(10)

        lbl_ref = QLabel("1. ВЫБОР ЭТАЛОННОГО ОБРАЗА")
        lbl_ref.setObjectName("title")
        left_layout.addWidget(lbl_ref)

        btn_row = QHBoxLayout()
        for letter in ["К", "О", "Л"]:
            btn = QPushButton(f"Буква «{letter}»")
            btn.clicked.connect(lambda checked, l=letter: self._load_reference(l))
            btn_row.addWidget(btn)
        left_layout.addLayout(btn_row)

        thumbs_row = QHBoxLayout()
        self.ref_widgets = {}
        for letter, vec in self.patterns.items():
            box = QVBoxLayout()
            lbl = QLabel(f"Эталон «{letter}»")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setObjectName("subtitle")
            w = PixelGridWidget(GRID_SIZE, editable=False, cell_px=8)
            w.set_state(vec)
            box.addWidget(w, alignment=Qt.AlignmentFlag.AlignCenter)
            box.addWidget(lbl)
            thumbs_row.addLayout(box)
            self.ref_widgets[letter] = w
        left_layout.addLayout(thumbs_row)

        left_layout.addSpacing(8)
        lbl_noise = QLabel("2. ЗАШУМЛЕНИЕ ОБРАЗА")
        lbl_noise.setObjectName("title")
        left_layout.addWidget(lbl_noise)

        noise_status_row = QHBoxLayout()
        noise_status_row.addWidget(QLabel("Выбранный уровень шума:"))
        self.lbl_noise_val = QLabel("0%")
        self.lbl_noise_val.setStyleSheet("color: #3b82f6; font-weight: bold;")
        noise_status_row.addStretch()
        noise_status_row.addWidget(self.lbl_noise_val)
        left_layout.addLayout(noise_status_row)

        preset_grid = QGridLayout()
        for idx, p_val in enumerate(NOISE_LEVELS):
            b = QPushButton(f"{p_val}%")
            b.clicked.connect(lambda checked, val=p_val: self._apply_preset_noise(val))
            preset_grid.addWidget(b, idx // 4, idx % 4)
        left_layout.addLayout(preset_grid)

        left_layout.addSpacing(8)
        lbl_mode = QLabel("3. РЕЖИМ ФУНКЦИОНИРОВАНИЯ")
        lbl_mode.setObjectName("title")
        left_layout.addWidget(lbl_mode)

        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Синхронный режим", "Асинхронный режим"])
        self.cb_mode.currentIndexChanged.connect(self._recognize_current)
        left_layout.addWidget(self.cb_mode)

        btn_recognize = QPushButton("Распознать образ")
        btn_recognize.setObjectName("primary")
        btn_recognize.clicked.connect(self._recognize_current)
        left_layout.addWidget(btn_recognize)

        btn_clear = QPushButton("Очистить холст")
        btn_clear.clicked.connect(self._clear_canvas)
        left_layout.addWidget(btn_clear)

        left_layout.addStretch()
        layout.addWidget(left_card)

        center_card = QFrame()
        center_card.setObjectName("card")
        center_layout = QVBoxLayout(center_card)

        lbl_canvas = QLabel("ВИЗУАЛИЗАЦИЯ СОСТОЯНИЙ НЕЙРОННОЙ СЕТИ (10×10 = 100 НЕЙРОНОВ)")
        lbl_canvas.setObjectName("title")
        lbl_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(lbl_canvas)

        hint = QLabel("Левая сетка интерактивна: кликайте или зажимайте ЛКМ для ручного рисования и искажения пикселей")
        hint.setObjectName("subtitle")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(hint)

        grids_row = QHBoxLayout()
        grids_row.setSpacing(30)

        in_box = QVBoxLayout()
        lbl_in = QLabel("Входной (тестовый) образ")
        lbl_in.setStyleSheet("font-weight: bold; font-size: 14px; color: #e4e4e7;")
        lbl_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grid_input = PixelGridWidget(GRID_SIZE, editable=True, cell_px=30)
        self.grid_input.on_change_callback = self._recognize_current
        in_box.addWidget(lbl_in)
        in_box.addWidget(self.grid_input, alignment=Qt.AlignmentFlag.AlignCenter)
        grids_row.addLayout(in_box)

        out_box = QVBoxLayout()
        lbl_out = QLabel("Восстановленный образ")
        lbl_out.setStyleSheet("font-weight: bold; font-size: 14px; color: #10b981;")
        lbl_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grid_output = PixelGridWidget(GRID_SIZE, editable=False, cell_px=30)
        out_box.addWidget(lbl_out)
        out_box.addWidget(self.grid_output, alignment=Qt.AlignmentFlag.AlignCenter)
        grids_row.addLayout(out_box)

        center_layout.addLayout(grids_row)

        self.bar_plot = pg.PlotWidget(title="Сходство (в %) восстановленного образа с эталонами")
        self.bar_plot.setMaximumHeight(230)
        self.bar_plot.setYRange(0, 105)
        self.bar_plot.showGrid(x=False, y=True, alpha=0.2)
        center_layout.addWidget(self.bar_plot)

        layout.addWidget(center_card, stretch=1)

        right_card = QFrame()
        right_card.setObjectName("card")
        right_card.setFixedWidth(360)
        right_layout = QVBoxLayout(right_card)

        lbl_verdict = QLabel("РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ")
        lbl_verdict.setObjectName("title")
        right_layout.addWidget(lbl_verdict)

        self.txt_verdict = QTextBrowser()
        right_layout.addWidget(self.txt_verdict)
        layout.addWidget(right_card)

    def _build_study_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setSpacing(12)

        top_bar = QFrame()
        top_bar.setObjectName("card")
        top_layout = QHBoxLayout(top_bar)

        top_layout.addWidget(QLabel("Количество тестовых образов на каждый класс и уровень шума:"))
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(10, 200)
        self.spin_trials.setValue(20)
        top_layout.addWidget(self.spin_trials)

        self.cb_study_mode = QComboBox()
        self.cb_study_mode.addItems(["Синхронный режим", "Асинхронный режим"])
        top_layout.addWidget(self.cb_study_mode)

        btn_run = QPushButton("Запустить серию испытаний")
        btn_run.setObjectName("primary")
        btn_run.clicked.connect(self._run_statistical_study)
        top_layout.addWidget(btn_run)
        top_layout.addStretch()
        layout.addWidget(top_bar)

        mid_layout = QHBoxLayout()

        self.plot_study = pg.PlotWidget(title="Зависимость точности распознавания от процента зашумления")
        self.plot_study.setLabel("bottom", "Процент зашумления образа, %")
        self.plot_study.setLabel("left", "Точность распознавания, %")
        self.plot_study.setXRange(5, 105)
        self.plot_study.setYRange(-5, 105)
        self.plot_study.showGrid(x=True, y=True, alpha=0.2)
        self.plot_study.addLegend(offset=(15, 15))
        mid_layout.addWidget(self.plot_study, stretch=3)

        self.tbl_study = QTableWidget(len(NOISE_LEVELS), 5)
        self.tbl_study.setHorizontalHeaderLabels(["Шум, %", "Буква «К»", "Буква «О»", "Буква «Л»", "Средняя, %"])
        self.tbl_study.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_study.setMinimumWidth(480)
        mid_layout.addWidget(self.tbl_study, stretch=2)

        layout.addLayout(mid_layout, stretch=1)

    def _build_samples_tab(self, parent: QWidget) -> None:
        layout = QHBoxLayout(parent)
        layout.setSpacing(15)

        left_frame = QFrame()
        left_frame.setObjectName("card")
        left_vbox = QVBoxLayout(left_frame)

        header_row = QHBoxLayout()
        lbl_samples = QLabel("ВИД ИСКАЖЕННЫХ И ВОССТАНОВЛЕННЫХ ОБРАЗОВ")
        lbl_samples.setObjectName("title")
        header_row.addWidget(lbl_samples)
        header_row.addStretch()

        header_row.addWidget(QLabel("Тестируемый образ:"))
        self.cb_sample_letter = QComboBox()
        self.cb_sample_letter.addItems(["К", "О", "Л"])
        self.cb_sample_letter.currentTextChanged.connect(self._update_samples_view)
        header_row.addWidget(self.cb_sample_letter)

        btn_regen = QPushButton("Обновить выборку шума")
        btn_regen.clicked.connect(self._update_samples_view)
        header_row.addWidget(btn_regen)
        left_vbox.addLayout(header_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #0f0f11; }")
        self.samples_container = QWidget()
        self.samples_grid = QGridLayout(self.samples_container)
        self.samples_grid.setSpacing(12)
        scroll.setWidget(self.samples_container)
        left_vbox.addWidget(scroll)

        layout.addWidget(left_frame, stretch=3)

        right_frame = QFrame()
        right_frame.setObjectName("card")
        right_vbox = QVBoxLayout(right_frame)

        lbl_w = QLabel("МАТРИЦА ВЕСОВЫХ КОЭФФИЦИЕНТОВ W (100×100)")
        lbl_w.setObjectName("title")
        right_vbox.addWidget(lbl_w)

        lbl_w_desc = QLabel(
            "Симметричная матрица связей между нейронами.\n"
            "Главная диагональ равна нулю (отсутствие обратной связи нейрона на себя)."
        )
        lbl_w_desc.setObjectName("subtitle")
        right_vbox.addWidget(lbl_w_desc)

        self.w_plot = pg.PlotWidget()
        self.w_plot.setAspectLocked(True)
        self.w_plot.invertY(True)
        self.w_img = pg.ImageItem()
        self.w_plot.addItem(self.w_img)
        self.w_img.setImage(self.network.weights)
        cmap = pg.colormap.get("CET-D1")
        self.w_img.setColorMap(cmap)
        right_vbox.addWidget(self.w_plot)

        layout.addWidget(right_frame, stretch=2)

    def _load_reference(self, letter: str) -> None:
        self.current_base_label = letter
        self.current_noise_percent = 0
        self.lbl_noise_val.setText("0%")
        self.grid_input.set_state(self.patterns[letter])
        self._recognize_current()

    def _apply_preset_noise(self, noise_val: int) -> None:
        self.current_noise_percent = noise_val
        self.lbl_noise_val.setText(f"{noise_val}%")
        base_vec = self.patterns[self.current_base_label]
        noisy_vec = apply_noise(base_vec, noise_val, rng=self.rng)
        self.grid_input.set_state(noisy_vec)
        self._recognize_current()

    def _clear_canvas(self) -> None:
        self.current_noise_percent = 0
        self.lbl_noise_val.setText("—")
        self.grid_input.set_state(np.full(N_NEURONS, -1, dtype=int))
        self._recognize_current()

    def _recognize_current(self) -> None:
        input_vec = self.grid_input.get_state()
        synchronous = (self.cb_mode.currentIndex() == 0)
        res = self.network.recall(input_vec, synchronous=synchronous, rng_seed=self.rng)
        self.grid_output.set_state(res.final_state)

        self.bar_plot.clear()
        letters = list(self.patterns.keys())
        sims = [res.similarities[l] for l in letters]
        bg = pg.BarGraphItem(x=[0, 1, 2], height=sims, width=0.45, brushes=["#3b82f6", "#10b981", "#f59e0b"])
        self.bar_plot.addItem(bg)
        ax = self.bar_plot.getAxis("bottom")
        ax.setTicks([[(0, "Буква «К»"), (1, "Буква «О»"), (2, "Буква «Л»")]])

        self._update_verdict_html(res)

    def _update_verdict_html(self, res) -> None:
        cap = self.network.max_capacity()
        dist_from_base = hamming_distance(res.initial_state, self.patterns[self.current_base_label])

        if res.matched_label:
            verdict_title = f"ОБРАЗ РАСПОЗНАН: БУКВА «{res.matched_label}»"
            verdict_color = "#10b981"
            verdict_desc = "Сеть успешно пришла в стабильное состояние, соответствующее эталону."
        elif res.inverted_label:
            verdict_title = f"ИНВЕРСНЫЙ ОБРАЗ «-{res.inverted_label}»"
            verdict_color = "#f59e0b"
            verdict_desc = "Из-за сильного зашумления (>50%) сеть сошлась к зеркальному негативу эталона."
        else:
            verdict_title = "НЕОПОЗНАННЫЙ ОБРАЗ"
            verdict_color = "#ef4444"
            verdict_desc = "Сеть стабилизировалась в побочном состоянии, не совпадающем с эталонами."

        html = f"""
        <div style="font-family: 'Segoe UI', sans-serif; color: #e4e4e7;">
            <div style="color: #71717a; font-size: 11px; text-transform: uppercase;">Вердикт сети</div>
            <div style="background-color: #09090b; padding: 12px; border-left: 5px solid {verdict_color}; margin: 8px 0 12px 0;">
                <div style="font-size: 15px; font-weight: bold; color: {verdict_color};">{verdict_title}</div>
                <div style="font-size: 12px; color: #a1a1aa; margin-top: 4px;">{verdict_desc}</div>
            </div>

            <table style="width: 100%; font-size: 13px; margin-bottom: 12px;">
                <tr>
                    <td style="color: #a1a1aa; padding: 4px 0;">Исходный эталон:</td>
                    <td style="text-align: right; font-weight: bold;">Буква «{self.current_base_label}»</td>
                </tr>
                <tr>
                    <td style="color: #a1a1aa; padding: 4px 0;">Искажено пикселей на входе:</td>
                    <td style="text-align: right; font-weight: bold;">{dist_from_base} из {N_NEURONS} ({dist_from_base}%)</td>
                </tr>
                <tr>
                    <td style="color: #a1a1aa; padding: 4px 0;">Итераций до стабилизации:</td>
                    <td style="text-align: right; font-weight: bold; color: #60a5fa;">{res.iterations}</td>
                </tr>
            </table>

            <hr style="border: none; border-bottom: 1px solid #27272a; margin: 12px 0;">
            <div style="color: #71717a; font-size: 11px; text-transform: uppercase; margin-bottom: 6px;">Сходство с эталонами выборки</div>
        """

        for label, sim in res.similarities.items():
            html += f"""
            <div style="background-color: #09090b; padding: 8px 10px; margin-bottom: 6px; border-radius: 4px;">
                <span style="font-weight: bold; color: #60a5fa;">Эталон «{label}»:</span>
                <span style="float: right; font-weight: bold;">{sim:.1f}% совпадения</span>
            </div>
            """

        html += f"""
            <hr style="border: none; border-bottom: 1px solid #27272a; margin: 12px 0;">
            <div style="color: #71717a; font-size: 11px; text-transform: uppercase; margin-bottom: 4px;">Параметры сети</div>
            <div style="font-size: 12px; color: #a1a1aa; line-height: 1.5;">
                • Количество нейронов: <b>n = {N_NEURONS} (10×10)</b><br>
                • Обучающих образов: <b>m = 3 (К, О, Л)</b><br>
                • Максимальная емкость сети: <b>{cap:.2f} образов</b>
            </div>
        </div>
        """
        self.txt_verdict.setHtml(html)

    def _run_statistical_study(self) -> None:
        trials = self.spin_trials.value()
        synchronous = (self.cb_study_mode.currentIndex() == 0)
        results = self.network.run_noise_study(trials_per_level=trials, synchronous=synchronous, seed=42)

        self.plot_study.clear()
        colors = {"К": "#3b82f6", "О": "#10b981", "Л": "#f59e0b", "AVG": "#ef4444"}

        for key, name in [("К", "Буква «К»"), ("О", "Буква «О»"), ("Л", "Буква «Л»")]:
            y_vals = [results[n][key] for n in NOISE_LEVELS]
            self.plot_study.plot(
                NOISE_LEVELS, y_vals,
                pen=pg.mkPen(colors[key], width=2),
                symbol="o", symbolSize=7, symbolBrush=colors[key],
                name=name
            )

        avg_vals = [results[n]["AVG"] for n in NOISE_LEVELS]
        self.plot_study.plot(
            NOISE_LEVELS, avg_vals,
            pen=pg.mkPen(colors["AVG"], width=3, style=Qt.PenStyle.DashLine),
            symbol="s", symbolSize=9, symbolBrush=colors["AVG"],
            name="Средняя точность"
        )

        for row_idx, noise in enumerate(NOISE_LEVELS):
            row_data = [
                f"{noise}%",
                f"{results[noise]['К']:.1f}%",
                f"{results[noise]['О']:.1f}%",
                f"{results[noise]['Л']:.1f}%",
                f"{results[noise]['AVG']:.1f}%"
            ]
            for col_idx, text in enumerate(row_data):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col_idx == 4:
                    val = results[noise]["AVG"]
                    if val >= 90:
                        item.setForeground(QColor("#10b981"))
                    elif val >= 50:
                        item.setForeground(QColor("#f59e0b"))
                    else:
                        item.setForeground(QColor("#ef4444"))
                self.tbl_study.setItem(row_idx, col_idx, item)

    def _update_samples_view(self) -> None:
        letter = self.cb_sample_letter.currentText()
        synchronous = (self.cb_mode.currentIndex() == 0)
        seed = int(self.rng.integers(1, 100000))
        samples = self.network.generate_noise_samples(letter, synchronous=synchronous, seed=seed)

        while self.samples_grid.count():
            child = self.samples_grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for idx, (noise, noisy_vec, res) in enumerate(samples):
            col_group = 0 if idx < 6 else 1
            row_idx = idx if idx < 6 else idx - 6

            card = QFrame()
            card.setStyleSheet("background-color: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 6px;")
            hbox = QHBoxLayout(card)

            res_text = f"«{res.matched_label}»" if res.matched_label else (f"«-{res.inverted_label}»" if res.inverted_label else "Ошибка")
            info_lbl = QLabel(
                f"<b>Шум: {noise}%</b><br>"
                f"<span style='color: #a1a1aa; font-size: 11px;'>"
                f"Итераций: {res.iterations}<br>"
                f"Итог: {res_text}"
                f"</span>"
            )
            info_lbl.setFixedWidth(105)
            hbox.addWidget(info_lbl)

            w_noisy = PixelGridWidget(GRID_SIZE, editable=False, cell_px=11)
            w_noisy.set_state(noisy_vec)
            hbox.addWidget(QLabel("Вход:"))
            hbox.addWidget(w_noisy)

            w_restored = PixelGridWidget(GRID_SIZE, editable=False, cell_px=11)
            w_restored.set_state(res.final_state)
            hbox.addWidget(QLabel("Выход:"))
            hbox.addWidget(w_restored)

            self.samples_grid.addWidget(card, row_idx, col_group)