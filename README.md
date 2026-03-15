**智慧农业-基于YOLOv8的水果蔬菜新鲜度检测系统**，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4e810343648647ad8989aa1cd6140a2e.png)


# 🍎 基于 YOLOv8 的水果蔬菜新鲜度智能检测系统 —— 25类全场景识别平台
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/847bdc0842624a4b993e2f8973425cb1.png)

## 一、项目概览表

| 项目维度       | 内容描述                                                                 |
|----------------|--------------------------------------------------------------------------|
| **项目名称**   | 基于YOLOv8的水果蔬菜新鲜度检测系统                                       |
| **核心算法**   | YOLOv8 (Ultralytics)                                                     |
| **检测类别**   | **25类**：涵盖新鲜、成熟、腐烂、未成熟四种状态包括：黄瓜、葫芦、生菜、洋葱、土豆、苹果、香蕉、葡萄、芒果、橙子 |
| **数据集规模** | 8164张高质量标注图片                                                     |
| **输入支持**   | 单张图片 / 图片文件夹(批量) / 视频文件 / 实时摄像头                      |
| **输出形式**   | 可视化标注界面 + 实时统计面板 + 结果保存 (图片/文本)                     |
| **前端框架**   | PyQt5 (桌面GUI，支持拖拽、实时预览、多线程)                              |
| **深度学习**   | PyTorch 1.9+ + Ultralytics YOLOv8                                        |
| **图像处理**   | OpenCV-python                                                            |
| **开发环境**   | PyCharm / VSCode + Anaconda (Python 3.9)                                 |
| **适用场景**   | 智慧农业、超市质检、冷链物流、食品溯源、毕业设计、AI竞赛                 |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7c79d4eba5104f30b205f34fc1d08df4.png)

## 二、系统功能矩阵表

| 功能模块         | 子功能                     | 描述                                                                 |
|------------------|----------------------------|----------------------------------------------------------------------|
| **图像检测**     | 单图识别                   | 选择单张图片，实时显示检测框、类别、置信度                           |
|                  | 批量检测                   | 选择文件夹，自动遍历所有图片，列表展示每张结果，支持导出             |
| **视频检测**     | 文件播放                   | 加载本地视频，逐帧检测，实时显示进度与统计                           |
|                  | 摄像头实时                 | 调用本地摄像头，毫秒级响应，适合现场演示                             |
| **参数配置**     | 阈值调节                   | 滑动条自定义 **置信度 (Conf)** 和 **IOU阈值**                        |
|                  | 显示选项                   | 勾选“显示标签”、“显示原图对比”                                     |
| **模型管理**     | 模型切换                   | 下拉菜单选择不同的 `.pt` 权重文件 (如 `best.pt`, `yolov8n.pt`)       |
| **结果展示**     | 实时统计                   | 界面右侧实时显示：目标总数、各类别数量分布                           |
|                  | 详情查看                   | 批量检测时，点击列表项可查看单张大图及详细数据                       |
| **数据导出**     | 结果保存                   | 保存标注后的图片、检测结果文本文件 (JSON/TXT)                        |

---

## 三、技术架构蓝图表

| 层级           | 技术组件                  | 作用说明                                                                 |
|----------------|---------------------------|--------------------------------------------------------------------------|
| **表现层 (UI)**| PyQt5                     | 构建桌面图形界面，包含菜单栏、工具栏、绘图区、控制区、状态栏           |
|                | QThread                   | 多线程处理视频/摄像头流，防止界面卡顿                                  |
| **业务逻辑层** | Python Controller         | 响应用户操作，调度检测任务，更新UI数据                                 |
| **AI引擎层**   | YOLOv8 (Ultralytics)      | 核心检测引擎，加载25类训练好的权重，执行推理                           |
|                | PyTorch                   | 深度学习后端，支持CPU/GPU加速                                          |
| **数据处理层** | OpenCV                    | 图像读取、缩放、格式转换 (BGR<->RGB)、视频帧提取                       |
| **存储层**     | 文件系统                  | 存储原始数据、模型权重、检测结果图片                                   |
|                | JSON/TXT                  | 结构化存储检测坐标、类别、置信度                                       |

---

## 四、25类检测类别详解表

| 状态分类 | 涉及蔬果                                     | 具体类别名称 (Class Names)                          |
|----------|----------------------------------------------|-----------------------------------------------------|
| **新鲜** | 黄瓜, 葫芦, 生菜, 洋葱, 土豆                 | `新鲜黄瓜`, `新鲜葫芦`, `新鲜生菜`, `新鲜洋葱`, `新鲜土豆` |
| **成熟** | 苹果, 香蕉, 葡萄, 芒果, 橙子                 | `成熟苹果`, `成熟香蕉`, `成熟葡萄`, `成熟芒果`, `成熟橙子` |
| **腐烂** | 苹果, 香蕉, 黄瓜, 葫芦, 葡萄, 生菜, 芒果, 洋葱, 橙子, 土豆 | `腐烂苹果`, `腐烂香蕉`, `腐烂黄瓜`, `腐烂葫芦`, `腐烂葡萄`, `腐烂生菜`, `腐烂芒果`, `腐烂洋葱`, `腐烂橙子`, `腐烂土豆` |
| **未成熟**| 苹果, 香蕉, 葡萄, 芒果, 橙子                 | `未成熟苹果`, `未成熟香蕉`, `未成熟葡萄`, `未成熟芒果`, `未成熟橙子` |

> 💡 **数据价值**：该数据集覆盖了从采摘到销售的全生命周期状态，能有效辅助分级分拣和损耗预警。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/36d9b4c09be44a29a07d3cef22cfc0fb.png)

## 五、最简易完整代码实现 (PyQt5 + YOLOv8 单文件版)

> 💡 **说明**：这是一个精简但功能完整的单文件示例，实现了**图片检测、摄像头实时检测、阈值调节、结果显示**。你可以直接保存为 `main.py` 运行。

```python
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QSlider, QGroupBox, QComboBox, QTextEdit, QScrollArea
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO
import cv2
import numpy as np

class DetectionThread(QThread):
    frame_signal = pyqtSignal(QImage, dict)

    def __init__(self, model, conf=0.25):
        super().__init__()
        self.model = model
        self.conf = conf
        self.running = False
        self.cap = None

    def run(self):
        self.running = True
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 执行检测
            results = self.model.predict(frame, conf=self.conf, verbose=False)
            result = results[0]
            
            # 绘制结果
            annotated_frame = result.plot()
            
            # 统计信息
            stats = {}
            if result.boxes is not None:
                for cls_id in result.boxes.cls:
                    name = self.model.names[int(cls_id)]
                    stats[name] = stats.get(name, 0) + 1
            
            # 转换格式供QT显示 (BGR -> RGB)
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            self.frame_signal.emit(q_image, stats)
        
        self.stop_camera()

    def start_camera(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return
        self.start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍎 水果蔬菜新鲜度检测系统 (YOLOv8)")
        self.setGeometry(100, 100, 1200, 800)
        
        # 加载模型 (请确保目录下有 best.pt 或使用默认 yolov8n.pt)
        try:
            # 优先尝试加载用户训练好的模型
            if os.path.exists('best.pt'):
                self.model = YOLO('best.pt')
                print("已加载自定义模型 best.pt")
            else:
                self.model = YOLO('yolov8n.pt')
                print("未找到 best.pt，已加载默认模型 yolov8n.pt (类别可能不匹配，请替换模型)")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None

        self.thread = None
        self.current_stats = {}

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左侧：控制与统计面板 ---
        left_panel = QVBoxLayout()
        
        # 1. 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["best.pt", "yolov8n.pt", "yolov8s.pt"])
        self.model_combo.currentTextChanged.connect(self.change_model)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        left_panel.addWidget(model_group)

        # 2. 参数设置
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout()
        
        # 置信度
        conf_label = QLabel(f"置信度阈值: 0.25")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(90)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(lambda v: conf_label.setText(f"置信度阈值: {v/100:.2f}"))
        param_layout.addWidget(conf_label)
        param_layout.addWidget(self.conf_slider)
        
        param_group.setLayout(param_layout)
        left_panel.addWidget(param_group)

        # 3. 操作按钮
        btn_group = QGroupBox("检测模式")
        btn_layout = QVBoxLayout()
        
        self.btn_img = QPushButton("📷 选择图片检测")
        self.btn_img.clicked.connect(self.detect_image)
        btn_layout.addWidget(self.btn_img)
        
        self.btn_cam = QPushButton("🎥 开启摄像头")
        self.btn_cam.clicked.connect(self.toggle_camera)
        btn_layout.addWidget(self.btn_cam)
        
        self.btn_video = QPushButton("🎬 选择视频检测 (暂未实现)")
        self.btn_video.setEnabled(False) # 简化版暂不开启
        btn_layout.addWidget(self.btn_video)
        
        btn_group.setLayout(btn_layout)
        left_panel.addWidget(btn_group)

        # 4. 统计信息
        stat_group = QGroupBox("实时统计")
        stat_layout = QVBoxLayout()
        self.stat_text = QTextEdit()
        self.stat_text.setReadOnly(True)
        self.stat_text.setFont(QFont("Arial", 10))
        stat_layout.addWidget(self.stat_text)
        stat_group.setLayout(stat_layout)
        left_panel.addWidget(stat_group)

        left_panel.addStretch()
        main_layout.addLayout(left_panel, 1)

        # --- 右侧：结果显示区域 ---
        right_panel = QVBoxLayout()
        self.image_label = QLabel("请点击按钮开始检测")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px dashed #ccc; color: #888; }")
        self.image_label.setMinimumSize(640, 480)
        
        right_panel.addWidget(self.image_label)
        main_layout.addLayout(right_panel, 3)

    def change_model(self, model_name):
        if os.path.exists(model_name):
            self.model = YOLO(model_name)
            self.stat_text.append(f"✅ 模型已切换至: {model_name}")
        else:
            self.stat_text.append(f"❌ 未找到模型文件: {model_name}")

    def detect_image(self):
        if not self.model:
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png *.bmp)")
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                return
            
            conf = self.conf_slider.value() / 100.0
            results = self.model.predict(img, conf=conf, verbose=False)
            result = results[0]
            
            annotated_img = result.plot()
            self.show_image(annotated_img)
            
            # 更新统计
            self.update_stats(result)

    def toggle_camera(self):
        if self.thread and self.thread.running:
            self.thread.stop_camera()
            self.btn_cam.setText("🎥 开启摄像头")
            self.image_label.setText("摄像头已停止")
            self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px dashed #ccc; color: #888; }")
        else:
            if not self.model:
                return
            self.thread = DetectionThread(self.model, conf=self.conf_slider.value()/100.0)
            self.thread.frame_signal.connect(self.update_camera_frame)
            self.thread.start_camera(0) # 0表示默认摄像头
            self.btn_cam.setText("⏹ 停止摄像头")

    def update_camera_frame(self, q_image, stats):
        pixmap = QPixmap.fromImage(q_image)
        # 自适应大小
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setStyleSheet("") # 移除背景色
        
        # 更新统计文本
        stat_str = ""
        total = sum(stats.values())
        stat_str += f"总目标数: {total}\n" + "-"*20 + "\n"
        for k, v in sorted(stats.items()):
            stat_str += f"{k}: {v}\n"
        self.stat_text.setText(stat_str)

    def show_image(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setStyleSheet("")

    def update_stats(self, result):
        stats = {}
        if result.boxes is not None:
            for cls_id in result.boxes.cls:
                name = self.model.names[int(cls_id)]
                stats[name] = stats.get(name, 0) + 1
        
        stat_str = f"总目标数: {sum(stats.values())}\n" + "-"*20 + "\n"
        for k, v in sorted(stats.items()):
            stat_str += f"{k}: {v}\n"
        self.stat_text.setText(stat_str)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

---

## 六、简易运行文档
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b69c5db532b4ec582b4cee92a2d539e.png)

### 📖 水果蔬菜新鲜度检测系统 - 快速上手指南

**版本**: 1.0  
**适用环境**: Windows 10/11, Python 3.9, Anaconda

---

#### 第1步：环境准备

1.  **安装 Anaconda** (如果尚未安装)。
2.  **创建虚拟环境**：
    ```bash
    conda create -n fruit_detect python=3.9
    conda activate fruit_detect
    ```
3.  **安装依赖库**：
    ```bash
    pip install ultralytics opencv-python PyQt5 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    *(注：如果没有NVIDIA显卡，去掉 `--index-url...` 部分即可安装CPU版)*

---

#### 第2步：文件准备

确保你的项目文件夹包含以下结构：
```text
Project_Root/
├── main.py              # 上面提供的代码
├── best.pt              # 【关键】训练好的25类水果蔬菜模型权重
├── dataset/             # (可选) 8164张数据集图片
└── requirements.txt     # 依赖列表
```
> ⚠️ **注意**：必须将训练好的 `best.pt` 文件放在同一目录下，否则系统会加载默认的COCO模型（只能检测人、车等，无法检测水果）。

---

#### 第3步：运行系统

在终端中运行：
```bash
python main.py
```

---

#### 第4步：功能使用

1.  **图片检测**：
    *   点击“📷 选择图片检测”。
    *   选择一张包含水果/蔬菜的图片。
    *   右侧窗口显示标注结果，左侧显示各类别数量（如：腐烂苹果 x2, 新鲜香蕉 x1）。

2.  **摄像头实时检测**：
    *   拿一个水果（或打印图片）对着摄像头。
    *   点击“🎥 开启摄像头”。
    *   系统实时框选并分类，统计面板动态刷新。
    *   拖动“置信度阈值”滑块，可过滤低概率检测。

3.  **切换模型**：
    *   如果你有不同的权重文件（如 `yolov8m.pt`），放入目录后在下拉框选择即可热切换。

---

#### 第5步：结果保存 (扩展建议)

当前简易版直接在界面显示。若需保存：
*   **图片**：在代码 `detect_image` 函数末尾添加 `cv2.imwrite('result.jpg', annotated_img)`。
*   **数据**：将 `stats` 字典写入 TXT 或 CSV 文件。

---



祝你的项目在智慧农业、食品安全领域大放异彩！🍇🥦🍋
内含8164张水果蔬菜新鲜度数据集
包括['新鲜黄瓜', '新鲜葫芦', '新鲜生菜', '新鲜洋葱', '新鲜土豆', '成熟苹果', '成熟香蕉', '成熟葡萄', '成熟芒果', '成熟橙子', '腐烂苹果', '腐烂香蕉', '腐烂黄瓜', '腐烂葫芦', '腐烂葡萄', '腐烂生菜', '腐烂芒果', '腐烂洋葱', '腐烂橙子', '腐烂土豆', '未成熟苹果', '未成熟香蕉', '未成熟葡萄', '未成熟芒果', '未成熟橙子']，25类

本项目已经训练好模型，
软件：Pycharm+Anaconda（或者VSCode+Anaconda）
环境：python3.9 opencv-python PyQt5 ultralytics torch1.9等
文件：
①完整程序文件（.py等）
②UI界面源文件、图标（.ui、.qrc、.py等）
③数据集图片，项目运行教程（.jpg、.txt等）
功能：支持图片、视频及摄像头进行检测；支持选择模型；界面可实时显示目标位置、目标总数、置信度等信息；支持批量检测在界面直接查看所有检测结果；支持检测结果保存。
①选择单张图片或者图片文件夹进行识别
②选择视频文件进行识别
③调用本地摄像头进行识别
④自定义置信度，IOU阈值
⑤选择显示标签和原图
⑥选择检测模型
⑦查看批量检测每一张检测结果
