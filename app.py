#!/usr/bin/env python3
"""
智能工业安全检测系统 Web 应用
支持火灾检测、安全带检测、仪表盘识别
Flask后端API服务
"""

import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import base64
import json
from datetime import datetime
import logging
import threading
import time
import tempfile
from werkzeug.utils import secure_filename
import requests
from dashscope import MultiModalConversation
from dotenv import load_dotenv
# import easyocr
import math
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import uuid
# from pathlib import Path
# import sys
# # 添加Fire-Detection路径到系统路径
# FIRE_DETECTION_ROOT = Path('Fire_Detection')
# if str(FIRE_DETECTION_ROOT) not in sys.path:
#     sys.path.append(str(FIRE_DETECTION_ROOT))
# # 导入Fire-Detection相关模块
# try:
#     from models.common import DetectMultiBackend
#     from utils.general import non_max_suppression, scale_coords
#     from utils.torch_utils import select_device
#     FIRE_DETECTION_AVAILABLE = True
# except ImportError as e:
#     print(f"Fire-Detection模块导入失败: {e}")
#     FIRE_DETECTION_AVAILABLE = False

# 加载环境变量
load_dotenv()

# 配置日志 - 在PaddleOCR初始化前设置
def setup_logging():
    """设置日志配置，防止被PaddleOCR覆盖"""
    # 获取根日志记录器
    root_logger = logging.getLogger()

    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # 添加处理器到根日志记录器
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    # 设置特定模块的日志级别
    logging.getLogger('werkzeug').setLevel(logging.INFO)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    return logging.getLogger(__name__)

# 初始化日志
logger = setup_logging()

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
models = {}  # 存储多个模型
model_loaded = {}  # 记录各模型加载状态
ocr_reader = None  # OCR识别器

# 阿里云多模态大模型相关变量
dashscope_api_key = None
qwen_api_available = False

# 本地Qwen3-VL多模态大模型相关变量
qwen3vl_model = None
qwen3vl_processor = None
qwen3vl_model_loaded = False
qwen3vl_model_path = os.getenv('QWEN3VL_MODEL_PATH', "runs/Qwen3-VL-4B-Instruct")
detect_th_fire_smoke = float(os.getenv('DETECT_THRESHOLD_FIRE_SMOKE', "0.5"))
detect_th_safety_harness = float(os.getenv('DETECT_THRESHOLD', "0.5"))
detect_th_safety_helmet = float(os.getenv('DETECT_THRESHOLD', "0.5"))
detect_th_fall_detection = float(os.getenv('DETECT_THRESHOLD', "0.5"))

# 配置
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
VIDEO_FOLDER = 'videos'

# 多模型路径配置
MODEL_CONFIGS = {
    'fire_smoke': {
        'name': '火灾烟雾检测',
        'path': os.getenv('FIRE_SMOKE_MODEL_PATH', 'runs/detect/fire_smoke_detection/weights/best.pt'),
        'backup_path': '',
        'threshold': detect_th_fire_smoke,
        'classes': ['smoke', 'fire']
    },
    'safety_harness': {
        'name': '安全带检测',
        'path': os.getenv('SAFETY_HARNESS_MODEL_PATH', 'runs/detect/safety_harness_detection/weights/best.pt'),
        'backup_path': None,
        'threshold': detect_th_safety_harness,
        'classes': ['safe harness', 'no harness']
    },
    'safety_helmet': {
        'name': '安全帽检测',
        'path': os.getenv('SAFETY_HELMET_MODEL_PATH', 'runs/detect/safety_helmet_detection/weights/best.pt'),
        'backup_path': None,
        'threshold': detect_th_safety_helmet,
        'classes': ['head', 'helmet', 'others', 'person']
    },
    # 'pressure_gauge': {
    #     'name': '仪表盘识别',
    #     'path': 'runs/detect/pressure_detection/weights/best.pt',
    #     'backup_path': None,
    #     'classes': ['base', 'circle_plate', 'maximum', 'minimum', 'needle', 'number', 'tip']
    # },
    'fall_detection': {
        'name': '跌倒检测',
        'path': os.getenv('FALL_DETECTION_MODEL_PATH', 'runs/detect/fall_detection/weights/best.pt'),
        'backup_path': None,
        'threshold': detect_th_fall_detection,
        'classes': ['Fall-Detected']
    }
}

# 阿里云多模态大模型配置
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取API密钥
QWEN_MODEL_NAME = "qwen-vl-max"  # 阿里云的多模态模型名称
QWEN_MODEL_NAME_OCR = "qwen-vl-ocr"
QWEN_MODEL_NAME_DEEP = "qvq-max"

# 相机接口配置
GET_CAMERA_INFO_URL = os.getenv('CAMERA_URL', 'http://127.0.0.1:5000/api/mock_camera_info')

# 模拟相机信息接口
@app.route('/api/mock_camera_info', methods=['GET'])
def mock_camera_info():
    """模拟相机信息接口，返回固定的相机列表"""
    return jsonify(camera_info_cache)

# 全局变量，用于存储相机信息
camera_info_cache = {
        "CameraObject": 
            [
            {
                "cameraId": "2024Q132Q131",
                "cameraName": f"摄像头1",
                "address": "rtsp://192.168.0.226:8554/live/mystream"
            },
            {
                "cameraId": "2024Q132Q1322",
                "cameraName": "摄像头2",
                "address": "uploads/fire.mp4"
            },
            {
                "cameraId": "2024Q132Q1322",
                "cameraName": "摄像头6",
                "address": "uploads/people.mp4"
            },
            {
                "cameraId": "2024Q132Q1322",
                "cameraName": "摄像头3",
                "address": "uploads/安全帽.mp4"
            },
            {
                "cameraId": "2024Q132Q1322",
                "cameraName": "摄像头4",
                "address": "uploads/跌倒.mp4"
            },
            {
                "cameraId": "2024Q132Q1322",
                "cameraName": "摄像头5",
                "address": "uploads/施工安全带.mp4"
            }
        ]
    }
last_camera_info_fetch_time = None
CAMERA_INFO_REFRESH_INTERVAL = 86400  # 24小时，单位秒

def update_camera_info():
    """从接口获取并更新相机信息"""
    global camera_info_cache, last_camera_info_fetch_time
    try:
        response = requests.get(GET_CAMERA_INFO_URL)
        if response.status_code == 200:
            camera_info_cache = response.json()
            last_camera_info_fetch_time = time.time()
            logger.info(f"成功获取相机信息，共 {len(camera_info_cache.get('CameraObject', []))} 个相机")
        else:
            logger.error(f"获取相机信息失败，状态码: {response.status_code}")
    except Exception as e:
        logger.error(f"获取相机信息时发生错误: {e}")




# 支持的文件格式
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

# 创建必要的文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# 视频处理相关变量
video_processing = False
current_video_path = None
video_results = []
processed_video_path = None  # 添加处理后视频路径的全局变量
current_model_type = None  # 添加当前模型类型

# 视频流处理相关变量
stream_processing = False
stream_results = []
stream_thread = None

# 实时视频结果推送相关变量
realtime_video_results = []  # 实时视频检测结果
video_frame_count = 0  # 当前处理的帧数
video_total_frames = 0  # 视频总帧数

# RTSP检测相关全局变量
rtsp_detection_active = False  # RTSP检测是否激活
rtsp_detection_thread = None  # RTSP检测线程
rtsp_stream_url = None  # RTSP流地址
rtsp_push_url = None  # 报警推送地址
rtsp_detection_results = []  # RTSP检测结果
rtsp_detection_lock = threading.Lock()  # 线程锁
rtsp_multi_model_mode = False  # 多模型检测模式
rtsp_push_frequency = 30  # 推送频率（秒）
rtsp_multi_model_threads = {}  # 多模型检测线程字典

# 独立的大模型检测线程管理变量
qwen3vl_detection_active = False  # 大模型检测是否激活
qwen3vl_detection_thread = None  # 大模型检测线程
qwen3vl_detection_interval = 5  # 大模型检测间隔（秒）
qwen3vl_detection_results = []  # 大模型检测结果
qwen3vl_detection_lock = threading.Lock()  # 大模型检测线程锁
qwen3vl_last_alert_time = {}  # 大模型检测最后报警时间

# 报警编码配置
ALERT_CODES = {
    'smoke': 2000,      # 烟雾
    'fire': 2001,       # 明火
    'Fall-Detected': 2002,  # 跌倒
    'no harness': 2003,  # 未系安全带
    'head': 2004    # 未戴安全帽
}

# 外部报警推送状态存储
external_alert_notifications = []  # 存储接收到的外部报警推送
external_alert_lock = threading.Lock()  # 线程锁

# ================= 多流管理与并发调度器 =================
# 限制GPU并发推理的信号量，避免显存溢出
GPU_MAX_CONCURRENCY = int(os.getenv('GPU_MAX_CONCURRENCY', '2'))
inference_semaphore = threading.Semaphore(GPU_MAX_CONCURRENCY)

# 报警推送线程池与并发限制，避免频繁创建线程导致阻塞
from concurrent.futures import ThreadPoolExecutor
ALERT_PUSH_MAX_WORKERS = int(os.getenv('ALERT_PUSH_MAX_WORKERS', '4'))
ALERT_PUSH_MAX_CONCURRENCY = int(os.getenv('ALERT_PUSH_MAX_CONCURRENCY', '2'))
alert_push_executor = ThreadPoolExecutor(max_workers=ALERT_PUSH_MAX_WORKERS)
alert_push_semaphore = threading.Semaphore(ALERT_PUSH_MAX_CONCURRENCY)

# 多RTSP流的管理容器
streams = {}
streams_lock = threading.Lock()

# 支持的基础模型列表（用于前端多选）
STREAM_MODELS_ALLOWED = ['fire_smoke', 'safety_harness', 'safety_helmet', 'fall_detection']


# def detect_fire_smoke_with_yolov5(image, conf_thres=0.25, iou_thres=0.45):
#     """使用Fire-Detection的YOLOv5模型进行火灾烟雾检测"""
#     global models, model_loaded
    
#     if not FIRE_DETECTION_AVAILABLE:
#         logger.error("Fire-Detection模块不可用，回退到原始检测")
#         return detect_with_original_model('fire_smoke', image)
    
#     if 'fire_smoke' not in models or not model_loaded.get('fire_smoke', False):
#         logger.error("火灾烟雾检测模型未加载")
#         return []
    
#     try:
#         # 获取模型
#         model = models['fire_smoke']
        
#         # 图像预处理
#         if isinstance(image, Image.Image):
#             img_array = np.array(image)
#         else:
#             img_array = image
            
#         # 转换为RGB格式
#         if len(img_array.shape) == 3 and img_array.shape[2] == 3:
#             img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
#         else:
#             img_rgb = img_array
        
#         # 使用Fire_Detection目录下的YOLOv5模型进行检测
#         # 创建输入张量
#         # 首先调整图像大小为模型所需的尺寸
#         img_resized = cv2.resize(img_rgb, (640, 640))
#         # 转换为BCHW格式 (Batch, Channel, Height, Width)
#         img = torch.from_numpy(img_resized).to(select_device(''))
#         img = img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         # 确保通道维度在正确的位置 (从HWC转换为CHW)
#         img = img.permute(2, 0, 1)
#         if len(img.shape) == 3:
#             img = img.unsqueeze(0)  # 扩展批次维度，确保BCHW格式
        
#         # 推理
#         pred = model(img, augment=False)
        
#         # 非极大值抑制
#         pred = non_max_suppression(pred, conf_thres, iou_thres, MODEL_CONFIGS['fire_smoke']['classes'], False, max_det=1000)
        
#         detections = []
#         for i, det in enumerate(pred):  # 每张图片
#             if len(det):
#                 # 将边界框从img_size缩放到im0大小
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_rgb.shape).round()
                
#                 # 处理检测结果
#                 for *xyxy, conf, cls in reversed(det):
#                     # 获取检测信息
#                     x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
#                     confidence = float(conf.item())
#                     class_id = int(cls.item())
                    
#                     # 获取类别名称
#                     class_names = ['smoke', 'fire']  # Fire-Detection模型的类别
#                     class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'
                    
#                     detection = {
#                         'class': class_name,
#                         'confidence': confidence,
#                         'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                         'method': 'fire_detection_yolov5'
#                     }
#                     detections.append(detection)
        
#         logger.info(f"Fire-Detection YOLOv5检测完成，发现 {len(detections)} 个目标")
#         return detections
        
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         logger.error(f"Fire-Detection YOLOv5检测失败: {str(e)}")
#         return []

def detect_with_original_model(model_type, image):
    """原始模型检测函数（备用）"""
    if model_type not in models or not model_loaded.get(model_type, False):
        return []
    
    try:
        results = models[model_type](image, conf=0.5)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    class_names = MODEL_CONFIGS[model_type]['classes']
                    class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'
                    
                    detection = {
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'method': 'original_model'
                    }
                    detections.append(detection)
        
        return detections
        
    except Exception as e:
        logger.error(f"原始模型检测失败: {str(e)}")
        return []

def allowed_file(filename, file_type='image'):
    """检查文件扩展名是否被允许"""
    if '.' not in filename:
        return False

    extension = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return extension in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return extension in ALLOWED_VIDEO_EXTENSIONS
    else:
        return extension in ALLOWED_IMAGE_EXTENSIONS or extension in ALLOWED_VIDEO_EXTENSIONS

def load_models():
    """加载所有YOLO模型"""
    global models, model_loaded, ocr_reader

    # 初始化增强OCR识别器
    # try:
    #     logger.info("初始化OCR识别器...")
    #     from enhanced_ocr import initialize_enhanced_ocr, get_enhanced_ocr

    #     # 初始化增强OCR（支持EasyOCR和PaddleOCR）
    #     enhanced_ocr = initialize_enhanced_ocr(
    #         use_easyocr=False,
    #         use_paddleocr=True,
    #         ocr_mode='paddleocr'  # 混合模式
    #     )
    #     ocr_reader = enhanced_ocr

    #     # PaddleOCR初始化后重新设置日志配置
    #     logger.info("增强OCR识别器初始化成功!")
    #     logger.info("重新配置日志系统...")
    #     setup_logging()  # 重新设置日志配置
    #     logger.info("日志系统重新配置完成!")

    # except ImportError as e:
    #     logger.warning(f"增强OCR模块导入失败，回退到EasyOCR: {str(e)}")
    #     try:
    #         ocr_reader = easyocr.Reader(['en'], gpu=True)
    #         logger.info("EasyOCR初始化成功!")
    #     except Exception as e2:
    #         logger.warning(f"GPU模式失败，使用CPU模式: {str(e2)}")
    #         try:
    #             ocr_reader = easyocr.Reader(['en'], gpu=False)
    #             logger.info("EasyOCR(CPU模式)初始化成功!")
    #         except Exception as e3:
    #             logger.error(f"OCR识别器初始化完全失败: {str(e3)}")
    #             ocr_reader = None
    # except Exception as e:
    #     logger.error(f"增强OCR识别器初始化失败: {str(e)}")
    #     ocr_reader = None

    # 加载各个模型
    for model_key, config in MODEL_CONFIGS.items():
        try:
            model_path = config['path']
            backup_path = config['backup_path']

            # 特殊处理fire_smoke模型，使用Fire_Detection目录下的模型
            # if model_key == 'fire_smoke' and FIRE_DETECTION_AVAILABLE:
            #     try:
            #         # 使用Fire_Detection目录下的DetectMultiBackend加载模型
            #         logger.info(f"使用Fire_Detection加载{config['name']}模型: {model_path}")
            #         device = select_device('')  # 使用默认设备
            #         models[model_key] = DetectMultiBackend(model_path, device=device).to(device).eval()
            #         model_loaded[model_key] = True
            #         logger.info(f"{config['name']}模型使用Fire_Detection加载成功!")
            #         continue  # 跳过下面的常规加载
            #     except Exception as e:
            #         import traceback
            #         traceback.print_exc()
            #         logger.error(f"使用Fire_Detection加载{config['name']}模型失败: {str(e)}")
            #         # 失败后尝试使用常规方法加载
            
            # 常规模型加载
            if os.path.exists(model_path):
                logger.info(f"加载{config['name']}模型: {model_path}")
                models[model_key] = YOLO(model_path).cuda()
                model_loaded[model_key] = True
                logger.info(f"{config['name']}模型加载成功!")
            elif backup_path and os.path.exists(backup_path):
                logger.info(f"加载{config['name']}备用模型: {backup_path}")
                models[model_key] = YOLO(backup_path)
                model_loaded[model_key] = True
                logger.info(f"{config['name']}备用模型加载成功!")
            else:
                logger.warning(f"{config['name']}模型文件不存在: {model_path}")
                model_loaded[model_key] = False

        except Exception as e:
            logger.error(f"{config['name']}模型加载失败: {str(e)}")
            model_loaded[model_key] = False

    # 检查是否至少有一个模型加载成功
    loaded_count = sum(model_loaded.values())
    logger.info(f"成功加载 {loaded_count}/{len(MODEL_CONFIGS)} 个模型")

    # 加载Qwen3-VL多模态大模型
    load_qwen3vl_model()

    return loaded_count > 0

def load_qwen3vl_model():
    """加载Qwen3-VL多模态大模型"""
    global qwen3vl_model, qwen3vl_processor, qwen3vl_model_loaded
    
    try:
        logger.info("开始加载Qwen3-VL多模态大模型...")
        
        # 检查模型路径是否存在
        if not os.path.exists(qwen3vl_model_path):
            logger.warning(f"Qwen3-VL模型路径不存在: {qwen3vl_model_path}")
            logger.info("跳过Qwen3-VL模型加载，系统将使用基础模型运行")
            qwen3vl_model_loaded = False
            return
        
        model_name = qwen3vl_model_path
        logger.info(f"从本地路径加载模型: {model_name}")
        
        # 设置较小的内存使用
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("模型加载超时")
        
        # 设置60秒超时
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        try:
            # 加载模型和处理器
            qwen3vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,  # 使用半精度减少内存
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True  # 减少CPU内存使用
            )
            
            qwen3vl_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            signal.alarm(0)  # 取消超时
            qwen3vl_model_loaded = True
            logger.info("Qwen3-VL多模态大模型加载成功!")
            
        except TimeoutError:
            signal.alarm(0)  # 取消超时
            logger.warning("Qwen3-VL模型加载超时，跳过加载，系统将使用基础模型运行")
            qwen3vl_model_loaded = False
            qwen3vl_model = None
            qwen3vl_processor = None
        
    except Exception as e:
        logger.error(f"Qwen3-VL模型加载失败: {str(e)}")
        logger.info("系统将使用基础模型运行")
        qwen3vl_model_loaded = False
        qwen3vl_model = None
        qwen3vl_processor = None

def calculate_distance(point1, point2):
    """计算两点之间的距离"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def convert_to_python_types(obj):
    """递归转换numpy类型为Python原生类型"""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(item) for item in obj)
    else:
        return obj

def get_box_center(box):
    """获取检测框的中心点"""
    x1, y1, x2, y2 = box
    center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
    return convert_to_python_types(center)  # 确保返回Python原生类型



def determine_reasonable_gauge_range(valid_numbers, all_detections):
    """
    根据检测到的组件确定合理的表盘范围
    避免离谱的读数错误
    """
    try:
        # # 检查是否有minimum和maximum标记
        # minimum_detections = [d for d in all_detections if d['class_name'] == 'minimum']
        # maximum_detections = [d for d in all_detections if d['class_name'] == 'maximum']

        # # 常见的工业表盘范围模式
        # common_ranges = [
        #     (0, 10), (0, 16), (0, 25), (0, 40), (0, 60), (0, 100),
        #     (0, 160), (0, 250), (0, 400), (0, 600), (0, 1000),
        #     (-1, 1), (-10, 10), (0, 1.6), (0, 2.5), (0, 4), (0, 6)
        # ]

        # # 如果有minimum和maximum，优先使用
        # if minimum_detections and maximum_detections:
        #     # 根据位置关系推断范围，通常是0到某个整数
        #     return {"min_value": 0, "max_value": 10, "confidence": 0.8, "method": "min_max_markers"}

        # # 如果有数字，尝试从数字推断
        # if valid_numbers and len(valid_numbers) >= 2:
        #     # 假设最常见的0-10范围，实际应该通过OCR识别数字
        #     return {"min_value": 0, "max_value": 10, "confidence": 0.7, "method": "number_inference"}

        # 默认使用最常见的0-10范围
        return {"min_value": 0, "max_value": 1000, "confidence": 0.6, "method": "default_range"}

    except Exception as e:
        logger.error(f"确定表盘范围失败: {e}")
        return {"min_value": 0, "max_value": 1000, "confidence": 0.3, "method": "error_fallback"}

def extract_number_from_box(image, box, ocr_reader):
    """从检测框中提取数字"""
    if ocr_reader is None:
        logger.warning("OCR识别器未初始化")
        return None

    try:
        x1, y1, x2, y2 = map(int, box)
        # 确保坐标在图像范围内
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        logger.debug(f"    OCR区域: ({x1},{y1}) -> ({x2},{y2}), 图像尺寸: {w}x{h}")

        # 裁剪数字区域
        number_region = image[y1:y2, x1:x2]

        if number_region.size == 0:
            logger.warning("    裁剪区域为空")
            return None

        logger.debug(f"    裁剪区域尺寸: {number_region.shape}")

        # 使用OCR识别数字（支持增强OCR）
        if hasattr(ocr_reader, 'extract_numbers'):
            # 使用增强OCR的数字提取功能
            logger.debug("    使用增强OCR进行数字识别")
            number_results = ocr_reader.extract_numbers(number_region)
            logger.debug(f"    增强OCR数字结果: {number_results}")

            if number_results:
                # 返回置信度最高的数字
                best_result = number_results[0]
                logger.debug(f"    增强OCR识别到数字: {best_result['value']} (置信度: {best_result['confidence']}, 来源: {best_result['source']})")
                return best_result['value']
        else:
            # 使用传统EasyOCR
            logger.debug("    使用传统EasyOCR进行数字识别")
            results = ocr_reader.readtext(number_region)
            logger.debug(f"    OCR原始结果: {results}")

            # 提取数字 - 降低置信度阈值并尝试多种模式
            best_number = None
            best_confidence = 0

            for (_, text, confidence) in results:
                logger.debug(f"    OCR文本: '{text}', 置信度: {confidence}")

                if confidence > 0.8:  # 降低置信度阈值
                    # 预处理：替换常见的OCR错误字符
                    text = text.replace('—', '-').replace('–', '-')  # 替换不同类型的破折号
                    text = text.replace(',', '.')  # 替换逗号为小数点
                    text = text.replace('O', '0').replace('o', '0')  # 替换字母O为数字0
                    text = text.replace('l', '1').replace('I', '1')  # 替换字母l和I为数字1

                    # 尝试多种数字提取模式
                    import re

                    # 模式1: 提取负数和正数（包括小数）
                    # 匹配 -1.5, -1, 1.5, 1 等格式
                    number_patterns = [
                        r'-?\d+\.\d+',  # 负数或正数小数，如 -1.5, 1.5
                        r'-?\d+',       # 负数或正数整数，如 -1, 1
                    ]

                    found_number = False
                    for pattern in number_patterns:
                        numbers = re.findall(pattern, text)
                        if numbers:
                            try:
                                number = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
                                if confidence > best_confidence:
                                    best_number = number
                                    best_confidence = confidence
                                    logger.debug(f"    识别到数字: {number} (模式: {pattern}, 置信度: {confidence})")
                                found_number = True
                                break  # 找到匹配就停止
                            except ValueError:
                                continue

                    # 模式2: 直接转换（处理纯数字文本，包括负号）
                    if not found_number:
                        try:
                            # 清理文本，只保留数字、小数点和负号
                            clean_text = re.sub(r'[^\d.\-]', '', text)
                            # 确保负号在开头
                            if '-' in clean_text:
                                # 移除所有负号，然后在开头加一个
                                clean_text = '-' + clean_text.replace('-', '')

                            if clean_text and clean_text not in ['.', '-', '-.']:
                                number = float(clean_text) if '.' in clean_text else int(clean_text)
                                if confidence > best_confidence:
                                    best_number = number
                                    best_confidence = confidence
                                    logger.debug(f"    清理后识别到数字: {number} (置信度: {confidence})")
                        except ValueError:
                            pass

            if best_number is not None:
                logger.info(f"    OCR成功识别数字: {best_number} (最佳置信度: {best_confidence})")
                return best_number
            else:
                logger.warning(f"    OCR未能识别出有效数字，共尝试 {len(results)} 个结果")
                return None

    except Exception as e:
        import traceback
        traceback.print_exc()   
        logger.error(f"OCR识别异常: {str(e)}")
        return None

def analyze_number_regions_with_vl_model(image, number_boxes):
    """使用VL大模型识别数字区域"""
    global dashscope_api_key, qwen_api_available

    if not qwen_api_available:
        return []

    recognized_numbers = []

    for i, number_box in enumerate(number_boxes):
        try:
            # 裁剪数字区域
            x1, y1, x2, y2 = map(int, number_box['box'])
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # 扩大裁剪区域以获得更好的上下文
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            number_region = image[y1:y2, x1:x2]

            if number_region.size == 0:
                continue

            # 转换为PIL图像并编码为base64
            pil_image = Image.fromarray(cv2.cvtColor(number_region, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            # 构建数字识别提示
            number_prompt = """请识别这个图片中的数字。这是从仪表盘上裁剪出来的数字区域。

请只回答数字本身，例如：
- 如果看到数字5，请回答：5
- 如果看到数字10，请回答：10
- 如果看到小数如2.5，请回答：2.5
- 如果无法识别，请回答：无法识别

注意：
1. 只回答数字，不要其他解释
2. 如果有多个数字，选择最清晰的一个
3. 如果是小数，保留小数点"""

            # 调用VL模型
            response = MultiModalConversation.call(
                model=QWEN_MODEL_NAME_OCR,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'image': f'data:image/jpeg;base64,{image_base64}'},
                        {'text': number_prompt}
                    ]
                }],
                max_tokens=50,
                temperature=0.1,       # 温度低，提高确定性
                top_p=0.3,              # top-p 值低，减少多样性
            )

            if response.status_code == 200:
                # 处理响应
                if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                    result_text = response.output.choices[0].message.content
                elif hasattr(response.output, 'text'):
                    result_text = response.output.text
                else:
                    result_text = str(response.output)

                # 处理响应文本（可能是列表）
                if isinstance(result_text, list):
                    if len(result_text) > 0:
                        result_text = str(result_text[0])
                    else:
                        result_text = ""
                else:
                    result_text = str(result_text)

                result_text = result_text.strip()
                logger.info(f"VL模型识别数字区域{i}: '{result_text}'")

                # 尝试提取数字（包括负数）
                import re

                # 预处理：替换常见的OCR错误字符
                result_text = result_text.replace('—', '-').replace('–', '-')  # 替换不同类型的破折号
                result_text = result_text.replace(',', '.')  # 替换逗号为小数点

                # 使用支持负数的正则表达式
                number_patterns = [
                    r'-?\d+\.\d+',  # 负数或正数小数，如 -1.5, 1.5
                    r'-?\d+',       # 负数或正数整数，如 -1, 1
                ]

                number_value = None
                for pattern in number_patterns:
                    numbers = re.findall(pattern, result_text)
                    if numbers:
                        try:
                            number_value = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
                            break
                        except ValueError:
                            continue

                if number_value is not None and result_text != '无法识别':
                    recognized_numbers.append({
                        'box_index': i,
                        'value': number_value,
                        'confidence': 'vl_model',
                        'center': number_box['center']
                    })
                    logger.info(f"VL模型成功识别数字: {number_value}")
                else:
                    logger.warning(f"VL模型无法识别数字区域{i}")
            else:
                logger.error(f"VL模型调用失败: {response.message}")

        except Exception as e:
            logger.error(f"VL模型识别数字区域{i}失败: {str(e)}")
            continue

    return recognized_numbers

def analyze_fire_smoke_with_vl_model(image):
    """使用VL大模型分析火灾和烟雾"""
    global dashscope_api_key, qwen_api_available

    if not qwen_api_available:
        return None

    try:
        # 将图像转换为base64
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image

        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # 构建火灾烟雾分析提示
        fire_smoke_prompt = """请分析这张图片是否存在火灾或烟雾。

请简洁回答：
检测结果: [有火灾/有烟雾/无异常]

分析要点：
1. 寻找明显的火焰（橙红色、跳动的光芒）
2. 寻找烟雾（灰色、白色或黑色的烟雾状物质）
3. 区分正常的蒸汽、雾气与危险的烟雾

如果不确定，请选择"无异常"。"""

        # 调用VL模型
        response = MultiModalConversation.call(
            model=QWEN_MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': [
                    {'image': f'data:image/jpeg;base64,{image_base64}'},
                    {'text': fire_smoke_prompt}
                ]
            }],
            max_tokens=500,
            temperature=0.1,       # 温度低，提高确定性
            top_p=0.3,              # top-p 值低，减少多样性
        )

        if response.status_code == 200:
            # 处理响应
            try:
                if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                    result_text = response.output.choices[0].message.content
                elif hasattr(response.output, 'text'):
                    result_text = response.output.text
                else:
                    result_text = str(response.output)

                logger.info(f"VL模型火灾烟雾分析: {result_text[:200]}...")

                # 解析VL模型的回答
                vl_result = parse_fire_smoke_vl_response(result_text)
                return vl_result
            except Exception as parse_error:
                logger.error(f"解析VL模型火灾烟雾响应失败: {str(parse_error)}")
                return None
        else:
            logger.error(f"VL模型火灾烟雾分析失败: {response.message}")
            return None

    except Exception as e:
        logger.error(f"VL模型火灾烟雾分析异常: {str(e)}")
        return None

def analyze_safety_harness_with_vl_model(image):
    """使用VL大模型分析安全带佩戴情况"""
    global dashscope_api_key, qwen_api_available

    if not qwen_api_available:
        return None

    try:
        # 将图像转换为base64
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image

        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # 构建安全带分析提示
        safety_harness_prompt = """请分析这张图片中工作人员的安全带佩戴情况。

请简洁回答：
检测结果: [已佩戴/未佩戴/无法确定]

分析要点：
1. 识别图片中的工作人员
2. 检查是否佩戴安全带（胸前和腰部的带子）
3. 安全带通常是鲜艳颜色（橙色、黄色、红色）

如果看不清楚，请选择"无法确定"。"""

        # 调用VL模型
        response = MultiModalConversation.call(
            model=QWEN_MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': [
                    {'image': f'data:image/jpeg;base64,{image_base64}'},
                    {'text': safety_harness_prompt}
                ]
            }],
            max_tokens=500,
            temperature=0.1,       # 温度低，提高确定性
            top_p=0.3,              # top-p 值低，减少多样性
        )

        if response.status_code == 200:
            # 处理响应
            try:
                if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                    result_text = response.output.choices[0].message.content
                elif hasattr(response.output, 'text'):
                    result_text = response.output.text
                else:
                    result_text = str(response.output)

                logger.info(f"VL模型安全带分析: {result_text[:200]}...")

                # 解析VL模型的回答
                vl_result = parse_safety_harness_vl_response(result_text)
                return vl_result
            except Exception as parse_error:
                logger.error(f"解析VL模型安全带响应失败: {str(parse_error)}")
                return None
        else:
            logger.error(f"VL模型安全带分析失败: {response.message}")
            return None

    except Exception as e:
        logger.error(f"VL模型安全带分析异常: {str(e)}")
        return None

def analyze_gauge_with_vl_model(image):
    """使用VL大模型分析整个仪表盘读数"""
    global dashscope_api_key, qwen_api_available

    if not qwen_api_available:
        return None

    try:
        # 将图像转换为base64
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image

        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # 构建仪表盘分析提示
        gauge_prompt = """请分析这个仪表盘图片，重点识别指针指向的数值范围。

                        请按照以下格式严格回答：
                        指针tip未识别成功，当前为大模型推测：
                        读数范围: [最小值]-[最大值]
                        当前指向: [具体数值]
                        置信度: [高/中/低]

                        分析步骤：
                        1. 首先找到仪表盘的指针（通常是一条细线或箭头）
                        2. 观察指针尖端（**细的一端**）周围最近的两个数字刻度
                        3. 根据指针的大体位置，给出该指向的数字

                        重要提示：
                        - 专注于指针的精确位置
                        - 优先识别指针最近的数字
                        - 观察指针细的一端，不要取粗的一端指向的数字
                        - 如果看不清指针，请说明
                        - 数值要准确，支持小数（如0.5, 1.2等）"""

        # 调用阿里云多模态大模型
        response = MultiModalConversation.call(
            model=QWEN_MODEL_NAME_DEEP,
            messages=[{
                'role': 'user',
                'content': [
                    {'image': f'data:image/jpeg;base64,{image_base64}'},
                    {'text': gauge_prompt}
                ]
            }],
            max_tokens=500,
            temperature=0.1,       # 温度低，提高确定性
            top_p=0.3,              # top-p 值低，减少多样性
            stream=True
        )

        # if response.status_code == 200:
        #     # 处理不同的响应格式
        #     try:
        #         if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
        #             result_text = response.output.choices[0].message.content
        #         elif hasattr(response.output, 'text'):
        #             result_text = response.output.text
        #         else:
        #             result_text = str(response.output)

        #         logger.info(f"VL模型原始响应: {result_text}")

        #         # 解析VL模型的回答
        #         vl_reading = parse_vl_gauge_response(result_text)
        #         return vl_reading
        #     except Exception as parse_error:
        #         logger.error(f"解析VL模型响应失败: {str(parse_error)}")
        #         logger.error(f"响应结构: {response.output}")
        #         return None
        # else:
        #     logger.error(f"VL模型调用失败: {response.message}")
        #     return None
        # 定义完整思考过程
        reasoning_content = ""
        # 定义完整回复
        answer_content = ""
        # 判断是否结束思考过程并开始回复
        is_answering = False

        print("=" * 20 + "思考过程" + "=" * 20)

        for chunk in response:
            # 如果思考过程与回复皆为空，则忽略
            message = chunk.output.choices[0].message
            reasoning_content_chunk = message.get("reasoning_content", None)
            if (chunk.output.choices[0].message.content == [] and
                reasoning_content_chunk == ""):
                pass
            else:
                # 如果当前为思考过程
                if reasoning_content_chunk != None and chunk.output.choices[0].message.content == []:
                    print(chunk.output.choices[0].message.reasoning_content, end="")
                    reasoning_content += chunk.output.choices[0].message.reasoning_content
                # 如果当前为回复
                elif chunk.output.choices[0].message.content != []:
                    if not is_answering:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                        is_answering = True
                    print(chunk.output.choices[0].message.content[0]["text"], end="")
                    answer_content += chunk.output.choices[0].message.content[0]["text"]
        logger.info(f"VL模型原始响应: {answer_content}")
        # 解析VL模型的回答
        vl_reading = parse_vl_gauge_response(answer_content)
        return vl_reading

    except Exception as e:
        logger.error(f"VL模型分析仪表盘失败: {str(e)}")
        return None

def analyze_safety_with_qwen3vl(image):
    """使用Qwen3-VL模型进行安全检测（烟雾、明火、跌倒、未系安全带、未戴安全帽）"""
    global qwen3vl_model, qwen3vl_processor, qwen3vl_model_loaded
    
    if not qwen3vl_model_loaded or qwen3vl_model is None or qwen3vl_processor is None:
        logger.warning("Qwen3-VL模型未加载，跳过多模态安全检测")
        return None
    
    try:
        # 将PIL图像转换为base64编码
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text", 
                        "text": """请仔细分析这张图片中的安全隐患，重点检测以下五种情况：
1. 烟雾（smoke）：是否有烟雾、烟气、雾状物质
2. 明火（fire）：是否有火焰、火光、燃烧现象
3. 跌倒（fall）：是否有人员跌倒、摔倒的情况
4. 未系安全带（no harness）：工作人员是否未佩戴或未正确佩戴安全带
5. 未戴安全帽（no helmet）：工作人员是否未佩戴或未正确佩戴安全帽

请按照以下JSON格式回答，只返回JSON，不要其他文字：
{
    "smoke": {"detected": true/false, "confidence": 0.0-1.0, "description": "描述"},
    "fire": {"detected": true/false, "confidence": 0.0-1.0, "description": "描述"},
    "fall": {"detected": true/false, "confidence": 0.0-1.0, "description": "描述"},
    "no_harness": {"detected": true/false, "confidence": 0.0-1.0, "description": "描述"},
    "no_helmet": {"detected": true/false, "confidence": 0.0-1.0, "description": "描述"}
}"""
                    },
                ],
            }
        ]
        
        # 准备输入
        inputs = qwen3vl_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(qwen3vl_model.device)
        
        # 生成回答
        with torch.no_grad():
            generated_ids = qwen3vl_model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = qwen3vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        logger.info(f"Qwen3-VL安全检测原始响应: {output_text}")
        
        # 解析JSON响应
        return parse_qwen3vl_safety_response(output_text)
        
    except Exception as e:
        logger.error(f"Qwen3-VL安全检测失败: {str(e)}")
        return None

def parse_qwen3vl_safety_response(response_text):
    """解析Qwen3-VL安全检测响应"""
    try:
        # 尝试提取JSON部分
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
            
            # 标准化结果格式
            safety_result = {
                'smoke': {
                    'detected': result.get('smoke', {}).get('detected', False),
                    'confidence': float(result.get('smoke', {}).get('confidence', 0.0)),
                    'description': result.get('smoke', {}).get('description', '')
                },
                'fire': {
                    'detected': result.get('fire', {}).get('detected', False),
                    'confidence': float(result.get('fire', {}).get('confidence', 0.0)),
                    'description': result.get('fire', {}).get('description', '')
                },
                'fall': {
                    'detected': result.get('fall', {}).get('detected', False),
                    'confidence': float(result.get('fall', {}).get('confidence', 0.0)),
                    'description': result.get('fall', {}).get('description', '')
                },
                'no_harness': {
                    'detected': result.get('no_harness', {}).get('detected', False),
                    'confidence': float(result.get('no_harness', {}).get('confidence', 0.0)),
                    'description': result.get('no_harness', {}).get('description', '')
                },
                'no_helmet': {
                    'detected': result.get('no_helmet', {}).get('detected', False),
                    'confidence': float(result.get('no_helmet', {}).get('confidence', 0.0)),
                    'description': result.get('no_helmet', {}).get('description', '')
                }
            }
            
            return safety_result
            
    except Exception as e:
        logger.error(f"解析Qwen3-VL安全检测响应失败: {str(e)}")
    
    # 如果解析失败，返回默认结果
    return {
        'smoke': {'detected': False, 'confidence': 0.0, 'description': '解析失败'},
        'fire': {'detected': False, 'confidence': 0.0, 'description': '解析失败'},
        'fall': {'detected': False, 'confidence': 0.0, 'description': '解析失败'},
        'no_harness': {'detected': False, 'confidence': 0.0, 'description': '解析失败'},
        'no_helmet': {'detected': False, 'confidence': 0.0, 'description': '解析失败'}
    }

def parse_fire_smoke_vl_response(response_text):
    """解析VL模型的火灾烟雾分析回答"""
    try:
        import re

        # 处理response_text可能是列表的情况
        if isinstance(response_text, list):
            if len(response_text) > 0:
                text = str(response_text[0])
            else:
                text = ""
        else:
            text = str(response_text)

        result = {
            'method': 'vl_model',
            'analysis_type': 'fire_smoke',
            'raw_response': text
        }

        # 只提取检测结果
        detection_patterns = [
            r'检测结果[：:]\s*\[([^\]]+)\]',
            r'检测结果[：:]\s*([^\n]+)',
        ]

        for pattern in detection_patterns:
            match = re.search(pattern, text)
            if match:
                result['detection_result'] = match.group(1).strip()
                break

        return result

    except Exception as e:
        logger.error(f"解析VL模型火灾烟雾回答失败: {str(e)}")
        return {
            'method': 'vl_model',
            'analysis_type': 'fire_smoke',
            'raw_response': str(response_text) if response_text else "无响应内容",
            'error': str(e)
        }

def parse_safety_harness_vl_response(response_text):
    """解析VL模型的安全带分析回答"""
    try:
        import re

        # 处理response_text可能是列表的情况
        if isinstance(response_text, list):
            if len(response_text) > 0:
                text = str(response_text[0])
            else:
                text = ""
        else:
            text = str(response_text)

        result = {
            'method': 'vl_model',
            'analysis_type': 'safety_harness',
            'raw_response': text
        }

        # 只提取检测结果
        detection_patterns = [
            r'检测结果[：:]\s*\[([^\]]+)\]',
            r'检测结果[：:]\s*([^\n]+)',
        ]

        for pattern in detection_patterns:
            match = re.search(pattern, text)
            if match:
                result['detection_result'] = match.group(1).strip()
                break

        return result

    except Exception as e:
        logger.error(f"解析VL模型安全带回答失败: {str(e)}")
        return {
            'method': 'vl_model',
            'analysis_type': 'safety_harness',
            'raw_response': str(response_text) if response_text else "无响应内容",
            'error': str(e)
        }

def parse_vl_gauge_response(response_text):
    """解析VL模型的仪表盘分析回答"""
    try:
        import re

        # 处理response_text可能是列表的情况
        if isinstance(response_text, list):
            if len(response_text) > 0:
                text = str(response_text[0])
            else:
                text = ""
        else:
            text = str(response_text)

        logger.info(f"解析VL模型响应文本: {text[:200]}...")  # 只显示前200字符

        result = {
            'method': 'vl_model',
            'raw_response': text,
            'confidence_level': '未知'
        }

        # 多种模式提取读数范围
        range_patterns = [
            r'读数范围[：:]\s*\[(\d+(?:\.\d+)?)\]\s*[-~]\s*\[(\d+(?:\.\d+)?)\]',  # [0]-[6]格式
            r'读数范围[：:]\s*(\d+(?:\.\d+)?)\s*[-~到]\s*(\d+(?:\.\d+)?)',        # 0-6格式
            r'读数范围[：:]\s*\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]',              # [0-6]格式
            r'范围.*?(\d+(?:\.\d+)?)\s*[-~到]\s*(\d+(?:\.\d+)?)',                # 范围0到6格式
        ]

        range_found = False
        for pattern in range_patterns:
            range_match = re.search(pattern, text)
            if range_match:
                min_val = float(range_match.group(1))
                max_val = float(range_match.group(2))
                result.update({
                    'reading_range': f"{min_val}~{max_val}",
                    'min_value': min_val,
                    'max_value': max_val
                })
                range_found = True
                logger.info(f"成功提取读数范围: {min_val}~{max_val}")
                break

        # 多种模式提取当前指向
        current_patterns = [
            r'当前指向[：:]\s*\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]',              # [0.2-0.3]格式
            r'当前指向[：:]\s*(\d+(?:\.\d+)?)\s*[-~到]\s*(\d+(?:\.\d+)?)',        # 0.2-0.3格式
            r'当前指向[：:]\s*(\d+(?:\.\d+)?)',                                  # 单个数值
            r'指针.*?(\d+(?:\.\d+)?)\s*[-~到]\s*(\d+(?:\.\d+)?)',                # 指针在0.2到0.3
            r'约.*?(\d+(?:\.\d+)?)',                                            # 约0.2
        ]

        for pattern in current_patterns:
            current_match = re.search(pattern, text)
            if current_match:
                if len(current_match.groups()) >= 2 and current_match.group(2):
                    # 范围格式
                    min_current = float(current_match.group(1))
                    max_current = float(current_match.group(2))
                    result['current_value'] = f"{min_current}~{max_current}"
                    logger.info(f"成功提取当前指向范围: {min_current}~{max_current}")
                else:
                    # 单个数值
                    result['current_value'] = float(current_match.group(1))
                    logger.info(f"成功提取当前指向: {current_match.group(1)}")
                break

        # 提取置信度
        confidence_patterns = [
            r'置信度[：:]\s*\[(\w+)\]',     # [高]格式
            r'置信度[：:]\s*(\w+)',         # 高格式
            r'置信度.*?(\w+)',              # 置信度为高
        ]

        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, text)
            if confidence_match:
                result['confidence_level'] = confidence_match.group(1)
                logger.info(f"成功提取置信度: {confidence_match.group(1)}")
                break

        # 如果没有找到标准格式，尝试提取数字
        if not range_found:
            # 尝试提取所有数字，找到可能的范围
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if len(numbers) >= 2:
                # 取前两个数字作为范围
                try:
                    min_val = float(numbers[0])
                    max_val = float(numbers[1])
                    if min_val != max_val:  # 确保不是同一个数字
                        result.update({
                            'reading_range': f"{min_val}~{max_val}",
                            'min_value': min_val,
                            'max_value': max_val
                        })
                        logger.info(f"从数字中推断读数范围: {min_val}~{max_val}")
                except ValueError:
                    pass

        return result

    except Exception as e:
        logger.error(f"解析VL模型回答失败: {str(e)}")
        # 安全处理response_text
        safe_response = str(response_text) if response_text else "无响应内容"
        return {
            'method': 'vl_model',
            'raw_response': safe_response,
            'error': str(e)
        }

def crop_image_by_circle_plate(image, circle_plate_box, padding=20):
    """根据circle_plate边界框裁剪图片"""
    x1, y1, x2, y2 = circle_plate_box

    # 添加padding
    x1 = max(0, int(x1 - padding))
    y1 = max(0, int(y1 - padding))
    x2 = min(image.shape[1], int(x2 + padding))
    y2 = min(image.shape[0], int(y2 + padding))

    # 裁剪图片
    cropped_image = image[y1:y2, x1:x2]

    # 返回裁剪后的图片和偏移量
    crop_offset = (x1, y1)

    return cropped_image, crop_offset

def convert_cropped_to_original_coords(cropped_xyxy, crop_offset):
    """将裁剪图片中的坐标转换回原图坐标"""
    x_offset, y_offset = crop_offset
    x1, y1, x2, y2 = cropped_xyxy

    return [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]

def process_pressure_gauge_detection(image, results, annotated_image):
    """处理仪表盘检测结果，使用两步检测法"""
    global ocr_reader, models

    # 获取压力表模型的类别配置
    pressure_classes = MODEL_CONFIGS['pressure_gauge']['classes']
    logger.info(f"压力表类别配置: {pressure_classes}")

    # 找到各类别的ID
    tip_class_id = None
    number_class_id = None

    for i, class_name in enumerate(pressure_classes):
        if class_name == 'tip':
            tip_class_id = i
        elif class_name == 'number':
            number_class_id = i

    logger.info(f"tip类别ID: {tip_class_id}, number类别ID: {number_class_id}")

    # 第一步：从原始检测结果中寻找circle_plate
    logger.info("第一步：从检测结果中寻找circle_plate...")
    circle_plates = []
    all_detections = []
    tip_boxes = []
    number_boxes = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                class_name = pressure_classes[class_id] if class_id < len(pressure_classes) else 'unknown'

                all_detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence_score,
                    'box': xyxy
                })

                # 寻找circle_plate
                if class_name == 'circle_plate':
                    circle_plates.append({
                        'box': xyxy,
                        'confidence': confidence_score,
                        'center': get_box_center(xyxy)
                    })
                elif class_name == tip_class_id:
                    tip_boxes.append({
                        'box': xyxy,
                        'confidence': confidence_score,
                        'center': get_box_center(xyxy)
                    })
                elif class_id == number_class_id:  # number
                    number_boxes.append({
                        'box': xyxy,
                        'confidence': confidence_score,
                        'center': get_box_center(xyxy)
                    })
            

    logger.info(f"第一步完成，找到 {len(circle_plates)} 个circle_plate, {len(tip_boxes)} 个tip")

    # 第二步：基于circle_plate裁剪图片并二次检测

    if len(tip_boxes) == 0:
        # 选择置信度最高的circle_plate
        best_circle_plate = max(circle_plates, key=lambda x: x['confidence'])
        logger.info(f"选择最佳circle_plate，置信度: {best_circle_plate['confidence']:.3f}")

        # 裁剪图片
        cropped_image, crop_offset = crop_image_by_circle_plate(image, best_circle_plate['box'])
        logger.info(f"第二步：基于circle_plate裁剪图片，裁剪区域大小: {cropped_image.shape}, 偏移量: {crop_offset}")

        # 在裁剪后的图片上进行二次检测
        logger.info("在裁剪图片上进行二次检测...")
        pressure_model = models.get('pressure_gauge')
        if pressure_model is None:
            logger.error("仪表盘模型未加载")
            return []

        second_results = pressure_model(cropped_image, conf=0.1)  # 使用较低的置信度阈值

        # 解析第二步检测结果
        # second_detections = []
        for result in second_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence_score = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    class_name = pressure_classes[class_id] if class_id < len(pressure_classes) else 'unknown'

                    # 跳过circle_plate，因为我们已经有了
                    if class_name == 'circle_plate':
                        continue

                    # 将坐标转换回原图坐标系
                    original_xyxy = convert_cropped_to_original_coords(xyxy, crop_offset)

                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence_score,
                        'box': original_xyxy
                    }

                    # all_detections.append(detection)
                    # second_detections.append(detection)

                    # 根据类别ID分类
                    if class_id == tip_class_id:  # tip
                        tip_boxes.append({
                            'box': original_xyxy,
                            'confidence': confidence_score,
                            'center': get_box_center(original_xyxy)
                        })
                        all_detections.append(detection)

                        color = (255, 0, 0)  # 红色用于指针

                        # 在图片上绘制边界框（确保坐标为整数）
                        x1, y1, x2, y2 = map(int, original_xyxy)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                        # 添加标签
                        label = f'{class_name}: {confidence_score:.2f}'
                        cv2.putText(annotated_image, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    

        logger.info(f"第二步完成，在裁剪图片中检测后，共检测到 {len(all_detections)} 个组件")

        # 保存裁剪后的图片用于调试
        # import cv2
        # cv2.imwrite('debug_cropped_gauge.jpg', cropped_image)
        # logger.info("已保存裁剪后的图片: debug_cropped_gauge.jpg")

    else:
        logger.info("tip已经找到，不用二次检测")

    logger.info(f"检测到的所有目标: {[(d['class_name'], d['confidence']) for d in all_detections]}")
    logger.info(f"找到 {len(tip_boxes)} 个tip, {len(number_boxes)} 个number")

    gauge_readings = []

    # 检查是否检测到tip和number
    if not tip_boxes:
        logger.warning("未检测到tip（指针），无法进行基于距离的分析")
        logger.info("直接使用VL大模型进行整体仪表盘分析...")

        # 直接使用VL大模型分析整个仪表盘
        vl_result = analyze_gauge_with_vl_model(image)
        if vl_result:
            gauge_readings.append(vl_result)
            logger.info("VL大模型成功分析仪表盘，无需tip定位")
        else:
            logger.error("VL大模型也无法分析仪表盘")

    elif not number_boxes:
        logger.warning("未检测到number，无法进行YOLO+OCR分析")
        logger.info("直接使用VL大模型进行整体仪表盘分析...")

        # 直接使用VL大模型分析整个仪表盘
        vl_result = analyze_gauge_with_vl_model(image)
        if vl_result:
            gauge_readings.append(vl_result)
            logger.info("VL大模型成功分析仪表盘，无需number定位")
        else:
            logger.error("VL大模型也无法分析仪表盘")

    else:
        # 首先选择要分析的tip（无论使用哪种算法都需要）
        if len(tip_boxes) > 1:
            logger.warning(f"检测到{len(tip_boxes)}个tip，通常一个仪表盘只有一个指针")
            # 按置信度排序，选择置信度最高的tip
            tip_boxes_sorted = sorted(tip_boxes, key=lambda x: x['confidence'], reverse=True)
            selected_tip = tip_boxes_sorted[0]
            logger.info(f"选择置信度最高的tip进行分析: 置信度{selected_tip['confidence']:.3f}")
        else:
            selected_tip = tip_boxes[0]
            logger.info(f"检测到1个tip，置信度: {selected_tip['confidence']:.3f}")

        # 尝试使用精确读数算法
        precise_reading_success = False
        try:
            from precise_gauge_reader import get_precise_gauge_reader

            # 转换检测结果为精确读数器需要的格式，并进行数据验证
            precise_detections = []
            valid_tips = []
            valid_numbers = []
            valid_bases = []

            for detection in all_detections:
                class_name = detection['class_name']
                confidence = detection['confidence']
                box = detection['box'] if isinstance(detection['box'], list) else detection['box'].tolist()

                # 数据验证：确保边界框合理
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                    # 检查边界框是否在图像范围内
                    if (0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and
                        0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0] and
                        x2 > x1 and y2 > y1):

                        detection_data = {
                            'class': class_name,
                            'confidence': confidence,
                            'box': box
                        }
                        precise_detections.append(detection_data)

                        # 分类收集高质量组件
                        if class_name == 'tip' and confidence > 0.1:
                            valid_tips.append(detection_data)
                        elif class_name == 'number' and confidence > 0.3:
                            valid_numbers.append(detection_data)
                        elif class_name == 'base' and confidence > 0.3:
                            valid_bases.append(detection_data)

            logger.info(f"数据验证后: 有效tip={len(valid_tips)}, 有效number={len(valid_numbers)}, 有效base={len(valid_bases)}")

            # 增强的精确读数逻辑
            precise_reader = get_precise_gauge_reader()

            # 移除预估范围逻辑，直接使用增强版精确读数
            logger.info("使用增强版精确读数算法（已移除预估范围验证）")

            precise_result = precise_reader.extract_precise_reading_enhanced(
                precise_detections, image, ocr_reader)

            if precise_result['success']:
                logger.info(f"精确读数算法成功: {precise_result['reading']} (方法: {precise_result['method']})")

                # 为前端兼容性，生成读数范围信息
                reading_value = precise_result['reading']

                # 根据精确读数生成合理的范围
                if reading_value == int(reading_value):
                    # 整数读数
                    min_val = max(0, int(reading_value) - 1)
                    max_val = int(reading_value) + 1
                    reading_range = f"{min_val}~{max_val}"
                else:
                    # 小数读数，取整数部分作为范围
                    base_val = int(reading_value)
                    min_val = base_val
                    max_val = base_val + 1
                    reading_range = f"{min_val}~{max_val}"

                gauge_readings.append({
                    # 精确读数结果放在第一位
                    'precise_reading': precise_result['reading'],
                    'precise_confidence': precise_result['confidence'],
                    'precise_method': precise_result.get('method', 'circular_scale'),

                    # 传统字段保持兼容性
                    'reading': precise_result['reading'],
                    'confidence': precise_result['confidence'],
                    'method': 'precise_yolo_based',

                    # 范围信息
                    'reading_range': reading_range,
                    'min_value': float(min_val),
                    'max_value': float(max_val),

                    # 位置信息
                    'tip_position': [float(selected_tip['center'][0]), float(selected_tip['center'][1])],

                    # 详细信息
                    'recognition_details': f"精确读数: {reading_value:.3f} (方法: {precise_result.get('method', 'circular_scale')}, 置信度: {precise_result['confidence']:.3f}, 范围: {reading_range})",
                    'details': precise_result.get('details', {})
                })
                precise_reading_success = True
            else:
                logger.warning(f"精确读数算法失败: {precise_result['error']}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"精确读数算法异常: {e}")

        # 如果精确读数失败，使用传统的范围识别方法
        if not precise_reading_success:
            logger.info("使用传统范围识别方法...")

            # 只处理选中的tip
            tip_center = selected_tip['center']
            logger.info(f"分析指针位置: {tip_center}")

            # 计算所有number到tip的距离
            distances = []
            for i, number in enumerate(number_boxes):
                distance = calculate_distance(tip_center, number['center'])
                distances.append((distance, i, number))
                logger.debug(f"  number {i} 距离: {distance:.2f}")

            # 按距离排序
            distances.sort(key=lambda x: x[0])
            logger.info(f"  最近的number距离排序: {[d[0] for d in distances[:3]]}")

            # 严格只处理最近的两个number
            if len(distances) < 2:
                logger.warning(f"  只检测到{len(distances)}个number，无法确定范围")
                logger.info("  尝试使用VL大模型进行整体分析...")
                vl_result = analyze_gauge_with_vl_model(image)
                if vl_result:
                    gauge_readings.append(vl_result)
            else:
                # 获取最近的两个number
                closest_two = distances[:2]
                logger.info(f"  选择最近的两个number: 距离{closest_two[0][0]:.2f}和{closest_two[1][0]:.2f}")

                # 首先尝试OCR识别这两个数字
                numbers = []
                ocr_attempts = []

                for i, (distance, number_idx, number_info) in enumerate(closest_two):
                    logger.info(f"  尝试OCR识别第{i+1}个最近的number (索引{number_idx}, 距离{distance:.2f})")

                    extracted_number = extract_number_from_box(image, number_info['box'], ocr_reader)
                    ocr_attempts.append({
                        'number_idx': int(number_idx),
                        'distance': float(distance),
                        'extracted': int(extracted_number) if extracted_number is not None else None
                    })

                    if extracted_number is not None:
                        numbers.append(extracted_number)
                        logger.info(f"    OCR成功识别数字: {extracted_number}")
                    else:
                        logger.warning(f"    OCR识别失败")

                logger.info(f"  OCR识别结果: 成功{len(numbers)}/2个数字")

                # 如果OCR无法识别这两个数字，使用VL模型
                if len(numbers) < 2:
                    logger.info(f"  OCR只识别出{len(numbers)}个数字，使用VL模型识别最近的两个数字区域...")

                    # 使用VL模型识别这两个数字区域
                    closest_two_boxes = [info[2] for info in closest_two]  # 提取number_info
                    vl_numbers = analyze_number_regions_with_vl_model(image, closest_two_boxes)

                    # 将VL识别的数字按距离顺序添加到numbers中
                    for vl_num in vl_numbers:
                        box_idx = vl_num['box_index']
                        if box_idx < len(closest_two) and ocr_attempts[box_idx]['extracted'] is None:
                            # 只有当OCR失败时才使用VL结果
                            numbers.append(vl_num['value'])
                            ocr_attempts[box_idx]['extracted'] = vl_num['value']
                            ocr_attempts[box_idx]['method'] = 'vl_model'
                            logger.info(f"    VL模型成功识别数字: {vl_num['value']}")

                    logger.info(f"  最终识别结果: 成功{len(numbers)}/2个数字")

                # 基于最近的两个数字生成读数范围
                if len(numbers) >= 2:
                    # 有2个数字，计算范围（从小到大）
                    sorted_numbers = sorted(numbers)
                    min_val = sorted_numbers[0]
                    max_val = sorted_numbers[1]

                    # 格式化数字显示（保持原始精度）
                    if isinstance(min_val, float) and min_val.is_integer():
                        min_str = str(int(min_val))
                    else:
                        min_str = str(min_val)

                    if isinstance(max_val, float) and max_val.is_integer():
                        max_str = str(int(max_val))
                    else:
                        max_str = str(max_val)

                    reading_range = f"{min_str}~{max_str}"

                    # 确定识别方法
                    method = 'yolo_ocr'
                    if any(attempt.get('method') == 'vl_model' for attempt in ocr_attempts):
                        method = 'yolo_ocr_vl'  # 混合方法

                    gauge_readings.append({
                        'method': method,
                        'tip_position': [float(tip_center[0]), float(tip_center[1])],
                        'closest_numbers': [float(min_val), float(max_val)],  # 严格只有最近的两个数字
                        'reading_range': reading_range,
                        'min_value': float(min_val),
                        'max_value': float(max_val),
                        'distance_info': {
                            'closest_distance': float(closest_two[0][0]),
                            'second_closest_distance': float(closest_two[1][0])
                        },
                        'recognition_details': f"基于离指针最近的两个数字: {min_str}和{max_str}"
                    })
                    logger.info(f"  成功创建读数范围: {reading_range} (方法: {method})")

                elif len(numbers) == 1:
                    # 只识别出1个数字，无法确定范围
                    single_val = numbers[0]
                    if single_val == int(single_val):
                        single_str = str(int(single_val))
                    else:
                        single_str = str(single_val)

                    logger.warning(f"  只识别出1个数字({single_str})，无法确定指针范围")

                else:
                    logger.warning(f"  指针: 最近的两个数字都无法识别")

    logger.info(f"YOLO+OCR分析完成，共生成 {len(gauge_readings)} 个读数")

    # 如果YOLO+OCR没有成功识别出读数，使用VL大模型作为备用方案
    if not gauge_readings:
        logger.info("YOLO+OCR未能识别仪表盘读数，尝试使用VL大模型...")
        vl_result = analyze_gauge_with_vl_model(image)
        if vl_result:
            gauge_readings.append(vl_result)
            logger.info("VL大模型成功分析仪表盘读数")
        else:
            logger.warning("VL大模型也无法分析仪表盘读数")

    # 确保所有返回的数据都是JSON可序列化的
    return convert_to_python_types(gauge_readings)

def init_dashscope_api():
    """初始化阿里云DashScope API"""
    global dashscope_api_key, qwen_api_available

    try:
        # 检查API密钥
        if not DASHSCOPE_API_KEY:
            logger.warning("未设置DASHSCOPE_API_KEY环境变量")
            qwen_api_available = False
            return False

        # 设置API密钥
        import dashscope
        dashscope.api_key = DASHSCOPE_API_KEY
        dashscope_api_key = DASHSCOPE_API_KEY

        # 测试API连接
        logger.info("测试阿里云DashScope API连接...")
        test_response = MultiModalConversation.call(
            model=QWEN_MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': [{'text': '你好'}]
            }],
            max_tokens=10
        )

        if test_response.status_code == 200:
            qwen_api_available = True
            logger.info("阿里云DashScope API初始化成功!")
            return True
        else:
            logger.error(f"API测试失败: {test_response}")
            qwen_api_available = False
            return False

    except Exception as e:
        logger.error(f"阿里云DashScope API初始化失败: {str(e)}")
        qwen_api_available = False
        return False

def classify_disaster_image(image_data):
    """使用阿里云API对图片进行灾情分类"""
    if not qwen_api_available:
        return {
            'success': False,
            'error': '阿里云API未初始化或不可用'
        }

    try:
        # 处理图片数据并转换为base64
        image_base64 = None

        if isinstance(image_data, str):
            # 如果是base64编码的图片
            if image_data.startswith('data:image'):
                image_base64 = image_data
            else:
                # 纯base64数据，添加前缀
                image_base64 = f"data:image/jpeg;base64,{image_data}"
        else:
            # 如果是文件对象，转换为base64
            image = Image.open(image_data)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()
            image_base64_str = base64.b64encode(image_bytes).decode()
            image_base64 = f"data:image/jpeg;base64,{image_base64_str}"

        # 构建灾情分析的提示词
        disaster_prompt = """请仔细分析这张图片中的灾情情况，并提供详细的分析报告。请按照以下格式回答：

1. 灾情类型：[火灾/烟雾/洪水/地震/台风/其他/无灾情]
2. 严重程度：[轻微/中等/严重/极严重]
3. 影响范围：[描述受影响的区域和范围]
4. 紧急程度：[低/中/高/极高]
5. 建议措施：[具体的应急处理建议]
6. 详细描述：[对图片中灾情的详细描述]

请基于图片内容进行客观、专业的分析。"""

        # 调用阿里云API
        response = MultiModalConversation.call(
            model=QWEN_MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': [
                    {'image': image_base64},
                    {'text': disaster_prompt}
                ]
            }],
            max_tokens=1000,
            temperature=0.1,       # 温度低，提高确定性
            top_p=0.3,              # top-p 值低，减少多样性
        )

        if response.status_code == 200:
            output_content = response.output.choices[0].message.content
            logger.info(f"图片API返回内容类型: {type(output_content)}")
            logger.info(f"图片API返回内容: {output_content}")

            # 处理API返回的内容格式
            if isinstance(output_content, list):
                # 如果返回的是列表，提取文本内容
                output_text = ""
                for item in output_content:
                    if isinstance(item, dict) and 'text' in item:
                        output_text += item['text'] + "\n"
                    elif isinstance(item, str):
                        output_text += item + "\n"
                    else:
                        output_text += str(item) + "\n"
            else:
                output_text = str(output_content)

            # 解析输出结果
            analysis_result = parse_disaster_analysis(output_text)

            return {
                'success': True,
                'analysis': analysis_result,
                'raw_output': output_text
            }
        else:
            logger.error(f"阿里云API调用失败: {response}")
            return {
                'success': False,
                'error': f'API调用失败: {response.message}'
            }

    except Exception as e:
        logger.error(f"灾情分类错误: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def parse_disaster_analysis(output_text):
    """解析灾情分析结果"""
    try:
        # 确保输入是字符串
        if not isinstance(output_text, str):
            output_text = str(output_text)

        analysis = {
            'disaster_type': '未知',
            'severity': '未知',
            'affected_area': '未知',
            'urgency': '未知',
            'recommendations': '未知',
            'description': '未知'
        }

        lines = output_text.split('\n')

        for line in lines:
            line = line.strip()
            # 处理多种格式的标识符
            if any(keyword in line for keyword in ['灾情类型：', '灾情类型:', '**灾情类型**', '1. 灾情类型', '1.**灾情类型**']):
                # 提取冒号或其他分隔符后的内容
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['disaster_type'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['严重程度：', '严重程度:', '**严重程度**', '2. 严重程度', '2.**严重程度**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['severity'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['影响范围：', '影响范围:', '**影响范围**', '3. 影响范围', '3.**影响范围**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['affected_area'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['紧急程度：', '紧急程度:', '**紧急程度**', '4. 紧急程度', '4.**紧急程度**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['urgency'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['建议措施：', '建议措施:', '**建议措施**', '5. 建议措施', '5.**建议措施**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['recommendations'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['详细描述：', '详细描述:', '**详细描述**', '6. 详细描述', '6.**详细描述**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['description'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break

        return analysis

    except Exception as e:
        logger.error(f"解析灾情分析结果失败: {str(e)}")
        return {
            'disaster_type': '解析失败',
            'severity': '解析失败',
            'affected_area': '解析失败',
            'urgency': '解析失败',
            'recommendations': '解析失败',
            'description': '解析失败'
        }

def classify_disaster_video(video_path, sample_frames=5):
    """使用Qwen2.5-VL对视频进行灾情分类"""
    if not qwen_api_available:
        return {
            'success': False,
            'error': '阿里云API未初始化或不可用'
        }

    try:
        # 从视频中提取关键帧
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': '无法打开视频文件'}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # 计算采样帧的位置
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        frame_images = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # 转换为base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
                image_base64_str = base64.b64encode(image_bytes).decode()
                image_base64 = f"data:image/jpeg;base64,{image_base64_str}"
                frame_images.append(image_base64)

        cap.release()

        if not frame_images:
            return {'success': False, 'error': '无法从视频中提取帧'}

        # 构建视频灾情分析的提示词
        video_prompt = f"""请仔细分析这个视频中的灾情情况。我提供了{len(frame_images)}个关键帧，请基于这些帧进行综合分析，并提供详细的分析报告。

请按照以下格式回答：

1. 灾情类型：[火灾/烟雾/洪水/地震/台风/其他/无灾情]
2. 严重程度：[轻微/中等/严重/极严重]
3. 发展趋势：[恶化/稳定/好转]
4. 影响范围：[描述受影响的区域和范围]
5. 紧急程度：[低/中/高/极高]
6. 建议措施：[具体的应急处理建议]
7. 时间特征：[描述灾情在时间上的变化]
8. 详细描述：[对视频中灾情的详细描述]

请基于视频内容进行客观、专业的分析。"""

        # 准备API调用的内容
        content = []
        for image_base64 in frame_images:
            content.append({'image': image_base64})
        content.append({'text': video_prompt})

        # 调用阿里云API
        response = MultiModalConversation.call(
            model=QWEN_MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': content
            }],
            max_tokens=1000,
            temperature=0.1,       # 温度低，提高确定性
            top_p=0.3,              # top-p 值低，减少多样性
        )

        if response.status_code == 200:
            output_content = response.output.choices[0].message.content
            logger.info(f"视频API返回内容类型: {type(output_content)}")
            logger.info(f"视频API返回内容: {output_content}")

            # 处理API返回的内容格式
            if isinstance(output_content, list):
                # 如果返回的是列表，提取文本内容
                output_text = ""
                for item in output_content:
                    if isinstance(item, dict) and 'text' in item:
                        output_text += item['text'] + "\n"
                    elif isinstance(item, str):
                        output_text += item + "\n"
                    else:
                        output_text += str(item) + "\n"
            else:
                output_text = str(output_content)

            # 解析输出结果
            analysis_result = parse_video_disaster_analysis(output_text)

            return {
                'success': True,
                'analysis': analysis_result,
                'raw_output': output_text,
                'video_info': {
                    'duration': duration,
                    'total_frames': total_frames,
                    'fps': fps,
                    'analyzed_frames': len(frame_images)
                }
            }
        else:
            logger.error(f"阿里云API调用失败: {response}")
            return {
                'success': False,
                'error': f'API调用失败: {response.message}'
            }

    except Exception as e:
        logger.error(f"视频灾情分类错误: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def parse_video_disaster_analysis(output_text):
    """解析视频灾情分析结果"""
    try:
        # 确保输入是字符串
        if not isinstance(output_text, str):
            output_text = str(output_text)

        analysis = {
            'disaster_type': '未知',
            'severity': '未知',
            'trend': '未知',
            'affected_area': '未知',
            'urgency': '未知',
            'recommendations': '未知',
            'time_characteristics': '未知',
            'description': '未知'
        }

        lines = output_text.split('\n')

        for line in lines:
            line = line.strip()
            # 处理多种格式的标识符
            if any(keyword in line for keyword in ['灾情类型：', '灾情类型:', '**灾情类型**', '1. 灾情类型', '1.**灾情类型**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['disaster_type'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['严重程度：', '严重程度:', '**严重程度**', '2. 严重程度', '2.**严重程度**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['severity'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['发展趋势：', '发展趋势:', '**发展趋势**', '3. 发展趋势', '3.**发展趋势**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['trend'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['影响范围：', '影响范围:', '**影响范围**', '4. 影响范围', '4.**影响范围**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['affected_area'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['紧急程度：', '紧急程度:', '**紧急程度**', '5. 紧急程度', '5.**紧急程度**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['urgency'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['建议措施：', '建议措施:', '**建议措施**', '6. 建议措施', '6.**建议措施**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['recommendations'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['时间特征：', '时间特征:', '**时间特征**', '7. 时间特征', '7.**时间特征**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['time_characteristics'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break
            elif any(keyword in line for keyword in ['详细描述：', '详细描述:', '**详细描述**', '8. 详细描述', '8.**详细描述**']):
                for sep in ['：', ':', '**：', '**:', '】', ']']:
                    if sep in line:
                        analysis['description'] = line.split(sep)[-1].strip().replace('*', '').replace('【', '').replace('】', '')
                        break

        return analysis

    except Exception as e:
        logger.error(f"解析视频灾情分析结果失败: {str(e)}")
        return {
            'disaster_type': '解析失败',
            'severity': '解析失败',
            'trend': '解析失败',
            'affected_area': '解析失败',
            'urgency': '解析失败',
            'recommendations': '解析失败',
            'time_characteristics': '解析失败',
            'description': '解析失败'
        }

# def process_image(image_data, confidence_threshold=0.5, model_type='fire_smoke'):
#     """处理图片并进行检测"""
#     try:
#         # 获取对应的模型
#         if model_type not in models:
#             raise ValueError(f"模型类型 {model_type} 不存在")

#         current_model = models[model_type]
#         model_config = MODEL_CONFIGS[model_type]

#         logger.info(f"🤖 正在使用模型: {model_config['name']} (路径: {model_config['path']})")

#         # 将base64图片数据转换为numpy数组
#         if isinstance(image_data, str):
#             # 如果是base64编码的图片
#             if image_data.startswith('data:image'):
#                 image_data = image_data.split(',')[1]

#             image_bytes = base64.b64decode(image_data)
#             image = Image.open(io.BytesIO(image_bytes))
#         else:
#             # 如果是文件对象
#             image = Image.open(image_data)

#         # 转换为RGB格式
#         if image.mode != 'RGB':
#             image = image.convert('RGB')

#         # 转换为numpy数组
#         img_array = np.array(image)

#         # 使用模型进行预测
#         logger.info(f"🔍 开始模型推理，图像尺寸: {img_array.shape}, 置信度阈值: {confidence_threshold}")
#         results = current_model(img_array, conf=confidence_threshold)
#         logger.info(f"✅ 模型推理完成")
        
#         # 处理检测结果
#         detections = []
#         annotated_image = img_array.copy()

#         # 获取类别名称列表
#         class_names = model_config['classes']

#         for result in results:
#             boxes = result.boxes
#             if boxes is not None:
#                 for box in boxes:
#                     # 获取边界框坐标
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                     confidence = box.conf[0].cpu().numpy()
#                     class_id = int(box.cls[0].cpu().numpy())

#                     # 获取类别名称
#                     class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'

#                     # 添加检测结果
#                     detections.append({
#                         'class': class_name,
#                         'confidence': float(confidence),
#                         'bbox': [float(x1), float(y1), float(x2), float(y2)]
#                     })

#                     # 根据模型类型选择颜色
#                     if model_type == 'fire_smoke':
#                         color = (0, 255, 0) if 'smoke' in class_name.lower() else (0, 0, 255)
#                     elif model_type == 'safety_harness':
#                         color = (0, 255, 0) if 'safe' in class_name.lower() else (255, 0, 0)
#                     else:
#                         color = (255, 255, 0)  # 黄色用于其他类型

#                     # 在图片上绘制边界框
#                     cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

#                     # 添加标签
#                     label = f'{class_name}: {confidence:.2f}'
#                     cv2.putText(annotated_image, label, (int(x1), int(y1-10)),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         # 如果没有检测到任何目标，使用VL大模型进行分析
#         vl_analysis = None
#         if len(detections) == 0:
#             logger.warning(f"未检测到任何{model_config['name']}相关目标")
#             logger.info("尝试使用VL大模型进行分析...")

#             if model_type == 'fire_smoke':
#                 vl_analysis = analyze_fire_smoke_with_vl_model(img_array)
#             elif model_type == 'safety_harness':
#                 vl_analysis = analyze_safety_harness_with_vl_model(img_array)

#             if vl_analysis:
#                 logger.info(f"VL大模型成功分析{model_config['name']}场景")
#             else:
#                 logger.warning(f"VL大模型也无法分析{model_config['name']}场景")

#         # 将结果图片转换为base64
#         annotated_pil = Image.fromarray(annotated_image)
#         buffer = io.BytesIO()
#         annotated_pil.save(buffer, format='JPEG')
#         annotated_base64 = base64.b64encode(buffer.getvalue()).decode()

#         result = {
#             'success': True,
#             'detections': detections,
#             'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
#             'detection_count': len(detections),
#             'model_type': model_type,
#             'model_name': model_config['name']
#         }

#         # 如果有VL分析结果，添加到返回结果中
#         if vl_analysis:
#             result['vl_analysis'] = vl_analysis

#         # 确保所有数据都是JSON可序列化的
#         return convert_to_python_types(result)

#     except Exception as e:
#         logger.error(f"图片处理错误: {str(e)}")
#         return {
#             'success': False,
#             'error': str(e)
#         }

def process_image(image_data, confidence_threshold=0.5, model_type='fire_smoke'):
    """处理图片并进行检测"""
    try:
        # 获取对应的模型
        if model_type not in models:
            raise ValueError(f"模型类型 {model_type} 不存在")

        model_config = MODEL_CONFIGS[model_type]
        logger.info(f"🤖 正在使用模型: {model_config['name']} (路径: {model_config['path']})")

        # 图像预处理
        # if isinstance(image_data, str):
        #     if image_data.startswith('data:image'):
        #         image_data = image_data.split(',')[1]
        #     image_bytes = base64.b64decode(image_data)
        #     image = Image.open(io.BytesIO(image_bytes))
        # else:
        image = Image.open(image_data)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)

        # 根据模型类型选择检测方法
        # if model_type == 'fire_smoke':
        #     # 使用Fire-Detection的YOLOv5算法
        #     logger.info("🔥 使用Fire-Detection YOLOv5算法进行火灾烟雾检测")
        #     detections = detect_fire_smoke_with_yolov5(image, conf_thres=confidence_threshold)
        # else:
        # 使用原始检测逻辑
        logger.info(f"🔍 开始模型推理，图像尺寸: {img_array.shape}, 置信度阈值: {confidence_threshold}")
        current_model = models[model_type]
        results = current_model(img_array)
        logger.info(f"✅ 模型推理完成")
        
        # 处理检测结果
        detections = []
        class_names = model_config['classes']
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
                    logger.info(f"🔍 检测到 {class_name}，置信度: {confidence:.2f}, 框坐标: {[float(x1), float(y1), float(x2), float(y2)]}")

        # 绘制检测结果
        annotated_image = img_array.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # 根据类别选择颜色
            if model_type == 'fire_smoke':
                color = (0, 0, 255) if 'fire' in class_name.lower() else (0, 255, 0)  # BGR格式
            else:
                color = (255, 255, 0)
                
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_image, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 转换回PIL图像并编码为base64
        annotated_pil = Image.fromarray(annotated_image)
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format='JPEG')
        annotated_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            'success': True,
            'detections': detections,
            'detection_count': len(detections),
            'annotated_image': f"data:image/jpeg;base64,{annotated_base64}",
            'model_info': {
                'name': model_config['name'],
                'type': model_type,
                'method': detections[0].get('method', 'standard') if detections else 'standard'
            }
        }

    except Exception as e:
        logger.error(f"图片处理失败: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'detections': [],
            'detection_count': 0
        }

def process_pressure_gauge_image(image_data, confidence_threshold=0.5):
    """处理仪表盘图片并进行读数识别"""
    try:
        # 获取仪表盘模型
        if 'pressure_gauge' not in models:
            raise ValueError("仪表盘识别模型未加载")

        current_model = models['pressure_gauge']
        model_config = MODEL_CONFIGS['pressure_gauge']

        # 将base64图片数据转换为numpy数组
        if isinstance(image_data, str):
            # 如果是base64编码的图片
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # 如果是文件对象
            image = Image.open(image_data)

        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 转换为numpy数组
        img_array = np.array(image)

        # 使用模型进行预测
        results = current_model(img_array, conf=confidence_threshold)

        # 处理检测结果
        detections = []
        annotated_image = img_array.copy()

        # 获取类别名称列表
        class_names = model_config['classes']

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # 获取类别名称
                    class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'

                    # 添加检测结果
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })

                    # 根据类别选择颜色
                    if class_name == 'tip':
                        color = (255, 0, 0)  # 红色用于指针
                    elif class_name == 'number':
                        color = (0, 255, 0)  # 绿色用于数字
                    else:
                        color = (0, 0, 255)  # 蓝色用于其他部件

                    # 在图片上绘制边界框
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # 添加标签
                    label = f'{class_name}: {confidence:.2f}'
                    cv2.putText(annotated_image, label, (int(x1), int(y1-10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 计算仪表盘读数
        gauge_readings = process_pressure_gauge_detection(img_array, results, annotated_image)

        # 将结果图片转换为base64
        annotated_pil = Image.fromarray(annotated_image)
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format='JPEG')
        annotated_base64 = base64.b64encode(buffer.getvalue()).decode()

        result = {
            'success': True,
            'detections': detections,
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
            'detection_count': len(detections),
            'gauge_readings': gauge_readings,
            'model_type': 'pressure_gauge',
            'model_name': model_config['name']
        }

        # 确保所有数据都是JSON可序列化的
        return convert_to_python_types(result)

    except Exception as e:
        logger.error(f"仪表盘处理错误: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def process_video(video_path, confidence_threshold=0.5, frame_skip=5, model_type='fire_smoke'):
    """处理视频文件并进行检测"""
    global video_processing, video_results, processed_video_path, current_model_type
    global realtime_video_results, video_frame_count, video_total_frames

    try:
        # 获取对应的模型
        if model_type not in models or not model_loaded.get(model_type, False):
            return {'success': False, 'error': f'{MODEL_CONFIGS[model_type]["name"]}模型未加载'}

        current_model = models[model_type]
        model_config = MODEL_CONFIGS[model_type]

        video_processing = True
        video_results = []
        realtime_video_results = []  # 清空实时结果
        video_frame_count = 0
        video_total_frames = 0
        current_model_type = model_type  # 记录当前模型类型

        logger.info(f"开始处理视频: {video_path}, 使用模型: {model_config['name']}")

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': '无法打开视频文件'}

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        video_total_frames = total_frames

        logger.info(f"视频信息: {total_frames}帧, {fps}FPS, 时长{duration:.1f}秒")

        frame_count = 0
        detection_count = 0
        processed_frames = 0

        # 创建输出视频的编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(RESULTS_FOLDER, f'detected_{os.path.basename(video_path)}')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            video_frame_count = frame_count  # 更新全局帧计数

            # 跳帧处理以提高速度
            if frame_count % frame_skip != 0:
                out.write(frame)
                continue

            processed_frames += 1

            # 使用模型进行预测
            if processed_frames == 1:  # 只在第一帧时打印，避免日志过多
                logger.info(f"🎬 视频处理使用模型: {model_config['name']}, 帧尺寸: {frame.shape}")
            results = current_model(frame, conf=confidence_threshold)

            # 处理检测结果
            frame_detections = []
            annotated_frame = frame.copy()

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        # 获取类别名称
                        class_names = model_config['classes']
                        class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'

                        # 添加检测结果
                        frame_detections.append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })

                        detection_count += 1

                        # 在帧上绘制边界框 - 根据模型类型选择颜色
                        if model_type == 'fire_smoke':
                            color = (0, 0, 255) if 'fire' in class_name.lower() else (0, 255, 0)  # BGR格式
                        elif model_type == 'safety_harness':
                            color = (0, 255, 0) if 'safe' in class_name.lower() else (0, 0, 255)
                        else:
                            color = (0, 255, 255)  # 黄色用于其他类型
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        # 添加标签
                        label = f'{class_name}: {confidence:.2f}'
                        cv2.putText(annotated_frame, label, (int(x1), int(y1-10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 保存检测结果
            if frame_detections:
                video_results.extend(frame_detections)

                # 添加到实时结果中（保持最新的100个检测结果）
                realtime_video_results.extend(frame_detections)
                if len(realtime_video_results) > 100:
                    realtime_video_results = realtime_video_results[-50:]  # 保留最新50个

                logger.info(f"第{frame_count}帧检测到{len(frame_detections)}个目标，实时推送结果")

            # 写入输出视频
            out.write(annotated_frame)

            # 每处理100帧输出一次进度
            if processed_frames % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")

        # 释放资源
        cap.release()
        out.release()

        video_processing = False
        processed_video_path = output_path  # 保存处理后的视频路径

        logger.info(f"视频处理完成: 处理了{processed_frames}帧, 检测到{detection_count}个目标")

        return {
            'success': True,
            'output_video': output_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'detection_count': detection_count,
            'detections': video_results,
            'duration': duration,
            'fps': fps
        }

    except Exception as e:
        video_processing = False
        processed_video_path = None  # 清除处理后的视频路径
        logger.error(f"视频处理错误: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def index():
    """主页"""
    # 从环境变量读取默认的报警推送地址，若未配置则使用内置mock地址
    default_push_url = os.getenv('ALERT_RECEIVER_URL', 'http://localhost:5000/api/mock/alert_receiver')
    return render_template('index.html', default_push_url=default_push_url)

@app.route('/api/status')
def api_status():
    """API状态检查"""
    # 检查是否至少有一个模型加载成功
    any_model_loaded = any(model_loaded.values())

    return jsonify({
        'status': 'running',
        'model_loaded': any_model_loaded,  # 前端需要的字段
        'models': {
            model_key: {
                'name': config['name'],
                'loaded': model_loaded.get(model_key, False),
                'path': config['path']
            }
            for model_key, config in MODEL_CONFIGS.items()
        },
        'ocr_available': ocr_reader is not None,
        'qwen_api_available': qwen_api_available,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """图片检测API - 支持多模型选择"""
    try:
        # 获取模型类型参数
        model_type = request.form.get('model_type', 'fire_smoke')
        model_name = MODEL_CONFIGS[model_type]['name']

        logger.info(f"🎯 调用模型: {model_name} (类型: {model_type})")

        # 检查模型是否加载
        if model_type not in model_loaded or not model_loaded[model_type]:
            logger.error(f"❌ 模型未加载: {model_name}")
            return jsonify({
                'success': False,
                'error': f'{model_name}模型未加载'
            }), 400

        # 获取置信度阈值
        confidence = float(request.form.get('confidence', 0.5))
        logger.info(f"📊 检测参数: 置信度={confidence}")

        # 检查是否有文件上传
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': '未选择文件'}), 400

            # 根据模型类型进行检测
            if model_type == 'pressure_gauge':
                logger.info(f"🔧 使用仪表盘识别模型处理图片: {file.filename}")
                result = process_pressure_gauge_image(file, confidence)
            else:
                logger.info(f"🔍 使用{model_name}处理图片: {file.filename}")
                result = process_image(file, confidence, model_type)

        elif 'image_data' in request.form:
            # 处理base64图片数据
            image_data = request.form['image_data']
            if model_type == 'pressure_gauge':
                result = process_pressure_gauge_image(image_data, confidence)
            else:
                result = process_image(image_data, confidence, model_type)

        else:
            return jsonify({'success': False, 'error': '未找到图片数据'}), 400

        return jsonify(result)

    except Exception as e:
        logger.error(f"检测API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detect_video', methods=['POST'])
def api_detect_video():
    """视频检测API"""
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': '模型未加载'
        }), 500

    if video_processing:
        return jsonify({
            'success': False,
            'error': '正在处理其他视频，请稍后再试'
        }), 400

    try:
        # 获取参数
        model_type = request.form.get('model_type', 'fire_smoke')
        model_name = MODEL_CONFIGS[model_type]['name']
        confidence = float(request.form.get('confidence', 0.5))
        frame_skip = int(request.form.get('frame_skip', 5))

        logger.info(f"🎬 视频检测调用模型: {model_name} (类型: {model_type})")
        logger.info(f"📊 视频检测参数: 置信度={confidence}, 跳帧={frame_skip}")

        # 检查模型是否加载
        if model_type not in model_loaded or not model_loaded[model_type]:
            logger.error(f"❌ 视频检测模型未加载: {model_name}")
            return jsonify({
                'success': False,
                'error': f'{model_name}模型未加载'
            }), 400

        # 检查是否有文件上传
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '未找到视频文件'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'}), 400

        if not allowed_file(file.filename, 'video'):
            return jsonify({'success': False, 'error': '不支持的视频格式'}), 400

        # 保存上传的视频文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        video_path = os.path.join(VIDEO_FOLDER, filename)
        file.save(video_path)

        # 在后台线程中处理视频
        def process_video_thread():
            global current_video_path, processed_video_path, video_results
            current_video_path = video_path
            processed_video_path = None  # 清除之前的处理结果
            video_results = []  # 清除之前的检测结果
            process_video(video_path, confidence, frame_skip, model_type)

        thread = threading.Thread(target=process_video_thread)
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': '视频上传成功，开始处理',
            'video_id': filename
        })

    except Exception as e:
        logger.error(f"视频检测API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/video_status')
def api_video_status():
    """获取视频处理状态"""
    return jsonify({
        'processing': video_processing,
        'current_video': current_video_path,
        'results_count': len(video_results) if video_results else 0,
        'processed_video': processed_video_path,
        'has_result': processed_video_path is not None and os.path.exists(processed_video_path) if processed_video_path else False
    })

@app.route('/api/video_results')
def api_video_results():
    """获取视频检测结果"""
    global current_model_type
    return jsonify({
        'success': True,
        'results': video_results,
        'total_detections': len(video_results) if video_results else 0,
        'model_type': current_model_type,  # 添加模型类型信息
        'output_video': processed_video_path  # 添加处理后的视频路径
    })

@app.route('/api/realtime_video_results')
def api_realtime_video_results():
    """获取实时视频检测结果"""
    global current_model_type, realtime_video_results, video_frame_count, video_total_frames, video_processing
    return jsonify({
        'success': True,
        'results': realtime_video_results,
        'total_detections': len(realtime_video_results),
        'model_type': current_model_type,
        'current_frame': video_frame_count,
        'total_frames': video_total_frames,
        'is_processing': video_processing,
        'progress': (video_frame_count / video_total_frames * 100) if video_total_frames > 0 else 0
    })

@app.route('/api/detect_stream', methods=['POST'])
def api_detect_stream():
    """开始视频流检测"""
    global stream_processing, stream_results, stream_thread, current_model_type

    try:
        data = request.get_json()
        stream_url = data.get('stream_url')
        confidence = data.get('confidence', 0.5)
        detection_interval = data.get('detection_interval', 30)
        model_type = data.get('model_type', 'fire_smoke')

        if not stream_url:
            return jsonify({'success': False, 'error': '视频流地址不能为空'})

        # 根据stream_url查找cameraId
        camera_id = None
        if camera_info_cache and 'CameraObject' in camera_info_cache:
            for camera in camera_info_cache['CameraObject']:
                if camera.get('address') == stream_url:
                    camera_id = camera.get('cameraId')
                    break
        
        if not camera_id:
            logger.warning(f"未在相机列表中找到与 {stream_url} 匹配的相机")
            # 可以选择生成一个临时的ID，或者返回错误
            camera_id = f"unknown_{uuid.uuid4().hex[:8]}"

        # 停止之前的流处理
        if stream_processing:
            stream_processing = False
            if stream_thread and stream_thread.is_alive():
                stream_thread.join(timeout=2)

        # 重置结果
        stream_results = []
        current_model_type = model_type

        # 启动新的流处理线程
        stream_thread = threading.Thread(
            target=process_video_stream,
            args=(stream_url, confidence, detection_interval, model_type, camera_id)
        )
        stream_thread.daemon = True
        stream_thread.start()

        return jsonify({'success': True, 'message': '视频流检测已启动'})

    except Exception as e:
        logger.error(f"启动视频流检测失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stream_results')
def api_stream_results():
    """获取视频流检测结果"""
    global current_model_type
    return jsonify({
        'success': True,
        'detections': stream_results,
        'total_detections': len(stream_results),
        'model_type': current_model_type,
        'is_processing': stream_processing
    })

@app.route('/api/stop_stream', methods=['POST'])
def api_stop_stream():
    """停止视频流检测"""
    global stream_processing, stream_thread

    try:
        stream_processing = False
        if stream_thread and stream_thread.is_alive():
            stream_thread.join(timeout=2)

        return jsonify({'success': True, 'message': '视频流检测已停止'})

    except Exception as e:
        logger.error(f"停止视频流检测失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download_video/<filename>')
def api_download_video(filename):
    """下载处理后的视频"""
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_results', methods=['POST'])
def api_clear_results():
    """清除处理结果和文件"""
    global processed_video_path, video_results, current_video_path

    try:
        # 删除处理后的视频文件
        if processed_video_path and os.path.exists(processed_video_path):
            os.remove(processed_video_path)
            logger.info(f"已删除处理后的视频文件: {processed_video_path}")

        # 清除全局变量
        processed_video_path = None
        video_results = []
        current_video_path = None

        return jsonify({
            'success': True,
            'message': '结果已清除'
        })
    except Exception as e:
        logger.error(f"清除结果失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/classify_image', methods=['POST'])
def api_classify_image():
    """图片灾情分类API"""
    if not qwen_api_available:
        return jsonify({
            'success': False,
            'error': '阿里云API未初始化或不可用'
        }), 500

    try:
        # 检查是否有文件上传
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': '未选择文件'}), 400

            # 处理上传的文件
            result = classify_disaster_image(file)

        elif 'image_data' in request.form:
            # 处理base64图片数据
            image_data = request.form['image_data']
            result = classify_disaster_image(image_data)

        else:
            return jsonify({'success': False, 'error': '未找到图片数据'}), 400

        return jsonify(result)

    except Exception as e:
        logger.error(f"图片灾情分类API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/classify_video', methods=['POST'])
def api_classify_video():
    """视频灾情分类API"""
    if not qwen_api_available:
        return jsonify({
            'success': False,
            'error': '阿里云API未初始化或不可用'
        }), 500

    try:
        # 检查是否有文件上传
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '未找到视频文件'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'}), 400

        if not allowed_file(file.filename, 'video'):
            return jsonify({'success': False, 'error': '不支持的视频格式'}), 400

        # 保存上传的视频文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"classify_{timestamp}_{filename}"
        video_path = os.path.join(VIDEO_FOLDER, filename)
        file.save(video_path)

        # 获取采样帧数参数
        sample_frames = int(request.form.get('sample_frames', 5))

        # 进行视频灾情分类
        result = classify_disaster_video(video_path, sample_frames)

        # 分类完成后删除临时视频文件
        try:
            os.remove(video_path)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        logger.error(f"视频灾情分类API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/reload', methods=['POST'])
def api_reload_model():
    """重新加载模型"""
    model_type = request.form.get('model_type', 'yolo')

    if model_type == 'all':
        success = load_models()
        return jsonify({
            'success': success,
            'models': {
                model_key: model_loaded.get(model_key, False)
                for model_key in MODEL_CONFIGS.keys()
            },
            'message': '所有模型重新加载完成' if success else '模型加载失败'
        })
    elif model_type in MODEL_CONFIGS:
        # 重新加载单个模型
        try:
            config = MODEL_CONFIGS[model_type]
            model_path = config['path']

            if os.path.exists(model_path):
                models[model_type] = YOLO(model_path)
                model_loaded[model_type] = True
                success = True
                message = f'{config["name"]}模型重新加载成功'
            else:
                model_loaded[model_type] = False
                success = False
                message = f'{config["name"]}模型文件不存在'

            return jsonify({
                'success': success,
                'model_loaded': model_loaded.get(model_type, False),
                'message': message
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    elif model_type == 'qwen':
        success = init_dashscope_api()
        return jsonify({
            'success': success,
            'qwen_api_available': qwen_api_available,
            'message': '阿里云API初始化成功' if success else '阿里云API初始化失败'
        })
    else:
        return jsonify({
            'success': False,
            'error': '不支持的模型类型'
        }), 400

def process_video_stream(stream_url, confidence_threshold=0.5, detection_interval=30, model_type='fire_smoke', camera_id=None):
    """处理视频流并进行实时检测"""
    global stream_processing, stream_results

    try:
        # 获取对应的模型
        if model_type not in models or not model_loaded.get(model_type, False):
            logger.error(f'{MODEL_CONFIGS[model_type]["name"]}模型未加载')
            return

        current_model = models[model_type]
        model_config = MODEL_CONFIGS[model_type]

        stream_processing = True
        logger.info(f"开始处理视频流: {stream_url}")

        # 打开视频流
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            logger.error(f"无法打开视频流: {stream_url}")
            stream_processing = False
            return

        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25  # 默认25fps

        while stream_processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("视频流读取失败，尝试重连...")
                time.sleep(1)
                continue

            frame_count += 1

            # 按检测间隔进行检测
            if frame_count % detection_interval == 0:
                try:
                    # 进行检测
                    results = current_model(frame, conf=confidence_threshold)

                    # 处理检测结果
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None and len(boxes) > 0:
                            for box in boxes:
                                # 获取检测信息
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = float(box.conf[0].cpu().numpy())
                                class_id = int(box.cls[0].cpu().numpy())
                                class_name = model_config['classes'][class_id]

                                # 计算时间戳
                                timestamp = frame_count / fps

                                # 添加到结果列表
                                detection = {
                                    'frame': frame_count,
                                    'time': timestamp,
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'type': model_type
                                }

                                # 发送报警通知
                                send_alert_notification(detection, camera_id)

                                stream_results.append(detection)

                                # 保持结果列表不超过1000个
                                if len(stream_results) > 1000:
                                    stream_results = stream_results[-500:]  # 保留最新的500个

                                logger.info(f"检测到 {class_name} (置信度: {confidence:.2f}) 在第{frame_count}帧")

                except Exception as e:
                    logger.error(f"检测处理失败: {str(e)}")
                    continue

        cap.release()
        logger.info("视频流处理结束")

    except Exception as e:
        logger.error(f"视频流处理异常: {str(e)}")
    finally:
        stream_processing = False

def send_alert_notification(alert_data, camera_id=None):
    """发送报警通知到指定的推送地址"""
    global rtsp_push_url, rtsp_current_frame
    
    if not rtsp_push_url:
        logger.warning("未设置推送地址，跳过报警推送")
        return False
    
    try:
        # 获取报警类型信息
        alert_type = alert_data.get("class")  # 从detection中获取class
        alarm_info = ALARM_TYPE_MAPPING.get(alert_type, {'algoId': '9999', 'algoName': alert_type})

        # 获取相机名称
        camera_name = "Unknown Camera"
        if camera_id and camera_info_cache and 'CameraObject' in camera_info_cache:
            for camera in camera_info_cache['CameraObject']:
                if camera.get('cameraId') == camera_id:
                    camera_name = camera.get('cameraName', camera_name)
                    break
        
        # 获取边界框信息并格式化为 x,y,w,h 格式
        bbox = alert_data.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            # bbox格式通常是 [x1, y1, x2, y2]，需要转换为 [x, y, w, h]
            x1, y1, x2, y2 = bbox[:4]
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            area = f"{x},{y},{w},{h}"
        else:
            area = "0,0,0,0"
        
        # 获取当前帧的base64编码
        image_base64 = ""
        if rtsp_current_frame is not None:
            try:
                with rtsp_current_frame_lock:
                    # 将OpenCV图像转换为PIL图像
                    frame_rgb = cv2.cvtColor(rtsp_current_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    pil_image.save(f"{camera_name}_{time.strftime('%Y%m%d%H%M%S')}.jpg", format='JPEG', quality=85)
                    
                    # 转换为base64
                    buffer = io.BytesIO()
                    # pil_image.save(buffer, format='JPEG', quality=85)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as e:
                logger.warning(f"获取图片base64编码失败: {str(e)}")
        
        # 生成报警ID（使用时间戳和随机数）
        import random
        alarm_id = int(time.time() * 1000) + random.randint(1000, 9999)
        
        # 格式化报警时间
        alarm_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 构造符合API文档格式的报警数据
        """{
 "owner":"毕升",
 "ownerId": "",
 "AlarmObject":[
  {
   "id":11358,
   "alarmTime":"2021-10-08 11:03:30",
   "area":"247,189,209,714",
   "algoId":"2000",
   "algoName":"烟雾",
   "status":0,
   "cameraId":"Dest0211MA3DC9QR0401",
   "cameraName":"测试机",
   "deviceId":"1233",
   "deviceName":"test111",
   "image":"http://10.168.0.1/images/alarm/Dest0211MA3DC9QR0401/20211008T110330_13088236_247,189,209,714.jpg",
   " imageBase64":"xxxxxxxxx",
   "describe":""
  }
 ]
}"""
        notification_data = {
            "owner": COMPANY_NAME,
            "ownerId": COMPANY_ID,
            "AlarmObject": [
                {
                    "id": alarm_id,
                    "alarmTime": alarm_time,
                    "area": area,
                    "algoId": alarm_info['algoId'],
                    "algoName": alarm_info['algoName'],
                    "status": 0,  # 0报警/1销警，固定填0
                    "cameraId": camera_id,
                    "cameraName": camera_name,
                    "deviceId": DEVICE_ID,
                    "deviceName": DEVICE_NAME,
                    "describe": f"检测到{alarm_info['algoName']}，置信度: {alert_data['confidence']:.2f}, 模型类型: {alert_data['type']}",
                    "image": "",  # 图片URL，如果有的话
                    "imageBase64": image_base64
                }
            ]
        }
        
        # 发送POST请求
        response = requests.post(
            rtsp_push_url,
            json=notification_data,
            timeout=10,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            logger.info(f"报警推送成功: {alarm_info['algoName']} (ID: {alarm_id}, 编码: {alarm_info['algoId']})")
            return True
        else:
            logger.error(f"报警推送失败，状态码: {response.status_code}, 响应: {response.text}")
            return False
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"报警推送异常: {str(e)}")
        return False

# 添加全局变量用于存储当前帧和检测结果
rtsp_current_frame = None
rtsp_current_frame_lock = threading.Lock()
rtsp_detection_frame_results = {}  # 存储每帧的检测结果

# 公司和设备信息配置
COMPANY_NAME = os.getenv('COMPANY_NAME', "")  # 分析设备所属公司名称
COMPANY_ID = os.getenv('COMPANY_ID', "")  # 分析设备所属公司编码
# CAMERA_ID = os.getenv('CAMERA_ID', "")  # 摄像机ID
# CAMERA_NAME = os.getenv('CAMERA_NAME', "")  # 摄像机名称
DEVICE_ID = os.getenv('DEVICE_ID', "")  # 分析设备ID
DEVICE_NAME = os.getenv('DEVICE_NAME', "")  # 分析设备名称

# 报警类型映射（algoId和algoName）
ALARM_TYPE_MAPPING = {
    'smoke': {'algoId': '2000', 'algoName': '烟雾'},
    'fire': {'algoId': '2001', 'algoName': '明火'},
    'Fall-Detected': {'algoId': '2002', 'algoName': '跌倒'},
    'no harness': {'algoId': '2003', 'algoName': '未系安全带'},
    'head': {'algoId': '2004', 'algoName': '未戴安全帽'}
}

def draw_detection_boxes(frame, detections):
    """在图像上绘制检测框"""
    if not detections:
        return frame
    
    # 复制帧以避免修改原始图像
    annotated_frame = frame.copy()
    
    try:
        for detection in detections:
            # 获取检测框坐标和信息
            bbox = detection.get('bbox', detection.get('box', []))
            if len(bbox) < 4:
                continue
                
            x1, y1, x2, y2 = map(int, bbox[:4])
            class_name = detection.get('class', detection.get('class_name', 'Unknown'))
            confidence = detection.get('confidence', 0.0)
            
            # 确保坐标在图像范围内
            h, w = annotated_frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # 根据类别选择颜色
            if 'fire' in class_name.lower():
                color = (0, 0, 255)  # 红色 (BGR)
            elif 'smoke' in class_name.lower():
                color = (0, 255, 255)  # 黄色 (BGR)
            elif 'fall' in class_name.lower() or 'detected' in class_name.lower():
                color = (255, 0, 0)  # 蓝色 (BGR)
            elif 'harness' in class_name.lower():
                color = (0, 255, 0) if 'safe' in class_name.lower() else (0, 0, 255)  # 绿色/红色
            else:
                color = (255, 255, 0)  # 青色 (BGR)
            
            # 绘制检测框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # 确保标签在图像范围内
            label_y = max(y1 - 10, label_size[1] + 5)
            label_x = min(x1, w - label_size[0] - 5)
            
            # 绘制标签背景
            cv2.rectangle(annotated_frame, 
                         (label_x - 2, label_y - label_size[1] - 2),
                         (label_x + label_size[0] + 2, label_y + 2),
                         color, -1)
            
            # 绘制标签文字
            cv2.putText(annotated_frame, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
    except Exception as e:
        logger.error(f"绘制检测框时出错: {str(e)}")
        return frame  # 出错时返回原始帧
    
    return annotated_frame

def send_alert_notification_for_stream(push_url, frame, alert_data, camera_id, camera_name):
    """针对指定流的报警推送，使用该流的推送地址与当前帧。
    注意：此函数为同步发送实现，可能阻塞；请优先使用 schedule_alert_push 进行异步调度。
    """
    global rtsp_current_frame_lock, external_alert_lock, external_alert_notifications
    try:
        if not push_url:
            logger.warning("未设置推送地址，跳过报警推送")
            return False

        alert_type = alert_data["alert_type"]
        alarm_info = ALARM_TYPE_MAPPING.get(alert_type, {'algoId': '9999', 'algoName': alert_type})

        bbox = alert_data.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            area = f"{x},{y},{w},{h}"
        else:
            area = "0,0,0,0"

        image_base64 = ""
        if frame is not None:
            try:
                # 优先裁剪报警框区域，降低编码开销
                crop_img = frame
                # try:
                #     if len(bbox) >= 4:
                #         x1, y1, x2, y2 = bbox[:4]
                #         x1 = max(0, int(x1))
                #         y1 = max(0, int(y1))
                #         x2 = max(x1 + 1, int(x2))
                #         y2 = max(y1 + 1, int(y2))
                #         h, w = frame.shape[:2]
                #         x1 = min(x1, w - 1)
                #         x2 = min(x2, w)
                #         y1 = min(y1, h - 1)
                #         y2 = min(y2, h)
                #         crop_img = frame[y1:y2, x1:x2]
                #         # 如果区域过小或异常，使用整帧
                #         if crop_img.size == 0 or crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
                #             crop_img = frame
                # except Exception:
                #     crop_img = frame

                # 限制最大宽度，降低编码数据量
                # try:
                #     max_w = 640
                #     h, w = crop_img.shape[:2]
                #     if w > max_w:
                #         scale = max_w / float(w)
                #         new_size = (max_w, max(1, int(h * scale)))
                #         crop_img = cv2.resize(crop_img, new_size, interpolation=cv2.INTER_AREA)
                # except Exception:
                #     pass

                # 使用OpenCV编码为JPEG（更高效，减少GIL争用）
                # cv2.imwrite(f"{camera_name}_{time.strftime('%Y%m%d%H%M%S')}.jpg", crop_img)
                ret, buf = cv2.imencode('.jpg', crop_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    image_base64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            except Exception as e:
                logger.warning(f"获取图片base64编码失败: {str(e)}")

        import random
        alarm_id = int(time.time() * 1000) + random.randint(1000, 9999)
        alarm_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        notification_data = {
            "owner": COMPANY_NAME,
            "ownerId": COMPANY_ID,
            "AlarmObject": [
                {
                    "id": alarm_id,
                    "alarmTime": alarm_time,
                    "area": area,
                    "algoId": alarm_info['algoId'],
                    "algoName": alarm_info['algoName'],
                    "status": 0,
                    "cameraId": camera_id,
                    "cameraName": camera_name,
                    "deviceId": DEVICE_ID,
                    "deviceName": DEVICE_NAME,
                    "describe": f"检测到{alarm_info['algoName']}，置信度: {alert_data['confidence']:.2f}, 模型类型: {alert_data['type']}",
                    "image": "",
                    "imageBase64": image_base64
                }
            ]
        }
        # 保存全局变量
        data = notification_data["AlarmObject"][0]
        # 构造标准化的报警数据
        alert_notification = {
            'id': data.get('id'),
            'timestamp': data.get('alarmTime'),
            'received_time': datetime.now().isoformat(),
            'alert_type': data.get('algoName', '未知'),
            'alert_code': data.get('algoId'),
            'description': data.get('Describe', ''),
            'camera_id': data.get('cameraId'),
            'camera_name': data.get('cameraName'),
            'device_id': data.get('deviceId'),
            'device_name': data.get('deviceName'),
            'area': data.get('area', ''),
            'status': data.get('status', 0),  # 0报警/1销警
            'image_base64': data.get('imageBase64', ''),
            'raw_data': data
        }
        
        # 存储到全局变量中
        with external_alert_lock:
            external_alert_notifications.append(alert_notification)
            # 保持列表不超过100条记录
            if len(external_alert_notifications) > 100:
                external_alert_notifications = external_alert_notifications[-50:]

        response = requests.post(
            push_url,
            json=notification_data,
            timeout=5,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            logger.info(f"报警推送成功: {alarm_info['algoName']} (ID: {alarm_id}, 编码: {alarm_info['algoId']})")
            return True
        else:
            logger.error(f"报警推送失败，状态码: {response.status_code}, 响应: {response.text}")
            return False
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"报警推送异常: {str(e)}")
        return False

# 异步调度报警推送，避免阻塞检测主循环
def schedule_alert_push(push_url, frame, alert_data, camera_id, camera_name):
    try:
        # 控制并发，若推送过多则跳过，避免阻塞主环
        acquired = alert_push_semaphore.acquire(blocking=False)
        if not acquired:
            logger.warning("报警推送繁忙，跳过本次推送以保护帧流畅度")
            return False

        frame_ref = frame  # 引用传递，后台线程进行编码
        alert_copy = dict(alert_data) if isinstance(alert_data, dict) else alert_data

        def _task():
            try:
                return send_alert_notification_for_stream(push_url, frame_ref, alert_copy, camera_id, camera_name)
            finally:
                try:
                    alert_push_semaphore.release()
                except Exception:
                    pass

        alert_push_executor.submit(_task)
        return True
    except Exception as e:
        logger.error(f"调度报警推送失败: {str(e)}")
        try:
            alert_push_semaphore.release()
        except Exception:
            pass
        return False

def process_rtsp_stream_worker(stream_id):
    """多流RTSP处理线程：按所选模型并发检测并推送报警，限制推理并发"""
    global streams
    try:
        with streams_lock:
            ctx = streams.get(stream_id)
        if not ctx:
            logger.error(f"流 {stream_id} 不存在")
            return

        url = ctx['url']
        models_selected = ctx['models']
        push_url = ctx['push_url']
        push_frequency = max(0.033333, float(ctx.get('push_frequency', 5)))

        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            logger.error(f"无法打开RTSP流: {url}")
            with streams_lock:
                ctx['active'] = False
            return

        frame_count = 0
        last_detection_time = 0.0

        while True:
            with streams_lock:
                running = streams.get(stream_id, {}).get('active', False)
            if not running:
                break

            ret, frame = cap.read()
            if not ret:
                logger.warning(f"流 {stream_id} 读取失败，尝试重连...")
                time.sleep(1)
                continue

            frame_count += 1
            # 更新当前帧
            try:
                ctx['frame_lock'].acquire()
                ctx['current_frame'] = frame.copy()
                ctx['frame_count'] = frame_count
            finally:
                ctx['frame_lock'].release()

            now = time.time()
            if now - last_detection_time >= push_frequency:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    # 并发执行所选模型检测
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    detections_all = []
                    with ThreadPoolExecutor(max_workers=max(1, len(models_selected))) as executor:
                        futures = {
                            executor.submit(detect_with_model, m, pil_image, frame_count): m
                            for m in models_selected if model_loaded.get(m, False)
                        }
                        for future in as_completed(futures):
                            try:
                                detections_all.extend(future.result())
                            except Exception as e:
                                logger.error(f"模型检测失败: {str(e)}")

                    # 处理检测结果并推送报警
                    for det in detections_all:
                        cls = det['class_name']
                        conf = det['confidence']
                        mtype = det['model_type']
                        if cls in ALERT_CODES:
                            alert_key = f"{mtype}_{cls}"
                            last_time = ctx['last_alert_time'].get(alert_key, 0.0)
                            if now - last_time >= push_frequency:
                                alert_data = {
                                    "timestamp": datetime.now().isoformat(),
                                    "alert_code": ALERT_CODES[cls],
                                    "alert_type": cls,
                                    "confidence": conf,
                                    "frame_number": frame_count,
                                    "model_type": mtype,
                                    "detection_mode": "multi_stream",
                                    "bbox": det['bbox'],
                                    "type": "yolo"
                                }
                                queued = schedule_alert_push(push_url, frame, alert_data, ctx['camera_id'], ctx['camera_name'])
                                # 标记为已入队，真实发送结果在后台线程日志中记录
                                alert_data['alert_sent'] = queued
                                # 写入结果
                                with streams_lock:
                                    ctx = streams.get(stream_id)
                                    if ctx:
                                        ctx['detection_results'].append(alert_data)
                                        if len(ctx['detection_results']) > 1000:
                                            ctx['detection_results'] = ctx['detection_results'][-500:]
                                        ctx['last_alert_time'][alert_key] = now

                    # 保存当前帧的检测框用于可视化
                    with streams_lock:
                        ctx = streams.get(stream_id)
                        if ctx:
                            ctx['frame_results'][frame_count] = detections_all
                            # 清理旧帧
                            if len(ctx['frame_results']) > 100:
                                old_keys = sorted(ctx['frame_results'].keys())[:-100]
                                for k in old_keys:
                                    ctx['frame_results'].pop(k, None)

                    last_detection_time = now
                except Exception as e:
                    logger.error(f"流 {stream_id} 帧处理失败: {str(e)}")
                    # 不中断，继续循环
            time.sleep(0.01)

        cap.release()
        with streams_lock:
            if stream_id in streams:
                streams[stream_id]['active'] = False
        logger.info(f"流 {stream_id} 处理结束")
    except Exception as e:
        logger.error(f"流 {stream_id} 线程异常: {str(e)}")
        with streams_lock:
            if stream_id in streams:
                streams[stream_id]['active'] = False

def process_rtsp_stream_worker_with_qwen3vl(stream_id):
    """多流RTSP处理线程：支持独立大模型检测线程的版本"""
    global streams
    try:
        with streams_lock:
            ctx = streams.get(stream_id)
        if not ctx:
            logger.error(f"流 {stream_id} 不存在")
            return

        url = ctx['url']
        models_selected = ctx['models']
        push_url = ctx['push_url']
        push_frequency = max(0.033333, float(ctx.get('push_frequency', 5)))
        enable_qwen3vl = ctx.get('enable_qwen3vl', False)
        qwen3vl_detection_types = ctx.get('qwen3vl_detection_types', [])

        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            logger.error(f"无法打开RTSP流: {url}")
            with streams_lock:
                ctx['active'] = False
            return

        # 如果启用了大模型检测，启动独立的大模型检测线程
        if enable_qwen3vl and qwen3vl_model_loaded:
            with streams_lock:
                ctx['qwen3vl_active'] = True
            qwen3vl_thread = threading.Thread(
                target=process_qwen3vl_independent_detection_for_stream, 
                args=(stream_id,), 
                daemon=True
            )
            qwen3vl_thread.start()
            with streams_lock:
                ctx['qwen3vl_thread'] = qwen3vl_thread
            logger.info(f"流 {stream_id} 启动独立大模型检测线程")

        frame_count = 0
        last_detection_time = 0.0

        while True:
            with streams_lock:
                running = streams.get(stream_id, {}).get('active', False)
            if not running:
                break

            ret, frame = cap.read()
            if not ret:
                logger.warning(f"流 {stream_id} 读取失败，尝试重连...")
                time.sleep(1)
                continue

            frame_count += 1
            # 更新当前帧
            try:
                ctx['frame_lock'].acquire()
                ctx['current_frame'] = frame.copy()
                ctx['frame_count'] = frame_count
            finally:
                ctx['frame_lock'].release()

            now = time.time()
            if now - last_detection_time >= push_frequency:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    # 并发执行所选模型检测
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    detections_all = []
                    with ThreadPoolExecutor(max_workers=max(1, len(models_selected))) as executor:
                        futures = {
                            executor.submit(detect_with_model, m, pil_image, frame_count): m
                            for m in models_selected if model_loaded.get(m, False)
                        }
                        for future in as_completed(futures):
                            try:
                                detections_all.extend(future.result())
                            except Exception as e:
                                logger.error(f"模型检测失败: {str(e)}")

                    # 处理检测结果并推送报警
                    for det in detections_all:
                        cls = det['class_name']
                        conf = det['confidence']
                        mtype = det['model_type']
                        if cls in ALERT_CODES:
                            alert_key = f"{mtype}_{cls}"
                            last_time = ctx['last_alert_time'].get(alert_key, 0.0)
                            if now - last_time >= push_frequency:
                                alert_data = {
                                    "timestamp": datetime.now().isoformat(),
                                    "alert_code": ALERT_CODES[cls],
                                    "alert_type": cls,
                                    "confidence": conf,
                                    "frame_number": frame_count,
                                    "model_type": mtype,
                                    "detection_mode": "multi_stream",
                                    "bbox": det['bbox'],
                                    "type": "yolo"
                                }
                                queued = schedule_alert_push(push_url, frame, alert_data, ctx['camera_id'], ctx['camera_name'])
                                # 标记为已入队，真实发送结果在后台线程日志中记录
                                alert_data['alert_sent'] = queued
                                # 写入结果
                                with streams_lock:
                                    ctx = streams.get(stream_id)
                                    if ctx:
                                        ctx['detection_results'].append(alert_data)
                                        if len(ctx['detection_results']) > 1000:
                                            ctx['detection_results'] = ctx['detection_results'][-500:]
                                        ctx['last_alert_time'][alert_key] = now

                    # 保存当前帧的检测框用于可视化
                    with streams_lock:
                        ctx = streams.get(stream_id)
                        if ctx:
                            ctx['frame_results'][frame_count] = detections_all
                            # 清理旧帧
                            if len(ctx['frame_results']) > 100:
                                old_keys = sorted(ctx['frame_results'].keys())[:-100]
                                for k in old_keys:
                                    ctx['frame_results'].pop(k, None)

                    last_detection_time = now
                except Exception as e:
                    logger.error(f"流 {stream_id} 帧处理失败: {str(e)}")
                    # 不中断，继续循环
            time.sleep(0.01)

        cap.release()
        
        # 停止大模型检测线程
        with streams_lock:
            ctx = streams.get(stream_id)
            if ctx:
                ctx['active'] = False
                ctx['qwen3vl_active'] = False
                qwen3vl_thread = ctx.get('qwen3vl_thread')
                if qwen3vl_thread and qwen3vl_thread.is_alive():
                    qwen3vl_thread.join(timeout=3)
                    logger.info(f"流 {stream_id} 独立大模型检测线程已停止")
        
        logger.info(f"流 {stream_id} 处理结束")
    except Exception as e:
        logger.error(f"流 {stream_id} 线程异常: {str(e)}")
        with streams_lock:
            ctx = streams.get(stream_id)
            if ctx:
                ctx['active'] = False
                ctx['qwen3vl_active'] = False


def process_qwen3vl_independent_detection_for_stream(stream_id):
    """为特定流的独立Qwen3-VL大模型检测线程函数"""
    global streams
    
    logger.info(f"启动流 {stream_id} 的独立Qwen3-VL大模型检测线程")
    
    try:
        while True:
            with streams_lock:
                ctx = streams.get(stream_id)
                if not ctx or not ctx.get('qwen3vl_active', False):
                    break
                
                qwen3vl_detection_types = ctx.get('qwen3vl_detection_types', [])
                push_url = ctx.get('push_url')
                push_frequency = max(0.033333, float(ctx.get('push_frequency', 5)))
            
            try:
                # 获取当前帧
                current_frame = None
                frame_count = 0
                with streams_lock:
                    ctx = streams.get(stream_id)
                    if ctx and ctx.get('current_frame') is not None:
                        try:
                            ctx['frame_lock'].acquire()
                            current_frame = ctx['current_frame'].copy()
                            frame_count = ctx['frame_count']
                        finally:
                            ctx['frame_lock'].release()
                
                # 如果有当前帧且大模型已加载，进行检测
                if current_frame is not None and qwen3vl_model_loaded:
                    # 转换为PIL图像
                    frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # 执行Qwen3-VL检测
                    qwen3vl_result = analyze_safety_with_qwen3vl(pil_image)
                    
                    if qwen3vl_result:
                        current_time = time.time()
                        
                        # 处理检测结果
                        for detection_type, result in qwen3vl_result.items():
                            if result['detected'] and result['confidence'] > 0.9:
                                # 映射检测类型到标准类名
                                class_mapping = {
                                    'smoke': 'smoke',
                                    'fire': 'fire', 
                                    'fall': 'Fall-Detected',
                                    'no_harness': 'no harness'
                                }
                                
                                if detection_type in class_mapping:
                                    class_name = class_mapping[detection_type]
                                    
                                    # 检查是否在启用的检测类型中
                                    detection_type_mapping = {
                                        'smoke': 'fire_smoke',
                                        'fire': 'fire_smoke',
                                        'Fall-Detected': 'fall_detection',
                                        'no harness': 'safety_harness',
                                        'head': 'safety_helmet'
                                    }
                                    
                                    required_type = detection_type_mapping.get(class_name)
                                    if required_type not in qwen3vl_detection_types:
                                        continue  # 跳过未启用的检测类型
                                    
                                    # 检查是否需要报警
                                    if class_name in ALERT_CODES:
                                        # 检查推送频率限制
                                        alert_key = f"qwen3vl_{class_name}"
                                        
                                        with streams_lock:
                                            ctx = streams.get(stream_id)
                                            if not ctx:
                                                break
                                            last_time = ctx['qwen3vl_last_alert_time'].get(alert_key, 0)
                                        
                                        if current_time - last_time >= push_frequency:
                                            alert_data = {
                                                "timestamp": datetime.now().isoformat(),
                                                "alert_code": ALERT_CODES[class_name],
                                                "alert_type": class_name,
                                                "confidence": result['confidence'],
                                                "frame_number": frame_count,
                                                "model_type": "qwen3vl_multimodal",
                                                "detection_mode": f"independent_qwen3vl_stream_{stream_id}",
                                                "bbox": [0, 0, current_frame.shape[1], current_frame.shape[0]],  # 全图检测
                                                "description": result['description'],
                                                "type": "多模态大模型"
                                            }
                                            
                                            # 发送报警通知
                                            queued = schedule_alert_push(push_url, current_frame, alert_data, ctx['camera_id'], ctx['camera_name'])
                                            alert_data["alert_sent"] = queued
                                            
                                            # 添加到流的检测结果
                                            with streams_lock:
                                                ctx = streams.get(stream_id)
                                                if ctx:
                                                    ctx['qwen3vl_results'].append(alert_data)
                                                    ctx['detection_results'].append(alert_data)  # 同时添加到主结果中
                                                    # 保持结果列表不超过500条
                                                    if len(ctx['qwen3vl_results']) > 500:
                                                        ctx['qwen3vl_results'] = ctx['qwen3vl_results'][-250:]
                                                    if len(ctx['detection_results']) > 1000:
                                                        ctx['detection_results'] = ctx['detection_results'][-500:]
                                                    # 更新最后报警时间
                                                    ctx['qwen3vl_last_alert_time'][alert_key] = current_time
                                            
                                            logger.info(f"流 {stream_id} 独立Qwen3-VL检测到 {class_name}: {result['description']} (置信度: {result['confidence']:.2f})")
                
                # 等待5秒进行下一次检测
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"流 {stream_id} 独立Qwen3-VL检测处理失败: {str(e)}")
                time.sleep(5)  # 出错时也要等待，避免频繁重试
                continue
        
        logger.info(f"流 {stream_id} 独立Qwen3-VL大模型检测线程结束")
        
    except Exception as e:
        logger.error(f"流 {stream_id} 独立Qwen3-VL检测线程异常: {str(e)}")
    finally:
        with streams_lock:
            ctx = streams.get(stream_id)
            if ctx:
                ctx['qwen3vl_active'] = False


# ================= 多流REST接口 =================
@app.route('/api/streams', methods=['GET'])
def api_streams_list():
    try:
        with streams_lock:
            items = []
            for sid, ctx in streams.items():
                items.append({
                    'id': sid,
                    'active': ctx.get('active', False),
                    'stream_url': ctx.get('url'),
                    'push_url': ctx.get('push_url'),
                    'camera_id': ctx.get('camera_id'),
                    'camera_name': ctx.get('camera_name'),
                    'models': ctx.get('models', []),
                    'push_frequency': ctx.get('push_frequency', 5),
                    'detection_count': len(ctx.get('detection_results', []))
                })
        return jsonify({'success': True, 'streams': items})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/streams/start', methods=['POST'])
def api_stream_start():
    try:
        data = request.get_json() or {}
        stream_url = data.get('stream_url')
        push_url = data.get('push_url')
        camera_id = data.get('camera_id')
        camera_name = data.get('camera_name')
        models_selected = data.get('models')
        push_frequency = int(data.get('push_frequency', 5))
        # 新增：支持大模型检测控制
        enable_qwen3vl = data.get('enable_qwen3vl', False)
        qwen3vl_detection_types = data.get('qwen3vl_detection_types', ['fire_smoke', 'safety_harness', 'fall_detection', 'safety_helmet'])

        if not stream_url:
            return jsonify({'success': False, 'error': '缺少stream_url'}), 400

        # 过滤非法模型
        models_selected = [m for m in models_selected if m in STREAM_MODELS_ALLOWED]
        if not models_selected:
            return jsonify({'success': False, 'error': '未选择有效模型'}), 400

        # 验证大模型检测类型
        valid_qwen3vl_types = ['fire_smoke', 'safety_harness', 'fall_detection', 'safety_helmet']
        qwen3vl_detection_types = [t for t in qwen3vl_detection_types if t in valid_qwen3vl_types]

        stream_id = str(uuid.uuid4())
        ctx = {
            'id': stream_id,
            'url': stream_url,
            'push_url': push_url,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'models': models_selected,
            'push_frequency': push_frequency,
            'active': True,
            'thread': None,
            'current_frame': None,
            'frame_lock': threading.Lock(),
            'frame_results': {},
            'frame_count': 0,
            'detection_results': [],
            'last_alert_time': {},
            # 新增：大模型检测相关配置
            'enable_qwen3vl': enable_qwen3vl,
            'qwen3vl_detection_types': qwen3vl_detection_types,
            'qwen3vl_thread': None,
            'qwen3vl_active': False,
            'qwen3vl_results': [],
            'qwen3vl_last_alert_time': {}
        }
        with streams_lock:
            streams[stream_id] = ctx

        # 启动主流处理线程
        t = threading.Thread(target=process_rtsp_stream_worker_with_qwen3vl, args=(stream_id,), daemon=True)
        t.start()
        with streams_lock:
            streams[stream_id]['thread'] = t

        return jsonify({'success': True, 'id': stream_id})
    except Exception as e:
        logger.error(f"启动流失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/streams/<stream_id>/stop', methods=['POST'])
def api_stream_stop(stream_id):
    try:
        with streams_lock:
            ctx = streams.get(stream_id)
            if not ctx:
                return jsonify({'success': False, 'error': '流不存在'}), 404
            ctx['active'] = False
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/streams/<stream_id>/status', methods=['GET'])
def api_stream_status(stream_id):
    try:
        with streams_lock:
            ctx = streams.get(stream_id)
            if not ctx:
                return jsonify({'success': False, 'error': '流不存在'}), 404
            status = {
                'id': stream_id,
                'active': ctx.get('active', False),
                'stream_url': ctx.get('url'),
                'push_url': ctx.get('push_url'),
                'models': ctx.get('models'),
                'push_frequency': ctx.get('push_frequency'),
                'detection_count': len(ctx.get('detection_results', [])),
                'frame_count': ctx.get('frame_count', 0)
            }
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/streams/<stream_id>/results', methods=['GET'])
def api_stream_results_multi(stream_id):
    try:
        with streams_lock:
            ctx = streams.get(stream_id)
            if not ctx:
                return jsonify({'success': False, 'error': '流不存在'}), 404
            results = ctx.get('detection_results', [])
        return jsonify({'success': True, 'results': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/streams/<stream_id>/stream')
def api_stream_image_multi(stream_id):
    def generate_frames_for_stream():
        while True:
            with streams_lock:
                ctx = streams.get(stream_id)
            if not ctx or not ctx.get('active', False):
                break
            try:
                # 快速复制当前帧和检测结果，减少锁持有时间
                frame = None
                current_frame_number = 0
                frame_detections = []
                
                # 使用try-finally确保锁被释放，并设置超时避免死锁
                lock_acquired = ctx['frame_lock'].acquire(timeout=0.1)
                if lock_acquired:
                    try:
                        if ctx['current_frame'] is not None:
                            frame = ctx['current_frame'].copy()
                        current_frame_number = ctx.get('frame_count', 0)
                        
                        # 快速获取最近几帧的检测结果
                        for fn in range(max(1, current_frame_number - 3), current_frame_number + 1):
                            dets = ctx['frame_results'].get(fn)
                            if dets:
                                frame_detections.extend(dets)
                    finally:
                        ctx['frame_lock'].release()
                else:
                    # 如果无法获取锁，使用上一帧或跳过
                    time.sleep(0.01)
                    continue

                if frame is not None:
                    # 在锁外进行绘制操作，避免阻塞帧更新
                    if frame_detections:
                        try:
                            frame = draw_detection_boxes(frame, frame_detections)
                        except Exception as e:
                            logger.error(f"绘制检测框失败: {str(e)}")
                    
                    # 使用较低的JPEG质量以提高编码速度
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        # 增加Content-Length，提升某些浏览器对MJPEG的兼容性
                        header = (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Cache-Control: no-cache, no-store, must-revalidate\r\n'
                            b'Pragma: no-cache\r\n'
                            b'Expires: 0\r\n'
                            + f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode('ascii')
                        )
                        yield header + frame_bytes + b'\r\n'
                else:
                    # 如果没有帧，发送空白帧避免前端等待
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    ret, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        header = (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            + f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode('ascii')
                        )
                        yield header + frame_bytes + b'\r\n'
                
                # 控制帧率，避免过度消耗CPU
                time.sleep(0.033)  # 约30fps
                
            except Exception as e:
                logger.error(f"流 {stream_id} 帧生成错误: {str(e)}")
                time.sleep(0.1)

    return Response(generate_frames_for_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_for_stream(), mimetype='multipart/x-mixed-replace; boundary=frame', headers=headers)
    
    # 定义颜色映射
    color_map = {
        'fire': (0, 0, 255),      # 红色
        'smoke': (128, 128, 128), # 灰色
        'Fall-Detected': (255, 0, 0),  # 蓝色
        'no harness': (0, 255, 255),   # 黄色
        'safe harness': (0, 255, 0)    # 绿色
    }
    
    for detection in detections:
        bbox = detection.get('bbox')
        class_name = detection.get('class_name', '')
        confidence = detection.get('confidence', 0)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            color = color_map.get(class_name, (255, 255, 255))
            
            # 绘制检测框
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), color, -1)
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return annotated_frame

# def detect_with_model(model_type, pil_image, frame_count):
#     """单个模型检测函数，用于多线程调用"""
#     if model_type not in models or not model_loaded.get(model_type, False):
#         return []
    
#     try:
#         results = models[model_type](pil_image, conf=0.5)
#         detections = []
        
#         for result in results:
#             boxes = result.boxes
#             if boxes is not None and len(boxes) > 0:
#                 for box in boxes:
#                     class_id = int(box.cls[0])
#                     confidence = float(box.conf[0])
#                     if confidence > 0.7:
#                         class_name = models[model_type].names[class_id]
                        
#                         # 获取边界框坐标
#                         xyxy = box.xyxy[0].cpu().numpy()
                        
#                         detection = {
#                             'model_type': model_type,
#                             'class_name': class_name,
#                             'confidence': confidence,
#                             'bbox': xyxy.tolist(),
#                             'frame_number': frame_count
#                         }
#                         detections.append(detection)
        
#         return detections
#     except Exception as e:
#         logger.error(f"模型 {model_type} 检测失败: {str(e)}")
#         return []

def detect_with_model(model_type, pil_image, frame_count):
    # global detect_th
    """单个模型检测函数，用于多线程调用"""
    if model_type not in models or not model_loaded.get(model_type, False):
        return []
    
    try:
        # if model_type == 'fire_smoke':
        #     # 使用Fire-Detection算法
        #     detections_raw = detect_fire_smoke_with_yolov5(pil_image, conf_thres=0.5)
        #     detections = []
            
        #     for det in detections_raw:
        #         if det['confidence'] > 0.3:
        #             detection = {
        #                 'model_type': model_type,
        #                 'class_name': det['class'],
        #                 'confidence': det['confidence'],
        #                 'bbox': det['bbox'],
        #                 'frame_number': frame_count,
        #                 'method': det.get('method', 'fire_detection_yolov5')
        #             }
        #             detections.append(detection)
        # else:
        # 使用原始检测逻辑
        logger.info(f"🔍 开始模型推理，置信度阈值: {MODEL_CONFIGS[model_type]['threshold']}")
        # 并发信号量保护GPU显存：限制同时推理
        with inference_semaphore:
            results = models[model_type](pil_image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if confidence > MODEL_CONFIGS[model_type]['threshold']:
                        class_name = models[model_type].names[class_id]
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        detection = {
                            'model_type': model_type,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': xyxy.tolist(),
                            'frame_number': frame_count,
                            'method': 'original_model'
                        }
                        logger.info(f"检测到 {class_name} 置信度 {confidence:.2f}")
                        detections.append(detection)
        
        return detections
    except Exception as e:
        logger.error(f"模型 {model_type} 检测失败: {str(e)}")
        return []

def process_multi_model_rtsp_detection():
    """多模型RTSP流检测处理函数 - 支持图像流返回和多线程检测"""
    global rtsp_detection_active, rtsp_stream_url, rtsp_detection_results
    global rtsp_push_frequency, rtsp_multi_model_threads
    global rtsp_current_frame, rtsp_detection_frame_results, video_frame_count
    
    logger.info(f"开始多模型RTSP流检测: {rtsp_stream_url}, 推送频率: {rtsp_push_frequency}秒")
    
    try:
        # 打开RTSP流
        cap = cv2.VideoCapture(rtsp_stream_url)
        if not cap.isOpened():
            logger.error(f"无法打开RTSP流: {rtsp_stream_url}")
            return
        
        frame_count = 0
        last_detection_time = 0  # 上次检测时间
        last_alert_time = {}  # 记录每种类型的最后报警时间
        
        # 定义要检测的模型类型
        model_types = ['fire_smoke', 'safety_harness', 'safety_helmet', 'fall_detection']
        
        while rtsp_detection_active:
            ret, frame = cap.read()
            if not ret:
                logger.warning("RTSP流读取失败，尝试重连...")
                time.sleep(1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # 更新当前帧（用于前端显示）
            with rtsp_current_frame_lock:
                rtsp_current_frame = frame.copy()
                video_frame_count = frame_count  # 同步帧计数器
            
            # 根据推送频率控制检测频率
            should_detect = (current_time - last_detection_time) >= rtsp_push_frequency
            
            if should_detect:
                try:
                    # 将帧转换为PIL图像
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # 使用多线程并行检测
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    
                    all_detections = []
                    
                    with ThreadPoolExecutor(max_workers=len(model_types)) as executor:
                        # 提交所有基础模型检测任务
                        future_to_model = {
                            executor.submit(detect_with_model, model_type, pil_image, frame_count): model_type
                            for model_type in model_types
                        }
                        
                        # 收集基础模型检测结果
                        for future in as_completed(future_to_model):
                            model_type = future_to_model[future]
                            try:
                                detections = future.result()
                                all_detections.extend(detections)
                            except Exception as e:
                                logger.error(f"多线程检测 - 模型 {model_type} 失败: {str(e)}")
                    
                    # 处理检测结果并生成报警
                    for detection in all_detections:
                        class_name = detection['class_name']
                        model_type = detection['model_type']
                        confidence = detection['confidence']
                        
                        # 检查是否需要报警
                        if class_name in ALERT_CODES:
                            # 检查推送频率限制
                            alert_key = f"{model_type}_{class_name}"
                            last_time = last_alert_time.get(alert_key, 0)
                            
                            if current_time - last_time >= rtsp_push_frequency:
                                alert_data = {
                                    "timestamp": datetime.now().isoformat(),
                                    "alert_code": ALERT_CODES[class_name],
                                    "alert_type": class_name,
                                    "confidence": confidence,
                                    "frame_number": frame_count,
                                    "model_type": model_type,
                                    "detection_mode": "multi_model",
                                    "bbox": detection['bbox'],
                                    "type": "yolo"
                                }
                                
                                # 异步调度报警推送，避免阻塞检测主循环
                                queued = schedule_alert_push(rtsp_push_url, frame, alert_data, ctx['camera_id'], ctx['camera_name'])
                                alert_data["alert_sent"] = queued
                                
                                # 添加到检测结果
                                with rtsp_detection_lock:
                                    rtsp_detection_results.append(alert_data)
                                    # 保持结果列表不超过1000条
                                    if len(rtsp_detection_results) > 1000:
                                        rtsp_detection_results = rtsp_detection_results[-500:]
                                    
                                    # 更新最后报警时间
                                    last_alert_time[alert_key] = current_time
                                    
                                    logger.info(f"多模型RTSP检测到 {class_name} (置信度: {confidence:.2f}) 在第{frame_count}帧, 模型: {model_type}")
                    
                    # 存储当前帧的检测结果（用于可视化）
                    rtsp_detection_frame_results[frame_count] = all_detections
                    
                    # 清理旧的帧结果（保留最近100帧）
                    if len(rtsp_detection_frame_results) > 100:
                        old_frames = sorted(rtsp_detection_frame_results.keys())[:-100]
                        for old_frame in old_frames:
                            del rtsp_detection_frame_results[old_frame]
                    
                    last_detection_time = current_time
                    
                except Exception as e:
                    logger.error(f"多模型RTSP帧处理失败: {str(e)}")
                    continue
            
            # 短暂休眠以避免过度占用CPU
            time.sleep(0.01)
        
        cap.release()
        logger.info("多模型RTSP流检测结束")
        
    except Exception as e:
        logger.error(f"多模型RTSP流检测异常: {str(e)}")
    finally:
        rtsp_detection_active = False

def process_qwen3vl_independent_detection():
    """独立的Qwen3-VL大模型检测线程函数 - 每5秒检测一次，不影响主线程"""
    global qwen3vl_detection_active, rtsp_current_frame, qwen3vl_detection_results
    global qwen3vl_detection_interval, qwen3vl_last_alert_time, rtsp_push_frequency
    
    logger.info(f"启动独立的Qwen3-VL大模型检测线程，检测间隔: {qwen3vl_detection_interval}秒")
    
    try:
        while qwen3vl_detection_active:
            try:
                # 获取当前帧
                current_frame = None
                with rtsp_current_frame_lock:
                    if rtsp_current_frame is not None:
                        current_frame = rtsp_current_frame.copy()
                
                # 如果有当前帧且大模型已加载，进行检测
                if current_frame is not None and qwen3vl_model_loaded:
                    # 转换为PIL图像
                    frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # 执行Qwen3-VL检测
                    qwen3vl_result = analyze_safety_with_qwen3vl(pil_image)
                    
                    if qwen3vl_result:
                        current_time = time.time()
                        
                        # 处理检测结果
                        for detection_type, result in qwen3vl_result.items():
                            if result['detected'] and result['confidence'] > 0.9:
                                # 映射检测类型到标准类名
                                class_mapping = {
                                    'smoke': 'smoke',
                                    'fire': 'fire', 
                                    'fall': 'Fall-Detected',
                                    'no_harness': 'no harness',
                                    'no_helmet': 'head'
                                }
                                
                                if detection_type in class_mapping:
                                    class_name = class_mapping[detection_type]
                                    
                                    # 检查是否需要报警
                                    if class_name in ALERT_CODES:
                                        # 检查推送频率限制
                                        alert_key = f"qwen3vl_{class_name}"
                                        last_time = qwen3vl_last_alert_time.get(alert_key, 0)
                                        
                                        if current_time - last_time >= rtsp_push_frequency:
                                            alert_data = {
                                                "timestamp": datetime.now().isoformat(),
                                                "alert_code": ALERT_CODES[class_name],
                                                "alert_type": class_name,
                                                "confidence": result['confidence'],
                                                "frame_number": video_frame_count,
                                                "model_type": "qwen3vl_multimodal",
                                                "detection_mode": "independent_qwen3vl",
                                                "bbox": [0, 0, current_frame.shape[1], current_frame.shape[0]],  # 全图检测
                                                "description": result['description'],
                                                "type": "多模态大模型"
                                            }
                                            
                                            # 发送报警通知
                                            alert_sent = send_alert_notification(alert_data)
                                            alert_data["alert_sent"] = alert_sent
                                            
                                            # 添加到检测结果
                                            with qwen3vl_detection_lock:
                                                qwen3vl_detection_results.append(alert_data)
                                                # 保持结果列表不超过500条
                                                if len(qwen3vl_detection_results) > 500:
                                                    qwen3vl_detection_results = qwen3vl_detection_results[-250:]
                                            
                                            # 同时添加到主检测结果中
                                            with rtsp_detection_lock:
                                                rtsp_detection_results.append(alert_data)
                                                if len(rtsp_detection_results) > 1000:
                                                    rtsp_detection_results = rtsp_detection_results[-500:]
                                            
                                            # 更新最后报警时间
                                            qwen3vl_last_alert_time[alert_key] = current_time
                                            
                                            logger.info(f"独立Qwen3-VL检测到 {class_name}: {result['description']} (置信度: {result['confidence']:.2f})")
                
                # 等待指定的检测间隔
                time.sleep(qwen3vl_detection_interval)
                
            except Exception as e:
                logger.error(f"独立Qwen3-VL检测处理失败: {str(e)}")
                time.sleep(qwen3vl_detection_interval)  # 出错时也要等待，避免频繁重试
                continue
        
        logger.info("独立Qwen3-VL大模型检测线程结束")
        
    except Exception as e:
        logger.error(f"独立Qwen3-VL检测线程异常: {str(e)}")
    finally:
        qwen3vl_detection_active = False

def process_rtsp_detection():
    """RTSP流多线程检测处理函数"""
    global rtsp_detection_active, rtsp_stream_url, rtsp_detection_results
    
    logger.info(f"开始RTSP流检测: {rtsp_stream_url}")
    
    try:
        # 打开RTSP流
        cap = cv2.VideoCapture(rtsp_stream_url)
        if not cap.isOpened():
            logger.error(f"无法打开RTSP流: {rtsp_stream_url}")
            return
        
        frame_count = 0
        detection_interval = 30  # 每30帧检测一次
        
        while rtsp_detection_active:
            ret, frame = cap.read()
            if not ret:
                logger.warning("RTSP流读取失败，尝试重连...")
                time.sleep(1)
                continue
            
            frame_count += 1
            
            # 按间隔进行检测
            if frame_count % detection_interval == 0:
                try:
                    # 将帧转换为PIL图像
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # 将图像转换为base64格式
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format='JPEG')
                    img_buffer.seek(0)
                    image_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    
                    # 对所有四个模型进行检测
                    model_types = ['fire_smoke', 'safety_harness', 'fall_detection']
                    
                    for model_type in model_types:
                        if model_type not in models or not model_loaded.get(model_type, False):
                            continue
                        
                        try:
                            # 进行检测
                            results = models[model_type](pil_image, conf=0.5)
                            
                            for result in results:
                                boxes = result.boxes
                                if boxes is not None and len(boxes) > 0:
                                    for box in boxes:
                                        # 获取检测信息
                                        class_id = int(box.cls[0])
                                        confidence = float(box.conf[0])
                                        class_name = models[model_type].names[class_id]
                                        
                                        # 检查是否需要报警
                                        if class_name in ALERT_CODES:
                                            alert_data = {
                                            "timestamp": datetime.now().isoformat(),
                                            "alert_code": ALERT_CODES[class_name],
                                            "alert_type": class_name,
                                            "confidence": confidence,
                                            "frame_number": frame_count,
                                            "model_type": model_type,
                                            "bbox": [x1, y1, x2, y2],  # 添加边界框信息
                                            "type": "yolo"
                                        }
                                        
                                        # 异步调度报警推送，避免阻塞检测主循环
                                        queued = schedule_alert_push(rtsp_push_url, frame, alert_data, ctx['camera_id'], ctx['camera_name'])
                                        alert_data["alert_sent"] = queued
                                        
                                        # 添加到检测结果
                                        with rtsp_detection_lock:
                                            rtsp_detection_results.append(alert_data)
                                            # 保持结果列表不超过1000条
                                            if len(rtsp_detection_results) > 1000:
                                                rtsp_detection_results = rtsp_detection_results[-500:]
                                            
                                            logger.info(f"RTSP检测到 {class_name} (置信度: {confidence:.2f}) 在第{frame_count}帧")
                        
                        except Exception as e:
                            logger.error(f"模型 {model_type} 检测失败: {str(e)}")
                            continue
                
                except Exception as e:
                    logger.error(f"RTSP帧处理失败: {str(e)}")
                    continue
            
            # 短暂休眠以避免过度占用CPU
            time.sleep(0.01)
        
        cap.release()
        logger.info("RTSP流检测结束")
        
    except Exception as e:
        logger.error(f"RTSP流检测异常: {str(e)}")
    finally:
        rtsp_detection_active = False

@app.route('/api/ocr_mode', methods=['GET', 'POST'])
def api_ocr_mode():
    """获取或设置OCR模式"""
    global ocr_reader

    if request.method == 'GET':
        # 获取当前OCR模式
        if hasattr(ocr_reader, 'ocr_mode'):
            return jsonify({
                'success': True,
                'ocr_mode': ocr_reader.ocr_mode,
                'use_easyocr': ocr_reader.use_easyocr,
                'use_paddleocr': ocr_reader.use_paddleocr,
                'available_modes': ['easyocr', 'paddleocr', 'hybrid']
            })
        else:
            return jsonify({
                'success': True,
                'ocr_mode': 'easyocr',
                'use_easyocr': True,
                'use_paddleocr': False,
                'available_modes': ['easyocr']
            })

    elif request.method == 'POST':
        # 设置OCR模式
        try:
            data = request.get_json()
            new_mode = data.get('ocr_mode', 'hybrid')

            if hasattr(ocr_reader, 'set_ocr_mode'):
                ocr_reader.set_ocr_mode(new_mode)
                return jsonify({
                    'success': True,
                    'message': f'OCR模式已设置为: {new_mode}',
                    'ocr_mode': new_mode
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '当前OCR不支持模式切换'
                })

        except Exception as e:
            logger.error(f"设置OCR模式失败: {str(e)}")
            return jsonify({'success': False, 'error': str(e)})

@app.route('/api/rtsp/start', methods=['POST'])
def api_start_rtsp_detection():
    """启动RTSP流检测"""
    global rtsp_detection_active, rtsp_detection_thread, rtsp_stream_url, rtsp_push_url
    global rtsp_multi_model_mode, rtsp_push_frequency, rtsp_multi_model_threads
    global qwen3vl_detection_active, qwen3vl_detection_thread
    
    try:
        data = request.get_json()
        stream_url = data.get('stream_url', '').strip()
        push_url = data.get('push_url', '').strip()
        multi_model = data.get('multi_model', False)
        push_frequency = data.get('push_frequency', 30)
        
        if not stream_url:
            return jsonify({
                'success': False,
                'error': 'RTSP流地址不能为空'
            })
        
        if rtsp_detection_active:
            return jsonify({
                'success': False,
                'error': 'RTSP检测已在运行中'
            })
        
        # 设置全局变量
        rtsp_stream_url = stream_url
        rtsp_push_url = push_url
        rtsp_multi_model_mode = multi_model
        rtsp_push_frequency = push_frequency
        rtsp_detection_active = True
        
        # 清空之前的检测结果
        with rtsp_detection_lock:
            rtsp_detection_results.clear()
        
        # 清空大模型检测结果
        with qwen3vl_detection_lock:
            qwen3vl_detection_results.clear()
        
        # 根据检测模式启动相应的线程
        if multi_model:
            # 多模型同时检测模式
            logger.info(f"启动多模型RTSP检测: {stream_url}, 推送频率: {push_frequency}秒")
            rtsp_detection_thread = threading.Thread(target=process_multi_model_rtsp_detection)
        else:
            # 单模型检测模式
            logger.info(f"启动单模型RTSP检测: {stream_url}")
            rtsp_detection_thread = threading.Thread(target=process_rtsp_detection)
        
        rtsp_detection_thread.daemon = True
        rtsp_detection_thread.start()
        
        # 启动独立的大模型检测线程
        if qwen3vl_model_loaded and not qwen3vl_detection_active:
            qwen3vl_detection_active = True
            qwen3vl_detection_thread = threading.Thread(target=process_qwen3vl_independent_detection)
            qwen3vl_detection_thread.daemon = True
            qwen3vl_detection_thread.start()
            logger.info(f"启动独立大模型检测线程，检测间隔: {qwen3vl_detection_interval}秒")
        
        mode_text = "多模型同时检测" if multi_model else "单模型检测"
        qwen3vl_status = "已启动" if qwen3vl_model_loaded and qwen3vl_detection_active else "未启动"
        
        return jsonify({
            'success': True,
            'message': f'RTSP {mode_text} 已启动',
            'stream_url': stream_url,
            'push_url': push_url,
            'multi_model': multi_model,
            'push_frequency': push_frequency,
            'qwen3vl_detection': qwen3vl_status,
            'qwen3vl_interval': qwen3vl_detection_interval,
            'alert_codes': ALERT_CODES
        })
        
    except Exception as e:
        logger.error(f"启动RTSP检测失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'启动RTSP检测失败: {str(e)}'
        })

@app.route('/api/rtsp/stop', methods=['POST'])
def api_stop_rtsp_detection():
    """停止RTSP流检测"""
    global rtsp_detection_active, rtsp_detection_thread, rtsp_multi_model_mode, rtsp_multi_model_threads
    global qwen3vl_detection_active, qwen3vl_detection_thread
    
    try:
        if not rtsp_detection_active:
            return jsonify({
                'success': False,
                'error': 'RTSP检测未在运行'
            })
        
        # 停止检测
        rtsp_detection_active = False
        
        # 停止独立的大模型检测线程
        if qwen3vl_detection_active:
            qwen3vl_detection_active = False
            if qwen3vl_detection_thread and qwen3vl_detection_thread.is_alive():
                qwen3vl_detection_thread.join(timeout=3)
            logger.info("独立大模型检测线程已停止")
        
        # 等待线程结束
        if rtsp_detection_thread and rtsp_detection_thread.is_alive():
            rtsp_detection_thread.join(timeout=5)
        
        # 清理多模型检测线程
        if rtsp_multi_model_mode:
            for thread_name, thread in rtsp_multi_model_threads.items():
                if thread and thread.is_alive():
                    thread.join(timeout=2)
            rtsp_multi_model_threads.clear()
            rtsp_multi_model_mode = False
        
        logger.info("RTSP检测已停止")
        
        return jsonify({
            'success': True,
            'message': 'RTSP检测已停止',
            'qwen3vl_detection': '已停止'
        })
        
    except Exception as e:
        logger.error(f"停止RTSP检测失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'停止RTSP检测失败: {str(e)}'
        })



@app.route('/api/rtsp/status')
def api_rtsp_status():
    """获取当前RTSP任务状态"""
    global rtsp_detection_active, rtsp_stream_url, rtsp_push_url, rtsp_multi_model_mode, rtsp_push_frequency
    
    if rtsp_detection_active:
        return jsonify({
            'active': True,
            'stream_url': rtsp_stream_url,
            'push_url': rtsp_push_url,
            'multi_model_mode': rtsp_multi_model_mode,
            'push_frequency': rtsp_push_frequency
        })
    else:
        return jsonify({
            'active': False
        })

@app.route('/api/rtsp/results', methods=['GET'])
def api_rtsp_results():
    """获取RTSP检测结果"""
    global rtsp_detection_results
    
    try:
        with rtsp_detection_lock:
            results = rtsp_detection_results.copy()
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"获取RTSP检测结果失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取检测结果失败: {str(e)}'
        })

@app.route('/api/rtsp/clear_results', methods=['POST'])
def api_clear_rtsp_results():
    """清空RTSP检测结果"""
    global rtsp_detection_results
    
    try:
        with rtsp_detection_lock:
            rtsp_detection_results.clear()
        
        logger.info("RTSP检测结果已清空")
        return jsonify({
            'success': True,
            'message': 'RTSP检测结果已清空'
        })
        
    except Exception as e:
        logger.error(f"清空RTSP检测结果失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'清空RTSP检测结果失败: {str(e)}'
        })

@app.route('/api/rtsp/stream')
def api_rtsp_stream():
    """获取RTSP实时图像流 - 根据检测结果返回标注图像或原图"""
    def generate_frames():
        global rtsp_current_frame, rtsp_current_frame_lock, rtsp_detection_active
        global rtsp_detection_frame_results, video_frame_count
        
        while rtsp_detection_active:
            try:
                # 仅在复制当前帧时加锁，避免编码期间阻塞生产者更新
                current_frame = None
                current_frame_number = 0
                with rtsp_current_frame_lock:
                    if rtsp_current_frame is not None:
                        current_frame = rtsp_current_frame.copy()
                        current_frame_number = video_frame_count

                if current_frame is not None:
                    # 查找最近几帧的检测结果（考虑到检测可能有延迟）
                    frame_detections = []
                    for frame_num in range(max(1, current_frame_number - 5), current_frame_number + 1):
                        if frame_num in rtsp_detection_frame_results:
                            frame_detections.extend(rtsp_detection_frame_results[frame_num])

                    # 如果有检测结果，绘制检测框（不持锁）
                    if frame_detections:
                        try:
                            annotated_frame = draw_detection_boxes(current_frame, frame_detections)
                            current_frame = annotated_frame
                            logger.debug(f"帧 {current_frame_number} 绘制了 {len(frame_detections)} 个检测框")
                        except Exception as e:
                            logger.error(f"绘制检测框失败: {str(e)}")
                            # 如果绘制失败，使用原图
                            pass

                    # 将OpenCV图像编码为JPEG（不持锁）
                    ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.1)  # 控制帧率，避免过高的CPU使用率
                
            except Exception as e:
                logger.error(f"生成图像流帧失败: {str(e)}")
                break
    
    if not rtsp_detection_active:
        return jsonify({
            'success': False,
            'error': 'RTSP检测未启动'
        }), 400
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/mock/alert_receiver', methods=['POST'])
def api_mock_alert_receiver():
    """模拟报警接收接口"""
    # global external_alert_notifications, external_alert_lock
    
    try:
        data = request.get_json()['AlarmObject'][0]
        
        # 打印接收到的报警数据
        print("=" * 50)
        print("📢 收到报警推送:")
        print(f"时间: {data.get('alarmTime', 'N/A')}")
        print(f"描述: {data.get('Describe', 'N/A')}")
        print(f"算法ID: {data.get('algoId', 'N/A')}")
        print(f"算法名称: {data.get('algoName', 'N/A')}")
        print(f"摄像机: {data.get('cameraName', 'N/A')}")
        print(f"摄像机ID: {data.get('cameraId', 'N/A')}")
        print(f"区域: {data.get('area', 'N/A')}")
        print("=" * 50)
        
        # 构造标准化的报警数据
        alert_notification = {
            'id': data.get('id'),
            'timestamp': data.get('alarmTime'),
            'received_time': datetime.now().isoformat(),
            'alert_type': data.get('algoName', '未知'),
            'alert_code': data.get('algoId'),
            'description': data.get('Describe', ''),
            'camera_id': data.get('cameraId'),
            'camera_name': data.get('cameraName'),
            'device_id': data.get('deviceId'),
            'device_name': data.get('deviceName'),
            'area': data.get('area', ''),
            'status': data.get('status', 0),  # 0报警/1销警
            'image_base64': data.get('imageBase64', ''),
            'raw_data': data
        }
        
        # # 存储到全局变量中
        # with external_alert_lock:
        #     external_alert_notifications.append(alert_notification)
        #     # 保持列表不超过100条记录
        #     if len(external_alert_notifications) > 100:
        #         external_alert_notifications = external_alert_notifications[-50:]
        
        # 记录到日志
        logger.info(f"接收到外部报警推送: {alert_notification['alert_type']} (ID: {alert_notification['id']})")
        
        return jsonify({
            'success': True,
            'message': '报警接收成功',
            'received_data': data
        })
        
    except Exception as e:
        logger.error(f"模拟报警接收失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'报警接收失败: {str(e)}'
        })

@app.route('/api/external_alerts', methods=['GET'])
def api_get_external_alerts():
    """获取外部报警推送状态"""
    global external_alert_notifications, external_alert_lock
    
    try:
        # 获取查询参数
        limit = request.args.get('limit', 10, type=int)  # 默认返回最新10条
        since = request.args.get('since')  # 获取指定时间之后的报警
        
        with external_alert_lock:
            alerts = external_alert_notifications.copy()
        
        # 如果指定了since参数，过滤时间
        if since:
            try:
                since_time = datetime.fromisoformat(since.replace('Z', '+00:00'))
                alerts = [alert for alert in alerts 
                         if datetime.fromisoformat(alert['received_time']) > since_time]
            except ValueError:
                pass  # 忽略无效的时间格式
        
        # 按接收时间倒序排列，返回最新的记录
        alerts = sorted(alerts, key=lambda x: x['received_time'], reverse=True)
        alerts = alerts[:limit]
        
        return jsonify({
            'success': True,
            'count': len(alerts),
            'total_count': len(external_alert_notifications),
            'alerts': alerts
        })
        
    except Exception as e:
        logger.error(f"获取外部报警推送失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取外部报警推送失败: {str(e)}'
        })

@app.route('/api/external_alerts/clear', methods=['POST'])
def api_clear_external_alerts():
    """清空外部报警推送记录"""
    global external_alert_notifications, external_alert_lock
    
    try:
        with external_alert_lock:
            cleared_count = len(external_alert_notifications)
            external_alert_notifications.clear()
        
        logger.info(f"清空了 {cleared_count} 条外部报警推送记录")
        
        return jsonify({
            'success': True,
            'message': f'已清空 {cleared_count} 条外部报警推送记录',
            'cleared_count': cleared_count
        })
        
    except Exception as e:
        logger.error(f"清空外部报警推送失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'清空外部报警推送失败: {str(e)}'
        })

if __name__ == '__main__':
    # 启动时加载模型
    print("🚀 启动智能工业安全检测系统...")
    logger.info("启动智能工业安全检测系统...")

    print("📦 开始加载模型...")
    load_models()
    print("✅ 模型加载完成")

    # 初始化阿里云API
    print("🌐 初始化阿里云DashScope API...")
    logger.info("初始化阿里云DashScope API...")
    init_dashscope_api()
    print("✅ API初始化完成")

    # 启动时更新一次相机信息
    print("📷 更新相机信息...")
    update_camera_info()
    print("✅ 相机信息更新完成")

    def camera_info_refresh_scheduler():
        """定时刷新相机信息"""
        while True:
            time.sleep(CAMERA_INFO_REFRESH_INTERVAL)
            logger.info("开始定时刷新相机信息...")
            update_camera_info()

    # 启动定时刷新线程
    print("🔄 启动相机信息定时刷新线程...")
    refresh_thread = threading.Thread(target=camera_info_refresh_scheduler, daemon=True)
    refresh_thread.start()
    print("✅ 定时刷新线程已启动")

    # 启动Flask应用
    print("🌐 启动Flask应用...")
    # 从环境变量读取端口，默认8003
    PORT = int(os.getenv('PORT', '8003'))
    print(f"📍 应用将在 http://127.0.0.1:{PORT} 启动")
    app.run(host='0.0.0.0', port=PORT, debug=False)
    # load_models()
    # process_image('/home/ysaiot/data_3/tkz_git_project/YOLOv8-Fire-and-Smoke-Detection/datasets/D-fire-yolov11/val/images/WEB06075.jpg')
