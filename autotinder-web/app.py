from flask import Flask, render_template, jsonify, Response, request
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
import threading
import queue
import logging
import sys
import json
import time
import uiautomator2 as u2
import math
import random
import os
import base64
import io
from datetime import datetime
from PIL import Image
import hashlib
import numpy as np
from io import BytesIO
import cv2
from google import genai
from google.genai import types
from requests.exceptions import RequestException, Timeout, ConnectionError

# Import database models
from models import db, Settings, ProfileLog

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tinder_automation.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'tinder_automation_secret'

# Initialize database
db.init_app(app)

# Initialize SocketIO with minimal logging
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', 
                   engineio_logger=False, socketio_logger=False)

class WebConsoleHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        # Define patterns to suppress
        self.suppress_patterns = [
            'AFC is enabled with max remote calls',
            'AFC remote call',
            'is done'
        ]
        self.suppress_loggers = [
            'google.genai',
            'google.ai.generativelanguage', 
            'google.auth',
            'httpx',
            'httpcore',
            'urllib3',
            'requests',
            'werkzeug',
            'AFC'
        ]
        
    def emit(self, record):
        try:
            # Filter out unwanted loggers
            if any(record.name.startswith(logger) for logger in self.suppress_loggers):
                return
            
            # Filter out unwanted message patterns
            msg = self.format(record)
            if any(pattern in msg for pattern in self.suppress_patterns):
                return
                
            self.log_queue.put(('log', msg))
        except Exception:
            pass

class ScreenMirror:
    def __init__(self, quality=80, scale_factor=0.4, target_fps=25):
        self.quality = quality
        self.scale_factor = scale_factor
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.device = None
        self.capturing = False
        self.clients = set()
        
        self._cached_size = None
        self._jpeg_params = {
            'format': 'JPEG',
            'quality': quality,
            'optimize': False,
            'progressive': False,
            'subsampling': 2,
            'qtables': 'web_low'
        }
        
    def set_device(self, device):
        """Set the device from automation manager"""
        self.device = device
        if device and self.scale_factor < 1.0:
            try:
                test_screenshot = device.screenshot()
                new_width = int(test_screenshot.width * self.scale_factor)
                new_height = int(test_screenshot.height * self.scale_factor)
                self._cached_size = (new_width, new_height)
            except:
                self._cached_size = None
    
    def capture_screenshot_fast(self):
        """Ultra-fast screenshot capture"""
        try:
            if not self.device:
                return None
            
            screenshot = self.device.screenshot()
            
            if self._cached_size and (screenshot.width, screenshot.height) != self._cached_size:
                screenshot = screenshot.resize(self._cached_size, Image.Resampling.NEAREST)
            
            img_buffer = io.BytesIO()
            screenshot.save(img_buffer, **self._jpeg_params)
            img_bytes = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('ascii')
            return f"data:image/jpeg;base64,{img_base64}"
            
        except:
            return None
    
    def capture_loop(self):
        """High-performance capture loop"""
        next_frame_time = time.time()
        
        while self.capturing:
            current_time = time.time()
            
            if current_time < next_frame_time:
                sleep_time = next_frame_time - current_time
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                current_time = time.time()
            
            screenshot_data = self.capture_screenshot_fast()
            
            if screenshot_data:
                try:
                    self.frame_queue.put_nowait({
                        'data': screenshot_data,
                        'timestamp': current_time
                    })
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait({
                            'data': screenshot_data,
                            'timestamp': current_time
                        })
                    except:
                        pass
            
            next_frame_time = current_time + self.frame_interval
            
    def broadcast_loop(self):
        """Broadcast frames to clients"""
        while self.capturing:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                if self.clients:
                    socketio.emit('frame_update', {
                        'frame': frame_data['data'],
                        'timestamp': frame_data['timestamp']
                    })
                    
            except queue.Empty:
                continue
            except:
                continue
    
    def start_capture(self):
        """Start the capture process"""
        if not self.device:
            return False
            
        if self.capturing:
            return True
            
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self.capturing = True
        
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.broadcast_thread = threading.Thread(target=self.broadcast_loop, daemon=True)
        
        self.capture_thread.start()
        self.broadcast_thread.start()
        
        return True
    
    def stop_capture(self):
        """Stop the capture process"""
        self.capturing = False
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2)
        if hasattr(self, 'broadcast_thread'):
            self.broadcast_thread.join(timeout=2)

class AndroidAutomator:
    def __init__(self, log_queue, settings_dict, device_id=None):
        self.log_queue = log_queue
        self.settings_dict = settings_dict
        self._validate_settings()
        
        # Load configuration from settings dictionary (NO DATABASE ACCESS)
        self.api_key = self.settings_dict['gemini_api_key']
        self.decision_prompt = self.settings_dict['decision_prompt']
        self.ui_resolver_prompt = self.settings_dict['ui_resolver_prompt']
        self.model_heavy = self.settings_dict['model_heavy']
        self.model_fast = self.settings_dict['model_fast']
        self.short_wait = self.settings_dict['short_wait']
        self.medium_wait = self.settings_dict['medium_wait']
        self.long_wait = self.settings_dict['long_wait']
        self.package_name = self.settings_dict['package_name']
        self.decision_criteria = self.settings_dict['decision_criteria']
        
        self.device = u2.connect(device_id) if device_id else u2.connect()
        self.running = True
        
        # Setup logging
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Suppress verbose library logging
        for logger_name in ['google.genai', 'google.ai.generativelanguage', 'google.auth', 'httpx', 'httpcore', 'urllib3', 'requests', 'werkzeug', 'AFC']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        # Device setup
        self.device_info = self.device.info
        self.device_width = self.device_info['displayWidth']
        self.device_height = self.device_info['displayHeight']
        
        dummy_screenshot = self.screenshot_to_memory()
        if dummy_screenshot:
            self.screenshot_width = dummy_screenshot.width
            self.screenshot_height = dummy_screenshot.height
            dummy_screenshot = None
        else:
            self.screenshot_width = 1080
            self.screenshot_height = 2400
        
        self.scale_x = self.device_width / self.screenshot_width
        self.scale_y = self.device_height / self.screenshot_height
        
        # Button detection configuration
        self.button_config = {
            'like': {'template': 'like.png', 'scales': [0.3, 0.5, 0.7, 0.9, 1.1], 'threshold': 0.65, 'ratio': (0.7, 1.4)},
            'dislike': {'template': 'dislike.png', 'scales': [0.3, 0.5, 0.7, 0.9, 1.1], 'threshold': 0.65, 'ratio': (0.7, 1.4)},
            'scroll': {'template': 'scroll.png', 'scales': [0.4, 0.6, 0.8, 1.0, 1.2], 'threshold': 0.7, 'ratio': (0.5, 2.0)}
        }
        
        self.genai_client = genai.Client(api_key=self.api_key)
        
        # Profile storage setup
        self.analyzed_profiles_dir = "analyzed_profiles"
        self._create_profiles_directory()
        
        self.profile_counts = {'total': 0, 'liked': 0, 'disliked': 0}
        
        self.web_print(f"Device: {self.device_info['productName']} ({self.device_width}x{self.device_height})")
        self.web_print("Automation initialized successfully")
    
    def reload_settings(self, settings_dict):
        """Reload settings from dictionary (NO DATABASE ACCESS)"""
        self.settings_dict = settings_dict
        self.api_key = self.settings_dict['gemini_api_key']
        self.decision_prompt = self.settings_dict['decision_prompt']
        self.ui_resolver_prompt = self.settings_dict['ui_resolver_prompt']
        self.model_heavy = self.settings_dict['model_heavy']
        self.model_fast = self.settings_dict['model_fast']
        self.short_wait = self.settings_dict['short_wait']
        self.medium_wait = self.settings_dict['medium_wait']
        self.long_wait = self.settings_dict['long_wait']
        self.package_name = self.settings_dict['package_name']
        self.decision_criteria = self.settings_dict['decision_criteria']
        
        # Recreate genai client if API key changed
        self.genai_client = genai.Client(api_key=self.api_key)
    
    def web_print(self, message):
        """Send message to web console"""
        if self.log_queue and message and str(message).strip():
            self.log_queue.put(('print', str(message).strip()))
    
    def _validate_settings(self):
        if not self.settings_dict['gemini_api_key'] or self.settings_dict['gemini_api_key'].strip() == "":
            raise ValueError("GEMINI_API_KEY cannot be empty")
        
        try:
            float(self.settings_dict['short_wait'])
            float(self.settings_dict['medium_wait']) 
            float(self.settings_dict['long_wait'])
        except (ValueError, TypeError):
            raise ValueError("Wait duration variables must be numeric")
    
    def _create_profiles_directory(self):
        try:
            if not os.path.exists(self.analyzed_profiles_dir):
                os.makedirs(self.analyzed_profiles_dir)
        except Exception as e:
            self.logger.error(f"Failed to create profiles directory: {e}")
    
    def _save_profile_screenshot(self, image_bytes, action, reason):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            action_prefix = "liked" if action == "[LIKE]" else "disliked"
            filename = f"{action_prefix}_profile_{timestamp}.jpg"
            filepath = os.path.join(self.analyzed_profiles_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            # Store profile data for later database insertion (NO DIRECT DB ACCESS)
            profile_data = {
                'action': action.replace('[', '').replace(']', ''),
                'reason': reason,
                'screenshot_path': filepath,
                'timestamp': datetime.now()
            }
            
            # Add to queue for database insertion in main thread
            self.log_queue.put(('profile_log', profile_data))
            
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save profile screenshot: {e}")
            return None
    
    def wait(self, seconds):
        """Wait with interrupt checking"""
        end_time = time.time() + seconds
        while time.time() < end_time and self.running:
            time.sleep(0.1)
        return self.running
    
    def screenshot_to_memory(self):
        try:
            buffer = BytesIO()
            self.device.screenshot().save(buffer, format='PNG')
            buffer.seek(0)
            image = Image.open(buffer).copy()
            buffer.close()
            return image
        except Exception as e:
            self.logger.error(f"Memory screenshot failed: {e}")
            return None
    
    def launch(self, package_name):
        try:
            self.device.app_start(package_name)
            return True
        except Exception as e:
            self.logger.error(f"Launch failed: {e}")
            return False
    
    def tap(self, x1, y1, x2, y2, click_radius_percent=90):
        try:
            width = x2 - x1
            height = y2 - y1
            screenshot_center_x = (x1 + x2) // 2
            screenshot_center_y = (y1 + y2) // 2
            
            max_radius = min(width, height) // 2
            fingertip_radius = (max_radius * click_radius_percent) / 100
            safe_radius = max(0, max_radius - fingertip_radius)
            
            if safe_radius > 0:
                angle = random.uniform(0, 2 * math.pi)
                random_radius = safe_radius * math.sqrt(random.uniform(0, 1))
                x_offset = int(random_radius * math.cos(angle))
                y_offset = int(random_radius * math.sin(angle))
                
                screenshot_x = screenshot_center_x + x_offset
                screenshot_y = screenshot_center_y + y_offset
                
                screenshot_x = max(x1 + fingertip_radius, min(x2 - fingertip_radius, screenshot_x))
                screenshot_y = max(y1 + fingertip_radius, min(y2 - fingertip_radius, screenshot_y))
            else:
                screenshot_x = screenshot_center_x
                screenshot_y = screenshot_center_y
            
            device_x = int(screenshot_x * self.scale_x)
            device_y = int(screenshot_y * self.scale_y)
            
            self.device.click(device_x, device_y)
            return True
        except Exception as e:
            self.logger.error(f"Tap failed: {e}")
            return False
    
    def tap_coordinates(self, x, y):
        try:
            self.device.click(x, y)
            return True
        except Exception as e:
            self.logger.error(f"Coordinate tap failed: {e}")
            return False
    
    def pil_to_cv2(self, pil_image):
        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.logger.error(f"PIL to CV2 conversion failed: {e}")
            return None
    
    def detect_buttons_from_pil(self, pil_image):
        try:
            img = self.pil_to_cv2(pil_image)
            if img is None:
                return {}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h_img, w_img = img.shape[:2]
            results = {}
            
            for name, cfg in self.button_config.items():
                template = cv2.imread(cfg['template'])
                if template is None:
                    results[name] = None
                    continue
                
                gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                th, tw = gray_tmpl.shape
                matches = []
                
                for scale in cfg['scales']:
                    w, h = int(tw * scale), int(th * scale)
                    if w < 10 or h < 10 or w > min(w_img, 300) or h > min(h_img, 300):
                        continue
                    
                    scaled = cv2.resize(gray_tmpl, (w, h))
                    result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                    _, conf, _, (x, y) = cv2.minMaxLoc(result)
                    
                    if conf >= cfg['threshold'] and cfg['ratio'][0] <= w/h <= cfg['ratio'][1]:
                        matches.append((conf, x, y, w, h))
                
                if matches:
                    matches.sort(reverse=True)
                    _, x, y, w, h = matches[0]
                    results[name] = [x, y, x + w, y + h]
                else:
                    results[name] = None
            
            return results
        except Exception as e:
            self.logger.error(f"Button detection failed: {e}")
            return {}
    
    def resolve_ui(self, screenshot_image):
        try:
            buffer = BytesIO()
            screenshot_image.save(buffer, format='JPEG')
            screenshot_bytes = buffer.getvalue()
            buffer.close()
            
            response_text = self._make_api_request(
                image_bytes=screenshot_bytes,
                text_prompt=None,
                system_prompt=self.ui_resolver_prompt,
                temperature=0.45,
                model=self.model_fast,
                max_retries=3,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                media_resolution="MEDIA_RESOLUTION_MEDIUM"
            )
            
            if isinstance(response_text, dict) and response_text.get("action"):
                return []
            
            bounding_boxes = json.loads(response_text)
            if not bounding_boxes:
                return []
            
            bbox = bounding_boxes[0]
            ymin, xmin, ymax, xmax = bbox["box_2d"]
            
            width, height = screenshot_image.size
            abs_x1 = int(xmin / 1000 * width)
            abs_y1 = int(ymin / 1000 * height)
            abs_x2 = int(xmax / 1000 * width)
            abs_y2 = int(ymax / 1000 * height)
            
            return [abs_x1, abs_y1, abs_x2, abs_y2]
            
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"UI resolution failed: {e}")
            return []
    
    def _check_and_resolve_overlays(self, max_attempts=5):
        for attempt in range(max_attempts):
            if not self.running:
                return False
                
            current_screenshot = self.screenshot_to_memory()
            if current_screenshot is None:
                continue
            
            ui_coordinates = self.resolve_ui(current_screenshot)
            
            if not ui_coordinates or len(ui_coordinates) != 4:
                return True
            
            self.web_print(f"Overlay detected, resolving... (attempt {attempt + 1}/{max_attempts})")
            x1, y1, x2, y2 = ui_coordinates
            
            if not self.tap(x1, y1, x2, y2):
                if not self.wait(self.short_wait):
                    return False
                continue
            
            if not self.wait(self.medium_wait):
                return False
        
        return False
    
    def detect_buttons_with_clean_ui(self, required_buttons, max_attempts=3):
        if not self._check_and_resolve_overlays():
            if not self.running:
                return None, None
            return None, None
        
        for attempt in range(max_attempts):
            if not self.running:
                return None, None
            
            current_screenshot = self.screenshot_to_memory()
            if current_screenshot is None:
                continue
            
            detected_buttons = self.detect_buttons_from_pil(current_screenshot)
            missing_buttons = [btn for btn in required_buttons if detected_buttons.get(btn) is None]
            
            if not missing_buttons:
                return detected_buttons, current_screenshot
            
            self.web_print(f"Missing buttons: {missing_buttons}, retrying...")
            if not self.wait(self.medium_wait):
                return None, None
        
        return None, None
    
    def _make_api_request(self, image_bytes, text_prompt=None, system_prompt="", temperature=0.5, model=None, max_retries=3, **config_kwargs):
        if model is None:
            model = self.model_heavy
        
        for attempt in range(max_retries):
            if not self.running:
                return {"reason": "Interrupted", "action": "[ERROR]"}
            
            try:
                if attempt > 0:
                    backoff_time = 2 ** attempt
                    end_time = time.time() + backoff_time
                    while time.time() < end_time and self.running:
                        time.sleep(0.1)
                    if not self.running:
                        return {"reason": "Interrupted", "action": "[ERROR]"}
                
                parts = [types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes)]
                if text_prompt:
                    parts.append(types.Part.from_text(text=text_prompt))
                
                contents = [types.Content(role="user", parts=parts)]
                config_params = {"temperature": temperature, "system_instruction": [types.Part.from_text(text=system_prompt)]}
                config_params.update(config_kwargs)
                
                response = self.genai_client.models.generate_content(
                    model=model, contents=contents, config=types.GenerateContentConfig(**config_params))
                
                if not self.running or not response or not hasattr(response, 'text') or not response.text:
                    return {"reason": "No response from AI", "action": "[ERROR]"}
                
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                return response_text.strip()
                
            except Exception as e:
                if not self.running:
                    return {"reason": "Interrupted", "action": "[ERROR]"}
                if attempt == max_retries - 1:
                    return {"reason": f"API Error: {str(e)}", "action": "[ERROR]"}
        
        return {"reason": "API failed after retries", "action": "[ERROR]"}
    
    def process(self, image_bytes, criteria):
        self.web_print("Analyzing profile...")
        start_time = time.time()
        
        text_prompt = f"CRITERIA: {criteria}"
        response_text = self._make_api_request(
            image_bytes=image_bytes, text_prompt=text_prompt, system_prompt=self.decision_prompt,
            temperature=0.5, model=self.model_heavy, max_retries=3)
        
        elapsed_time = time.time() - start_time
        self.web_print(f"Analysis completed in {elapsed_time:.2f} seconds")
        
        return response_text
    
    def _controlled_scroll(self, scroll_pixels=None):
        if scroll_pixels is None:
            scroll_pixels = int(self.device_height * 0.75)
        
        try:
            center_x = self.device_width // 2
            start_y = int(self.device_height * 0.7)
            end_y = max(int(self.device_height * 0.1), start_y - scroll_pixels)
            self.device.swipe(center_x, start_y, center_x, end_y, 0.3)
            return True
        except:
            return False
    
    def _get_image_hash(self, image):
        try:
            return hashlib.md5(image.convert('L').resize((100, 100)).tobytes()).hexdigest()
        except:
            return None
    
    def _images_similar(self, img1, img2, threshold=0.95):
        try:
            size = (300, 300)
            arr1 = np.array(img1.resize(size).convert('L'))
            arr2 = np.array(img2.resize(size).convert('L'))
            correlation = np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]
            return correlation >= threshold if not np.isnan(correlation) else True
        except:
            return False
    
    def _is_duplicate_image(self, new_image, existing_images, existing_hashes):
        try:
            new_hash = self._get_image_hash(new_image)
            if new_hash in existing_hashes:
                return True
            
            recent_images = existing_images[-3:] if len(existing_images) > 3 else existing_images
            return any(self._images_similar(new_image, img, 0.98) for img in recent_images)
        except:
            return False
    
    def _cycle_through_user_photos(self, max_photos=20):
        try:
            tap_x = int(self.device_width * 0.8)
            tap_y = self.device_height // 4
            photo_images = []
            photo_hashes = []
            
            for i in range(max_photos):
                if not self.running:
                    break
                
                if not self.tap_coordinates(tap_x, tap_y):
                    break
                
                if not self.wait(self.medium_wait):
                    break
                
                current_image = self.screenshot_to_memory()
                if current_image is None:
                    break
                
                if self._is_duplicate_image(current_image, photo_images, photo_hashes):
                    break
                
                photo_images.append(current_image)
                photo_hashes.append(self._get_image_hash(current_image))
            
            return photo_images
        except Exception as e:
            self.logger.error(f"Photo cycling failed: {e}")
            return []
    
    def _stitch_in_capture_order(self, all_images):
        try:
            if not all_images:
                return None
            
            total_images = len(all_images)
            first_row_count = (total_images + 1) // 2
            second_row_count = total_images - first_row_count
            
            first_row_images = all_images[:first_row_count]
            second_row_images = all_images[first_row_count:] if second_row_count > 0 else []
            
            max_height = max(img.height for img in all_images)
            max_width = max(img.width for img in all_images)
            
            row1_width = sum(img.width for img in first_row_images)
            row2_width = sum(img.width for img in second_row_images) if second_row_images else 0
            
            if second_row_count > 0 and first_row_count > second_row_count:
                row2_width += (first_row_count - second_row_count) * max_width
            
            final_width = max(row1_width, row2_width)
            final_height = max_height * 2
            final_image = Image.new('RGB', (final_width, final_height), color='white')
            
            # Paste first row
            current_x = 0
            for img in first_row_images:
                final_image.paste(img, (current_x, 0))
                current_x += img.width
            
            # Paste second row
            if second_row_images:
                current_x = 0
                for img in second_row_images:
                    final_image.paste(img, (current_x, max_height))
                    current_x += img.width
                
                # Fill missing slots
                if first_row_count > second_row_count:
                    for _ in range(first_row_count - second_row_count):
                        black_img = Image.new('RGB', (max_width, max_height), color='black')
                        final_image.paste(black_img, (current_x, max_height))
                        current_x += max_width
            
            return final_image
        except Exception as e:
            self.logger.error(f"Stitching failed: {e}")
            return None
    
    def long_screenshot_with_photo_cycle(self, max_scrolls=20, scroll_delay=0.8, max_photos=15, scale_factor=0.5):
        try:
            if not self.running:
                return None
            
            scroll_pixels = int(self.device_height * 0.675)
            all_images = []
            
            # Initial screenshot
            initial_image = self.screenshot_to_memory()
            if initial_image is None or not self.running:
                return None
            all_images.append(initial_image)
            
            # Cycle through photos
            photo_images = self._cycle_through_user_photos(max_photos)
            if not self.running:
                return None
            all_images.extend(photo_images)
            
            # Scroll through profile
            scroll_images = []
            scroll_hashes = [self._get_image_hash(initial_image)]
            all_scroll_images = [initial_image]
            consecutive_similar = 0
            
            for i in range(1, max_scrolls + 1):
                if not self.running:
                    break
                
                if not self._controlled_scroll(scroll_pixels):
                    break
                
                if not self.wait(scroll_delay):
                    break
                
                current_image = self.screenshot_to_memory()
                if current_image is None:
                    break
                
                if self._is_duplicate_image(current_image, all_scroll_images, scroll_hashes):
                    break
                
                # Check similarity
                prev_image = scroll_images[-1] if scroll_images else initial_image
                if self._images_similar(prev_image, current_image, 0.95):
                    consecutive_similar += 1
                    if consecutive_similar >= 3:
                        break
                else:
                    consecutive_similar = 0
                
                scroll_images.append(current_image)
                all_scroll_images.append(current_image)
                scroll_hashes.append(self._get_image_hash(current_image))
            
            if not self.running:
                return None
            
            all_images.extend(scroll_images)
            
            # Stitch images
            final_image = self._stitch_in_capture_order(all_images)
            if final_image is None or not self.running:
                return None
            
            # Scale and convert
            if scale_factor != 1.0:
                new_width = int(final_image.width * scale_factor)
                new_height = int(final_image.height * scale_factor)
                final_image = final_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            if final_image.mode in ('RGBA', 'LA'):
                rgb_image = Image.new('RGB', final_image.size, (255, 255, 255))
                rgb_image.paste(final_image, mask=final_image.split()[-1] if final_image.mode == 'RGBA' else None)
                final_image = rgb_image
            
            buffer = BytesIO()
            final_image.save(buffer, format='JPEG', quality=80, optimize=True)
            image_bytes = buffer.getvalue()
            buffer.close()
            
            return image_bytes
        except Exception as e:
            self.logger.error(f"Capture failed: {e}")
            return None
    
    def _process_single_profile(self, criteria):
        try:
            if not self.running:
                return False
            
            # Get required buttons with clean UI
            required_buttons = ['like', 'dislike', 'scroll']
            detected_buttons, current_screenshot = self.detect_buttons_with_clean_ui(required_buttons)
            
            if detected_buttons is None or not self.running:
                return False
            
            # Click scroll button
            scroll_coords = detected_buttons['scroll']
            if not self.tap(scroll_coords[0], scroll_coords[1], scroll_coords[2], scroll_coords[3]):
                return False
            
            if not self.wait(self.medium_wait):
                return False
            
            # Check for overlays after scroll
            self.web_print("Checking for overlays after scroll...")
            if not self._check_and_resolve_overlays():
                if not self.running:
                    return False
                return False
            
            # Capture comprehensive screenshot
            self.web_print("Capturing comprehensive profile screenshot...")
            image_bytes = self.long_screenshot_with_photo_cycle()
            
            if image_bytes is None or not self.running:
                return False
            
            # Get final buttons
            final_required_buttons = ['like', 'dislike']
            final_buttons, final_screenshot = self.detect_buttons_with_clean_ui(final_required_buttons)
            
            if final_buttons is None or not self.running:
                return False
            
            if final_buttons.get('like') is None or final_buttons.get('dislike') is None:
                return False
            
            # Process profile with AI
            decision_data = self.process(image_bytes, criteria)
            
            if not self.running:
                return False
            
            if isinstance(decision_data, dict) and decision_data.get("action"):
                return False
            
            try:
                decision_json = json.loads(decision_data) if isinstance(decision_data, str) else decision_data
                reason = decision_json.get("reason", "No reason provided")
                action = decision_json.get("action", "[DISLIKE]")
            except (json.JSONDecodeError, AttributeError):
                return False
            
            if not self.running:
                return False
            
            # Save profile screenshot
            self._save_profile_screenshot(image_bytes, action, reason)
            
            self.web_print(f"Decision: {action}")
            self.web_print(f"Reason: {reason}")
            
            if not self.running:
                return False
            
            # Execute action
            if action == "[LIKE]":
                like_coords = final_buttons['like']
                if self.tap(like_coords[0], like_coords[1], like_coords[2], like_coords[3]):
                    self.profile_counts['liked'] += 1
                else:
                    return False
            elif action == "[DISLIKE]":
                dislike_coords = final_buttons['dislike']
                if self.tap(dislike_coords[0], dislike_coords[1], dislike_coords[2], dislike_coords[3]):
                    self.profile_counts['disliked'] += 1
                else:
                    return False
            else:
                return False
            
            self.profile_counts['total'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Single profile processing failed: {e}")
            return False
    
    def run_continuous_workflow(self):
        try:
            self.web_print("Starting Tinder automation...")
            self.web_print("Automation can be stopped via the web interface")
            
            # Launch app
            self.web_print("Launching Tinder app...")
            if not self.launch(self.package_name):
                return False
            
            if not self.wait(self.long_wait):
                self.web_print("Automation stopped during app launch")
                return True
            
            # Initial UI validation
            self.web_print("Validating UI and detecting buttons...")
            required_buttons = ['like', 'dislike', 'scroll']
            initial_buttons, initial_screenshot = self.detect_buttons_with_clean_ui(required_buttons)
            
            if initial_buttons is None or not self.running:
                if not self.running:
                    self.web_print("Automation stopped during initial setup")
                    return True
                return False
            
            self.web_print("Automation started successfully - beginning profile processing...")
            
            profile_count = 0
            while self.running:
                try:
                    profile_count += 1
                    self.web_print(f"Processing profile #{profile_count}...")
                    
                    if not self._process_single_profile(self.decision_criteria):
                        if not self.running:
                            break
                        self.web_print(f"Failed to process profile #{profile_count}")
                        if not self.wait(self.medium_wait):
                            break
                        continue
                    
                    like_rate = (self.profile_counts['liked'] / max(self.profile_counts['total'], 1)) * 100
                    self.web_print(f"Profile #{profile_count} completed | Stats: {self.profile_counts['total']} total, {self.profile_counts['liked']} liked ({like_rate:.1f}%), {self.profile_counts['disliked']} disliked")
                    
                    if self.running:
                        if not self.wait(self.medium_wait):
                            break
                    
                except Exception as e:
                    if not self.running:
                        break
                    self.logger.error(f"Error processing profile #{profile_count}: {e}")
                    if not self.wait(self.medium_wait):
                        break
                    continue
            
            self.web_print("Automation stopped")
            self.web_print("Final Statistics:")
            self.web_print(f"  Total Profiles: {self.profile_counts['total']}")
            self.web_print(f"  Liked: {self.profile_counts['liked']}")
            self.web_print(f"  Disliked: {self.profile_counts['disliked']}")
            if self.profile_counts['total'] > 0:
                like_percentage = (self.profile_counts['liked'] / self.profile_counts['total']) * 100
                self.web_print(f"  Like Rate: {like_percentage:.1f}%")
            return True
            
        except Exception as e:
            if not self.running:
                self.web_print("Automation stopped")
                return True
            self.logger.error(f"Continuous workflow failed: {e}")
            return False

class AutomationManager:
    def __init__(self):
        self.automator = None
        self.thread = None
        self.is_running = False
        self.log_queue = queue.Queue()
        
        # Initialize screen mirror
        self.screen_mirror = ScreenMirror(quality=30, scale_factor=0.4, target_fps=20)
        
    def start_automation(self):
        if self.is_running:
            return False, "Automation is already running"
            
        try:
            def run_automation():
                try:
                    self.log_queue.put(('status', 'Initializing automation...'))
                    
                    # Get settings within app context BEFORE creating thread
                    with app.app_context():
                        settings = Settings.get_settings()
                        settings_dict = settings.to_dict()
                    
                    # Create automator with settings dictionary (NO DATABASE ACCESS NEEDED)
                    self.automator = AndroidAutomator(self.log_queue, settings_dict)
                    self.is_running = True
                    
                    # Setup web console logging
                    web_handler = WebConsoleHandler(self.log_queue)
                    web_handler.setLevel(logging.INFO)
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    web_handler.setFormatter(formatter)
                    
                    self.automator.logger.addHandler(web_handler)
                    self.automator.logger.setLevel(logging.INFO)
                    
                    root_logger = logging.getLogger()
                    root_logger.addHandler(web_handler)
                    root_logger.setLevel(logging.INFO)
                    
                    # Re-suppress verbose library logging after root logger setup
                    for logger_name in ['google.genai', 'google.ai.generativelanguage', 'google.auth', 'httpx', 'httpcore', 'urllib3', 'requests', 'werkzeug', 'AFC']:
                        logging.getLogger(logger_name).setLevel(logging.ERROR)
                    
                    # Setup screen mirroring with device
                    self.screen_mirror.set_device(self.automator.device)
                    
                    self.log_queue.put(('status', 'Starting Tinder automation...'))
                    
                    # Start screen mirroring
                    if self.screen_mirror.start_capture():
                        self.log_queue.put(('status', 'Screen mirroring started'))
                        socketio.emit('mirror_status', {'status': 'started', 'message': 'Screen mirroring active'})
                    
                    # Run automation
                    success = self.automator.run_continuous_workflow()
                    
                    if success:
                        self.log_queue.put(('status', 'Automation completed successfully'))
                    else:
                        self.log_queue.put(('status', 'Automation failed'))
                        
                except Exception as e:
                    error_msg = str(e)
                    if "Can't find any android device/emulator" in error_msg:
                        self.log_queue.put(('error', 'No Android device found'))
                        self.log_queue.put(('error', 'Please ensure:'))
                        self.log_queue.put(('error', '• Android device is connected via USB'))
                        self.log_queue.put(('error', '• USB debugging is enabled on the device'))
                        self.log_queue.put(('error', '• Device drivers are properly installed'))
                        self.log_queue.put(('error', '• Device is authorized for debugging'))
                        self.log_queue.put(('error', 'You can verify the connection by running "adb devices" in command line'))
                    else:
                        self.log_queue.put(('error', f'Automation error: {error_msg}'))
                        import traceback
                        self.log_queue.put(('error', f'Traceback: {traceback.format_exc()}'))
                finally:
                    # Stop screen mirroring
                    self.screen_mirror.stop_capture()
                    socketio.emit('mirror_status', {'status': 'stopped', 'message': 'Screen mirroring stopped'})
                    
                    self.is_running = False
                    self.log_queue.put(('status', 'Automation stopped'))
                    
            self.thread = threading.Thread(target=run_automation, daemon=True)
            self.thread.start()
            
            return True, "Automation started successfully"
            
        except Exception as e:
            self.is_running = False
            return False, f"Failed to start automation: {str(e)}"
    
    def stop_automation(self):
        if not self.is_running:
            return False, "Automation is not running"
            
        try:
            if self.automator:
                self.automator.running = False
                self.log_queue.put(('status', 'Stop signal sent, waiting for graceful shutdown...'))
            
            # Stop screen mirroring
            self.screen_mirror.stop_capture()
            socketio.emit('mirror_status', {'status': 'stopped', 'message': 'Screen mirroring stopped'})
                
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=10)
                
            self.is_running = False
            return True, "Automation stopped successfully"
            
        except Exception as e:
            self.is_running = False
            return False, f"Error stopping automation: {str(e)}"
    
    def get_status(self):
        stats = {'total': 0, 'liked': 0, 'disliked': 0}
        if self.automator and hasattr(self.automator, 'profile_counts'):
            stats = self.automator.profile_counts.copy()
            
        return {
            'running': self.is_running,
            'stats': stats
        }
    
    def reload_automator_settings(self):
        """Reload settings in running automator if it exists"""
        if self.automator and self.is_running:
            try:
                with app.app_context():
                    settings = Settings.get_settings()
                    settings_dict = settings.to_dict()
                self.automator.reload_settings(settings_dict)
                self.log_queue.put(('status', 'Settings reloaded in running automation'))
                return True
            except Exception as e:
                self.log_queue.put(('error', f'Failed to reload settings: {str(e)}'))
                return False
        return True

automation_manager = AutomationManager()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_automation():
    success, message = automation_manager.start_automation()
    return jsonify({'success': success, 'message': message})

@app.route('/stop', methods=['POST'])
def stop_automation():
    success, message = automation_manager.stop_automation()
    return jsonify({'success': success, 'message': message})

@app.route('/status')
def get_status():
    return jsonify(automation_manager.get_status())

# Settings API routes
@app.route('/api/settings', methods=['GET'])
def get_settings():
    try:
        settings = Settings.get_settings()
        return jsonify({
            'success': True,
            'settings': settings.to_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to get settings: {str(e)}'
        }), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        settings = Settings.get_settings()
        settings.update_from_dict(data)
        db.session.commit()
        
        # Reload settings in running automator if exists
        automation_manager.reload_automator_settings()
        
        return jsonify({
            'success': True,
            'message': 'Settings updated successfully',
            'settings': settings.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Failed to update settings: {str(e)}'
        }), 500

@app.route('/logs')
def stream_logs():
    def generate():
        while True:
            try:
                try:
                    log_type, message = automation_manager.log_queue.get(timeout=1)
                    
                    # Handle profile log entries within app context
                    if log_type == 'profile_log':
                        try:
                            with app.app_context():
                                profile_log = ProfileLog(
                                    action=message['action'],
                                    reason=message['reason'],
                                    screenshot_path=message['screenshot_path']
                                )
                                db.session.add(profile_log)
                                db.session.commit()
                        except Exception as e:
                            print(f"Failed to save profile log: {e}")
                        continue
                    
                    timestamp = time.strftime('%H:%M:%S')
                    
                    data = {
                        'type': log_type,
                        'message': message,
                        'timestamp': timestamp
                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                    
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    
            except GeneratorExit:
                break
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
    })

# SocketIO handlers for screen mirroring
@socketio.on('connect')
def handle_connect():
    automation_manager.screen_mirror.clients.add(request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    automation_manager.screen_mirror.clients.discard(request.sid)

# Initialize database
def init_db():
    with app.app_context():
        db.create_all()
        # Create default settings if none exist
        Settings.get_settings()

if __name__ == '__main__':
    print("Starting Tinder Automation Web Interface...")
    print("Initializing database...")
    
    init_db()
    
    print("Open http://localhost:5000 in your browser")
    
    # Suppress SocketIO startup messages
    logging.getLogger('socketio.server').setLevel(logging.WARNING)
    logging.getLogger('engineio.server').setLevel(logging.WARNING)
    
    try:
        socketio.run(app, debug=False, host='0.0.0.0', port=5000, use_reloader=False, log_output=False)
    except KeyboardInterrupt:
        automation_manager.screen_mirror.stop_capture()
    except Exception as e:
        automation_manager.screen_mirror.stop_capture()