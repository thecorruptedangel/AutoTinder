import uiautomator2 as u2
import time
import logging
import math
import random
import json
import os
import signal
from datetime import datetime
from PIL import Image
import hashlib
import numpy as np
from io import BytesIO
import cv2
from google import genai
from google.genai import types
from requests.exceptions import RequestException, Timeout, ConnectionError
import config

class AndroidAutomator:
    def __init__(self, device_id=None):
        # Validate that all required config variables are defined
        self._validate_config()
        
        # Load configuration from config.py
        self.api_key = config.GEMINI_API_KEY
        self.decision_prompt = config.DECISION_PROMPT
        self.ui_resolver_prompt = config.UI_RESOLVER_PROMPT
        
        # Load model names
        self.model_heavy = config.MODEL_HEAVY
        self.model_fast = config.MODEL_FAST
        
        # Load wait durations
        self.short_wait = config.SHORT_WAIT
        self.medium_wait = config.MEDIUM_WAIT
        self.long_wait = config.LONG_WAIT
        
        # Load app configuration
        self.package_name = config.PACKAGE_NAME
        self.decision_criteria = config.DECISION_CRITERIA
        
        self.device = u2.connect(device_id) if device_id else u2.connect()
        self.running = True
        
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Minimal logging setup
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        # Suppress verbose library logging
        logging.getLogger('google.genai').setLevel(logging.WARNING)
        logging.getLogger('google.ai.generativelanguage').setLevel(logging.WARNING)
        logging.getLogger('google.auth').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
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
            self.logger.warning("Failed to get screenshot dimensions, using fallback 1080x2400")
        
        self.scale_x = self.device_width / self.screenshot_width
        self.scale_y = self.device_height / self.screenshot_height
        
        self.button_config = {
            'like': {'template': 'like.png', 'scales': [0.3, 0.5, 0.7, 0.9, 1.1], 'threshold': 0.65, 'ratio': (0.7, 1.4)},
            'dislike': {'template': 'dislike.png', 'scales': [0.3, 0.5, 0.7, 0.9, 1.1], 'threshold': 0.65, 'ratio': (0.7, 1.4)},
            'scroll': {'template': 'scroll.png', 'scales': [0.4, 0.6, 0.8, 1.0, 1.2], 'threshold': 0.7, 'ratio': (0.5, 2.0)}
        }
        
        self.genai_client = genai.Client(
            api_key=self.api_key,
        )
        
        self.analyzed_profiles_dir = "analyzed_profiles"
        self._create_profiles_directory()
        
        self.profile_counts = {
            'total': 0,
            'liked': 0,
            'disliked': 0
        }
        
        print(f"Device: {self.device_info['productName']} ({self.device_width}x{self.device_height})")
        print("Automation initialized successfully")
    
    def _validate_config(self):
        """Validate that all required configuration variables are defined in config.py"""
        required_vars = [
            'GEMINI_API_KEY',
            'DECISION_PROMPT', 
            'UI_RESOLVER_PROMPT',
            'MODEL_HEAVY',
            'MODEL_FAST',
            'SHORT_WAIT',
            'MEDIUM_WAIT',
            'LONG_WAIT',
            'PACKAGE_NAME',
            'DECISION_CRITERIA'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not hasattr(config, var):
                missing_vars.append(var)
            elif getattr(config, var) is None:
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required configuration variables in config.py: {', '.join(missing_vars)}")
        
        # Validate API key is not empty
        if not config.GEMINI_API_KEY or config.GEMINI_API_KEY.strip() == "":
            raise ValueError("GEMINI_API_KEY cannot be empty")
        
        # Validate wait durations are numeric
        try:
            float(config.SHORT_WAIT)
            float(config.MEDIUM_WAIT) 
            float(config.LONG_WAIT)
        except (ValueError, TypeError):
            raise ValueError("Wait duration variables (SHORT_WAIT, MEDIUM_WAIT, LONG_WAIT) must be numeric")
    
    def signal_handler(self, signum, frame):
        print("Stopping automation gracefully...")
        self.running = False
    
    def _create_profiles_directory(self):
        try:
            if not os.path.exists(self.analyzed_profiles_dir):
                os.makedirs(self.analyzed_profiles_dir)
        except Exception as e:
            self.logger.error(f"Failed to create profiles directory: {e}")
    
    def _save_profile_screenshot(self, image_bytes, action):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            action_prefix = "liked" if action == "[LIKE]" else "disliked"
            filename = f"{action_prefix}_profile_{timestamp}.jpg"
            filepath = os.path.join(self.analyzed_profiles_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save profile screenshot: {e}")
            return None
    
    def pil_to_cv2(self, pil_image):
        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return cv2_image
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
                    self.logger.warning(f"Template {cfg['template']} not found")
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
    
    def _make_api_request(self, image_bytes, text_prompt=None, system_prompt="", temperature=0.5, model=None, max_retries=3, **config_kwargs):
        if model is None:
            model = self.model_heavy
            
        last_error = None
        
        for attempt in range(max_retries):
            if not self.running:  # Check if we should stop
                return {"reason": "Interrupted", "action": "[ERROR]"}
                
            try:
                if attempt > 0:
                    # Interrupt-aware sleep for retry backoff
                    backoff_time = 2 ** attempt
                    end_time = time.time() + backoff_time
                    while time.time() < end_time and self.running:
                        time.sleep(0.1)
                    if not self.running:
                        return {"reason": "Interrupted", "action": "[ERROR]"}
                
                parts = [
                    types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=image_bytes,
                    ),
                ]
                
                if text_prompt:
                    parts.append(types.Part.from_text(text=text_prompt))
                
                contents = [
                    types.Content(
                        role="user",
                        parts=parts,
                    ),
                ]
                
                config_params = {
                    "temperature": temperature,
                    "system_instruction": [
                        types.Part.from_text(text=system_prompt),
                    ],
                }
                config_params.update(config_kwargs)
                
                generate_content_config = types.GenerateContentConfig(**config_params)
                
                response = self.genai_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                if not self.running:
                    return {"reason": "Interrupted", "action": "[ERROR]"}
                
                if not response:  
                    self.logger.error("No response from AI")
                    return {"reason": "No response from AI", "action": "[ERROR]"}
                
                if not hasattr(response, 'text') or not response.text:
                    self.logger.error("Empty response text from AI")
                    return {"reason": "Empty response from AI", "action": "[ERROR]"}
                
                response_text = response.text.strip()
                
                if response_text.startswith("```json"):
                    response_text = response_text[7:]  
                
                if response_text.endswith("```"):
                    response_text = response_text[:-3] 
                
                response_text = response_text.strip()
                
                return response_text
                
            except Exception as e:
                last_error = e
                if not self.running:
                    return {"reason": "Interrupted", "action": "[ERROR]"}
                    
                if attempt == max_retries - 1:
                    self.logger.error(f"API request failed after {max_retries} attempts: {e}")
                
                if not self._is_retryable_error(e):
                    error_code = self._map_error_to_action_code(e)
                    return {"reason": f"API Error: {str(e)}", "action": error_code}
                
                if attempt == max_retries - 1:
                    break
        
        error_code = self._map_error_to_action_code(last_error)
        return {"reason": f"API Error after {max_retries} attempts: {str(last_error)}", "action": error_code}
    
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
            
            coordinates = [abs_x1, abs_y1, abs_x2, abs_y2]
            return coordinates
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse UI resolution response: {e}")
            return []
        except Exception as e:
            self.logger.error(f"UI resolution failed: {e}")
            return []

    def _check_and_resolve_overlays(self, max_attempts=5):
        """
        Proactively check for overlays and resolve them until UI is clean.
        Returns True if UI is clean, False if failed to resolve or interrupted.
        """
        for attempt in range(max_attempts):
            if not self.running:
                return False
                
            current_screenshot = self.screenshot_to_memory()
            if current_screenshot is None:
                continue
            
            # Check for overlays using resolve_ui
            ui_coordinates = self.resolve_ui(current_screenshot)
            
            # If no overlay coordinates returned, UI is clean
            if not ui_coordinates or len(ui_coordinates) != 4:
                return True
            
            # Overlay detected, attempt to dismiss it
            print(f"⚠ Overlay detected, attempting to resolve (attempt {attempt + 1}/{max_attempts})")
            x1, y1, x2, y2 = ui_coordinates
            
            if not self.tap(x1, y1, x2, y2):
                self.logger.warning(f"Failed to tap overlay dismissal coordinates")
                if not self.wait(self.short_wait):
                    return False
                continue
            
            # Wait for UI to update after dismissal
            if not self.wait(self.medium_wait):
                return False
            
            current_screenshot = None
        
        self.logger.error(f"Failed to resolve overlays after {max_attempts} attempts")
        return False

    def detect_buttons_with_clean_ui(self, required_buttons, max_attempts=3):
        """
        Detect buttons only after ensuring UI is clean of overlays.
        This method handles overlay checking internally - no need for external calls.
        """
        # First, ensure UI is clean of overlays (ONLY ONCE)
        if not self._check_and_resolve_overlays():
            if not self.running:
                return None, None
            self.logger.error("Failed to resolve overlays before button detection")
            return None, None
        
        # Now retry button detection with clean UI (no more overlay checks)
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
            
            # If buttons still missing, wait and try again (but don't re-check overlays)
            print(f"Missing buttons: {missing_buttons}, retrying... (attempt {attempt + 1}/{max_attempts})")
            if not self.wait(self.medium_wait):
                return None, None
            
            current_screenshot = None
        
        self.logger.error(f"Failed to find required buttons after {max_attempts} attempts with clean UI")
        return None, None
    
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
    
    def launch(self, package_name):
        try:
            self.device.app_start(package_name)
            return True
        except Exception as e:
            self.logger.error(f"Launch failed: {e}")
            return False
    
    def wait(self, seconds):
        """Wait with interrupt checking"""
        end_time = time.time() + seconds
        while time.time() < end_time and self.running:
            time.sleep(0.1)  # Check every 100ms
        return self.running
    
    def screenshot(self, filename):
        try:
            self.device.screenshot(filename)
            return True
        except Exception as e:
            self.logger.error(f"Screenshot failed: {e}")
            return False
    
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
    
    def _get_image_hash(self, image):
        try:
            img = image.convert('L').resize((100, 100))
            return hashlib.md5(img.tobytes()).hexdigest()
        except:
            return None
    
    def _images_similar(self, img1, img2, threshold=0.95):
        try:
            size = (300, 300)
            img1_resized = img1.resize(size).convert('L')
            img2_resized = img2.resize(size).convert('L')
            
            arr1 = np.array(img1_resized)
            arr2 = np.array(img2_resized)
            
            correlation = np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 1.0
            
            return correlation >= threshold
        except:
            return False
    
    def _is_duplicate_image(self, new_image, existing_images, existing_hashes, strict_threshold=0.98):
        try:
            new_hash = self._get_image_hash(new_image)
            
            if new_hash in existing_hashes:
                return True
            
            recent_images = existing_images[-3:] if len(existing_images) > 3 else existing_images
            
            for existing_image in recent_images:
                if self._images_similar(new_image, existing_image, strict_threshold):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Duplicate check failed: {e}")
            return False
    
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
    
    def _calculate_photo_cycle_tap_location(self):
        segment_height = self.device_height // 4
        tap_y = segment_height
        tap_x = int(self.device_width * 0.8)
        return tap_x, tap_y
    
    def _cycle_through_user_photos(self, tap_delay=1.0, max_photos=20):
        """
        Cycle through user photos - NO overlay checking during this operation
        as it's a normal UI state, not an obstruction.
        """
        try:
            tap_x, tap_y = self._calculate_photo_cycle_tap_location()
            photo_images = []
            photo_hashes = []
            
            for i in range(max_photos):
                if not self.running:  # Check if we should stop
                    break
                    
                if not self.tap_coordinates(tap_x, tap_y):
                    break
                
                if not self.wait(self.medium_wait):  # Wait returns False if interrupted
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
            total_images = len(all_images)
            if total_images == 0:
                return None
            
            first_row_count = (total_images + 1) // 2
            second_row_count = total_images - first_row_count
            
            first_row_images = all_images[:first_row_count]
            second_row_images = all_images[first_row_count:] if second_row_count > 0 else []
            
            max_height = max(img.height for img in all_images)
            max_width = max(img.width for img in all_images)
            
            row1_width = sum(img.width for img in first_row_images)
            row2_width = sum(img.width for img in second_row_images) if second_row_images else 0
            
            if second_row_count > 0 and first_row_count > second_row_count:
                missing_images = first_row_count - second_row_count
                row2_width += missing_images * max_width
            
            final_width = max(row1_width, row2_width)
            final_height = max_height * 2
            
            final_image = Image.new('RGB', (final_width, final_height), color='white')
            
            current_x = 0
            for img in first_row_images:
                final_image.paste(img, (current_x, 0))
                current_x += img.width
            
            if second_row_images:
                current_x = 0
                for img in second_row_images:
                    final_image.paste(img, (current_x, max_height))
                    current_x += img.width
                
                if first_row_count > second_row_count:
                    missing_count = first_row_count - second_row_count
                    for i in range(missing_count):
                        black_img = Image.new('RGB', (max_width, max_height), color='black')
                        final_image.paste(black_img, (current_x, max_height))
                        current_x += max_width
            
            return final_image
        except Exception as e:
            self.logger.error(f"Stitching failed: {e}")
            return None
    
    def _scale_image(self, image, scale_factor):
        try:
            original_width, original_height = image.size
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return scaled_image
        except:
            return image
    
    def long_screenshot_with_photo_cycle(self, max_scrolls=50, scroll_delay=1.0, 
                                       tap_delay=1.0, similarity_threshold=0.95, scroll_pixels=None, 
                                       max_photos=20, scale_factor=0.5):
        """
        Capture comprehensive profile screenshot - NO overlay checking during scrolling/cycling
        as these are normal UI operations, not obstructions.
        """
        try:
            if not self.running:
                return None
                
            if scroll_pixels is None:
                scroll_pixels = int(self.device_height * 0.675)
            
            all_images = []
            
            initial_image = self.screenshot_to_memory()
            if initial_image is None or not self.running:
                return None
            all_images.append(initial_image)
            
            # Cycle through photos - no overlay checking during this normal operation
            photo_images = self._cycle_through_user_photos(tap_delay, max_photos)
            if not self.running:
                return None
            all_images.extend(photo_images)
            
            # Scroll through profile - no overlay checking during this normal operation  
            scroll_images = []
            scroll_hashes = []
            all_scroll_images = [initial_image]
            consecutive_similar = 0
            
            initial_hash = self._get_image_hash(initial_image)
            scroll_hashes.append(initial_hash)
            
            for i in range(1, max_scrolls + 1):
                if not self.running:  # Check if we should stop
                    break
                    
                if not self._controlled_scroll(scroll_pixels):
                    break
                
                # Use our interrupt-aware wait
                if not self.wait(scroll_delay):
                    break
                
                current_image = self.screenshot_to_memory()
                if current_image is None:
                    break
                
                if self._is_duplicate_image(current_image, all_scroll_images, scroll_hashes):
                    break
                
                if len(scroll_images) > 0:
                    if self._images_similar(scroll_images[-1], current_image, similarity_threshold):
                        consecutive_similar += 1
                        if consecutive_similar >= 3:
                            break
                    else:
                        consecutive_similar = 0
                elif len(scroll_images) == 0:
                    if self._images_similar(initial_image, current_image, similarity_threshold):
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
            
            if len(all_images) == 0:
                return None
            
            final_image = self._stitch_in_capture_order(all_images)
            if final_image is None or not self.running:
                return None
            
            if scale_factor != 1.0:
                final_image = self._scale_image(final_image, scale_factor)
            
            if final_image.mode in ('RGBA', 'LA'):
                rgb_image = Image.new('RGB', final_image.size, (255, 255, 255))
                rgb_image.paste(final_image, mask=final_image.split()[-1] if final_image.mode == 'RGBA' else None)
                final_image = rgb_image
            
            buffer = BytesIO()
            final_image.save(buffer, format='JPEG', quality=80, optimize=True)
            image_bytes = buffer.getvalue()
            buffer.close()
            
            all_images.clear()
            
            return image_bytes
            
        except Exception as e:
            self.logger.error(f"Capture failed: {e}")
            return None
    
    def _map_error_to_action_code(self, error):
        error_str = str(error).lower()
        
        if "400" in error_str:
            if "invalid_argument" in error_str:
                return "[INVALID_ARGUMENT]"
            elif "failed_precondition" in error_str:
                return "[FAILED_PRECONDITION]"
            else:
                return "[INVALID_ARGUMENT]"
        elif "403" in error_str:
            return "[PERMISSION_DENIED]"
        elif "404" in error_str:
            return "[NOT_FOUND]"
        elif "429" in error_str:
            return "[RESOURCE_EXHAUSTED]"
        elif "500" in error_str:
            return "[INTERNAL]"
        elif "503" in error_str:
            return "[UNAVAILABLE]"
        elif "504" in error_str:
            return "[DEADLINE_EXCEEDED]"
        
        if "deadline" in error_str or "timeout" in error_str:
            return "[DEADLINE_EXCEEDED]"
        elif "rate limit" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
            return "[RESOURCE_EXHAUSTED]"
        elif "permission" in error_str or "access" in error_str:
            return "[PERMISSION_DENIED]"
        elif "not found" in error_str:
            return "[NOT_FOUND]"
        elif "invalid" in error_str or "malformed" in error_str:
            return "[INVALID_ARGUMENT]"
        elif "unavailable" in error_str or "overloaded" in error_str:
            return "[UNAVAILABLE]"
        elif "internal" in error_str or "server" in error_str:
            return "[INTERNAL]"
        
        return "[ERROR]"
    
    def _is_retryable_error(self, error):
        error_str = str(error).lower()
        
        retryable_patterns = [
            "timeout",
            "connection",
            "network",
            "unavailable",
            "503",
            "500",
            "deadline_exceeded",
            "504"
        ]
        
        if isinstance(error, (ConnectionError, Timeout, RequestException)):
            return True
        
        for pattern in retryable_patterns:
            if pattern in error_str:
                return True
        
        return False
    
    def process(self, image_bytes, criteria, max_retries=3):
        try:
            print("Analyzing profile...")    
            start_time = time.time()
            
            text_prompt = f"CRITERIA: {criteria}"
            response_text = self._make_api_request(
                image_bytes=image_bytes,
                text_prompt=text_prompt,
                system_prompt=self.decision_prompt,
                temperature=0.5,
                model=self.model_heavy,
                max_retries=max_retries
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print(f"Analysis completed in {elapsed_time:.2f} seconds")
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Process function failed: {e}")
            error_code = self._map_error_to_action_code(e)
            return {"reason": f"Process Error: {str(e)}", "action": error_code}
    
    def _process_single_profile(self, criteria):
        try:
            if not self.running:
                return False
                
            # STEP 1: Get required buttons with clean UI (overlay check happens inside detect_buttons_with_clean_ui)
            required_buttons = ['like', 'dislike', 'scroll']
            detected_buttons, current_screenshot = self.detect_buttons_with_clean_ui(required_buttons)
            
            if detected_buttons is None or not self.running:
                return False
            
            # STEP 2: Click scroll button to start profile exploration
            scroll_coords = detected_buttons['scroll']
            if not self.tap(scroll_coords[0], scroll_coords[1], scroll_coords[2], scroll_coords[3]):
                return False
            
            if not self.wait(self.medium_wait):
                return False
            
            # STEP 2.5: Check for overlays that might appear after clicking scroll button
            print("Checking for overlays after scroll button click...")
            if not self._check_and_resolve_overlays():
                if not self.running:
                    return False
                self.logger.error("Failed to resolve overlays that appeared after clicking scroll button")
                return False
            
            # STEP 3: Capture comprehensive screenshot (no overlay checking during this normal operation)
            print("Capturing comprehensive profile screenshot...")
            image_bytes = self.long_screenshot_with_photo_cycle(
                max_scrolls=20, 
                scroll_delay=0.8,
                tap_delay=1.0,
                max_photos=15,
                scale_factor=0.5
            )
            
            if image_bytes is None or not self.running:
                return False
            
            # STEP 4: Get final buttons with clean UI (overlay check happens inside detect_buttons_with_clean_ui)
            final_required_buttons = ['like', 'dislike']
            final_buttons, final_screenshot = self.detect_buttons_with_clean_ui(final_required_buttons)
            
            if final_buttons is None or not self.running:
                return False
                
            if final_buttons.get('like') is None or final_buttons.get('dislike') is None:
                self.logger.error("Like or Dislike button not found in final screenshot")
                return False
            
            # STEP 5: Process profile with AI
            decision_data = self.process(image_bytes, criteria)
            
            if not self.running:
                return False
            
            if isinstance(decision_data, dict) and decision_data.get("action"):
                self.logger.error(f"AI processing failed: {decision_data.get('reason')}")
                return False
            
            try:
                if isinstance(decision_data, str):
                    decision_json = json.loads(decision_data)
                else:
                    decision_json = decision_data
                
                reason = decision_json.get("reason", "No reason provided")
                action = decision_json.get("action", "[DISLIKE]")
                
            except (json.JSONDecodeError, AttributeError) as e:
                self.logger.error(f"Failed to parse decision JSON: {e}")
                return False
            
            if not self.running:
                return False
            
            # STEP 6: Save profile screenshot
            saved_filepath = self._save_profile_screenshot(image_bytes, action)
            if saved_filepath is None:
                self.logger.error("Failed to save profile screenshot")
                return False
            
            print(f"Decision: {action}")
            print(f"Reason: {reason}")
            
            if not self.running:
                return False
            
            # STEP 7: Execute action
            if action == "[LIKE]":
                like_coords = final_buttons['like']
                if self.tap(like_coords[0], like_coords[1], like_coords[2], like_coords[3]):
                    self.profile_counts['liked'] += 1
                else:
                    self.logger.error("Failed to tap LIKE button")
                    return False
                    
            elif action == "[DISLIKE]":
                dislike_coords = final_buttons['dislike']
                if self.tap(dislike_coords[0], dislike_coords[1], dislike_coords[2], dislike_coords[3]):
                    self.profile_counts['disliked'] += 1
                else:
                    self.logger.error("Failed to tap DISLIKE button")
                    return False
            else:
                self.logger.error(f"Unknown action: {action}")
                return False
            
            self.profile_counts['total'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Single profile processing failed: {e}")
            return False
    
    def run_continuous_workflow(self):
        try:
            print("Starting Tinder automation...")
            print("Press Ctrl+C to stop")
            
            # Launch app
            if not self.launch(self.package_name):
                self.logger.error("Failed to launch app")
                return False
            
            if not self.wait(self.long_wait):
                print("Interrupted during app launch")
                return True
            
            # Initial UI validation with clean UI (overlay check happens inside detect_buttons_with_clean_ui)
            required_buttons = ['like', 'dislike', 'scroll']
            initial_buttons, initial_screenshot = self.detect_buttons_with_clean_ui(required_buttons)
            
            if initial_buttons is None or not self.running:
                if not self.running:
                    print("Interrupted during initial setup")
                    return True
                self.logger.error("Failed to find required buttons during initial setup")
                return False
            
            print("Automation started successfully")
            
            profile_count = 0
            while self.running:
                try:
                    profile_count += 1
                    print(f"\nProcessing profile #{profile_count}...")
                    
                    if not self._process_single_profile(self.decision_criteria):
                        if not self.running:
                            break
                        print(f"Failed to process profile #{profile_count}")
                        if not self.wait(self.medium_wait):
                            break
                        continue
                    
                    like_rate = (self.profile_counts['liked'] / max(self.profile_counts['total'], 1)) * 100
                    print(f"Profile #{profile_count} completed | Stats: {self.profile_counts['total']} total, {self.profile_counts['liked']} liked ({like_rate:.1f}%), {self.profile_counts['disliked']} disliked")
                    
                    if self.running:
                        if not self.wait(self.medium_wait):
                            break
                    
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    if not self.running:
                        break
                    self.logger.error(f"Error processing profile #{profile_count}: {e}")
                    if not self.wait(self.medium_wait):
                        break
                    continue
            
            print("\nAutomation stopped")
            print("Final Statistics:")
            print(f"  Total Profiles: {self.profile_counts['total']}")
            print(f"  Liked: {self.profile_counts['liked']}")
            print(f"  Disliked: {self.profile_counts['disliked']}")
            if self.profile_counts['total'] > 0:
                like_percentage = (self.profile_counts['liked'] / self.profile_counts['total']) * 100
                print(f"  Like Rate: {like_percentage:.1f}%")
            return True
            
        except KeyboardInterrupt:
            self.running = False
            print("\nAutomation interrupted")
            return True
        except Exception as e:
            self.logger.error(f"Continuous workflow failed: {e}")
            return False


if __name__ == "__main__":
    automator = AndroidAutomator()
    
    success = automator.run_continuous_workflow()
    if success:
        print("Automation completed successfully!")
    else:
        print("Automation failed - check logs for details")