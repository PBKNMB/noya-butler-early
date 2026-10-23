import psutil
import subprocess
import requests
import json
import threading
import re
import time
import queue
import logging
import numpy as np
import speech_recognition as sr
import pyttsx3
import pygetwindow as gw
# 依赖安装 (pip install ...):
# psutil requests pyttsx3 speechrecognition numpy sounddevice
# chromadb win10toast pynput faster-whisper ics pygetwindow
from datetime import datetime, timedelta
import uuid
from win10toast import ToastNotifier
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener
import chromadb
from chromadb.utils import embedding_functions
from faster_whisper import WhisperModel
from ics import Calendar, Event

logging.basicConfig(
    filename='butler.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"
CHROMA_PATH = "./chroma_data"

BATTERY_LOW_THRESHOLD = 20
CPU_CHECK_INTERVAL = 120
CPU_SAMPLE_DURATION = 60
CPU_HIGH_THRESHOLD = 75
IDLE_MINUTES = 10
MONITOR_INTERVAL = 30
TOAST_DURATION = 5
MAX_HISTORY = 20
chromadb_lock = threading.Lock()
PERSONA = "你是一个温柔可靠的女仆，说话时使用“您”等敬语，语气轻柔，会主动关心用户，偶尔在句尾加上“呢~”或“哦”。"
persona_lock = threading.Lock()

PERSONA_PRESETS = {
    "温柔": "你是一个温柔可靠的女仆，说话时使用“您”等敬语，语气轻柔，会主动关心用户。",
    "冷酷": "你是一个高效简洁的助手，说话简短直接，不主动提问。",
    "活泼": "你是一个活泼开朗的伙伴，喜欢用颜文字，称呼用户为主上。(=^▽^=)",
}

def switch_persona(name: str) -> str:
    global PERSONA
    new_text = PERSONA_PRESETS.get(name, PERSONA)
    with persona_lock:
        PERSONA = new_text
    logging.info(f"性格已切换为: {name}")
    return f"性格已切换为: {name}"

class HabitRecorder:
    def __init__(self, log_file="habits.json"):
        self.log_file = log_file
        self.lock = threading.Lock()
        self._load()
    def _load(self):
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {"events": []}
    def _save(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def record_event(self, event_type, detail=""):
        with self.lock:
            self.data["events"].append({
                "timestamp": time.time(),
                "type": event_type,
                "detail": detail
            })
            self._save()
            if len(self.data["events"]) > 1000:
                self.data["events"] = self.data["events"][-1000:]
    def analyze_patterns(self):
        with self.lock:
            events = list(self.data["events"])
        if len(events) < 5:
            return []
        hour_counts = {}
        for e in events:
            hour = datetime.fromtimestamp(e["timestamp"]).hour
            key = (hour, e["type"], e["detail"])
            hour_counts[key] = hour_counts.get(key, 0) + 1
        patterns = []
        for (hour, typ, detail), cnt in hour_counts.items():
            if cnt >= 3:
                patterns.append({
                    "hour": hour,
                    "type": typ,
                    "detail": detail,
                    "confidence": cnt / len(events)
                })
        return patterns

habit_recorder = HabitRecorder()
reminded_today = set()
reminded_lock = threading.Lock()
last_clean_date = None
last_morning_greeting = None
last_evening_greeting = None

def _generate_and_notify_llm_suggestion():
    suggestion = generate_llm_suggestions()
    if suggestion:
        auto_queue.put(f"{suggestion}")
        notifier.show_toast("每日建议", suggestion, duration=10)

def habit_analysis_loop():
    global last_clean_date, reminded_today, last_morning_greeting, last_evening_greeting

    while True:
        time.sleep(3600)
        current_hour = datetime.now().hour
        today = datetime.now().strftime("%Y-%m-%d")
        if current_hour == 8 and last_morning_greeting != today:
            weather = get_weather()
            msg = f"早安！今天天气: {weather}。祝您一天愉快"
            auto_queue.put(msg)
            notifier.show_toast("早安", msg, duration=8)
            last_morning_greeting = today
        if current_hour == 21 and last_evening_greeting != today:
            msg = f"晚安,请勿操劳,明天见。"
            auto_queue.put(msg)
            notifier.show_toast("晚安", msg, duration=8)
            last_evening_greeting = today
        if last_clean_date != today:
            with reminded_lock:
                reminded_today.clear()
                last_clean_date = today
            threading.Thread(target=_generate_and_notify_llm_suggestion, daemon=True).start()

        patterns = habit_recorder.analyze_patterns()
        for p in patterns:
            if p["hour"] != current_hour:
                continue
            if p["type"] == "launch_app":
                msg = f"习惯建议: 你经常在 {current_hour}:00 左右打开 {p['detail']},需要我帮忙吗？"
            elif p["type"] == "user_query":
                msg = f"你常在此时询问「{p['detail']}」,需要我帮你记下来吗？"
            elif p["type"] == "set_reminder":
                msg = f"你通常这时会设置提醒，比如「{p['detail']}」，需要我帮你记下吗？"
            else:
                msg = f"习惯提醒：你经常在这个时间{p['detail']}。"
        
            pattern_key = (current_hour, p["type"], p["detail"])
            with reminded_lock:
                if pattern_key in reminded_today:
                    continue
                reminded_today.add(pattern_key)
            auto_queue.put(msg)
            notifier.show_toast("习惯建议", msg, duration=8)
        

def generate_llm_suggestions():
    try:
        with habit_recorder.lock:
            events = habit_recorder.data.get("events", [])
            if len(events) < 5:
                return ""
            recent = events[-50:]
        prompt = f"""根据用户最近行为记录,请给出1-2条主动建议.
行为记录:
{json.dumps(recent, ensure_ascii=False, indent=2)}
直接输出建议,每条一行,不要其他内容."""
        resp = call_ollama([{"role": "user", "content": prompt}])
        return resp.get("message", {}).get("content", "")
    except Exception as e:
        logging.error(f"LLM 生成建议失败:{e}")
        return ""
            
class Notifier:
    def __init__(self):
        self.toaster = ToastNotifier()
    def show_toast(self, title: str, message: str, duration: int = TOAST_DURATION):
        try:
            self.toaster.show_toast(title, message, duration=duration, threaded=True)
        except Exception as e:
            logging.error(f"通知失败: {e}")
notifier = Notifier()
auto_queue = queue.Queue()

def listen_for_command(timeout=5, phrase_limit=8):
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0
    try:
        with sr.Microphone(sample_rate=16000) as source:
            print("正在适应环境噪音")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("please speaking")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        text = recognizer.recognize_google(audio, language='zh-CN')
        logging.info(f"识别结果: {text}")
        return text
    except sr.WaitTimeoutError:
        print("未检测到语音")
        return ""
    except sr.UnknownValueError:
        print("无法识别语音")
        return ""
    except sr.RequestError as e:
        logging.error(f"语音服务出错: {e}")
        print(f"语音服务出错: {e}")
        return ""
    except OSError as e:
        logging.critical(f"麦克风不可用: {e}")
        print(f"麦克风访问失败: {e}")
        return ""

class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 0.9)
        voices = self.engine.getProperty('voices')
        for v in voices:
            if 'female' in v.name.lower():
                self.engine.setProperty('voice', v.id)
                break
        self._queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
    def _worker(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logging.exception("语言播报失败")
    
    def speak(self, text):
        self._queue.put(text)
    def stop(self):
        self._queue.put(None)
        self._worker_thread.join(timeout=2)

speaker = Speaker()
whisper_model = None
whisper_lock = threading.Lock()
whisper_loaded = False

def _load_whisper_background():
    global whisper_model, whisper_loaded
    try:
        print("正在后台加载 Whisper")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        with whisper_lock:
            whisper_model = model
            whisper_loaded = True
        print("Whisper 就绪")
    except Exception as e:
        logging.error(f"Whisper 加载失败: {e}")
        print("语音识别不可用")
def listen_whisper(timeout=5, phrase_limit=8):
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0
    try:
        with sr.Microphone(sample_rate=16000) as source:
            print("正在适应环境噪音")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("请说话...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        with whisper_lock:
            if not whisper_loaded or whisper_model is None:
                print("语言模型尚未就绪，请稍后重试或文字输入。")
                return ""
        audio_np = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
        model = whisper_model
        segments, _ = model.transcribe(audio_np, language="zh", beam_size=5)
        text = " ".join([seg.text.strip() for seg in segments])
        if text:
            logging.info(f"Whisper 识别: {text}")
            print(f"识别结果: {text}")
        return text
    except sr.WaitTimeoutError:
        print("未检测到语音")
        return ""
    except Exception as e:
        logging.error(f"语言识别失败: {e}")
        return ""

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)
def get_memory_usage():
    return psutil.virtual_memory().percent
def open_notepad():
    subprocess.Popen(["notepad.exe"])
    habit_recorder.record_event("launch_app", "记事本")
    return "记事本已打开"
def get_current_time():
    now = datetime.now()
    time_str = now.strftime("%Y年%m月%d日 %H:%M:%S")
    # 移除原来的提示文字，工具执行函数里会另加强制后缀
    return time_str

_weather_cache = {}
_weather_cache_ts = {}
weather_lock = threading.Lock() 
def get_weather(city="长沙"):
    """返回城市天气字符串,如 'Light Rain +26°C'"""
    global _weather_cache, _weather_cache_ts
    with weather_lock:
        now = time.time()
        if city in _weather_cache and now - _weather_cache_ts.get(city, 0) < 300:
            return _weather_cache[city]
    try:
        url = f"https://wttr.in/{city}?format=%C+%t&lang=zh"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            weather = resp.text.strip()
            with weather_lock:
                _weather_cache[city] = weather
                _weather_cache_ts[city] = time.time()
            return weather
        else:
            return "天气信息暂时无法获取"
    except Exception as e:
        logging.error(f"天气获取失败: {e}")
        return "天气服务异常"

def parse_schedule(user_input):
    """从用户输入中提取事件名和 datetime, 返回 (event_name, datetime) 或 (None,None)"""
    match = re.search(
        r'(明天|后天|今天)\s*(上午|下午|晚上)?\s*(\d+)\s*点\s*(.*)',
        user_input
    )
    if match:
        day_word = match.group(1)
        am_pm = match.group(2)
        hour = int(match.group(3))
        event_name = match.group(4).strip()
        now = datetime.now()
        if day_word == "今天":
            date = now.date()
        elif day_word == "明天":
            date = now.date() + timedelta(days=1)
        elif day_word == "后天":
            date = now.date() + timedelta(days=2)
        else:
            return None, None
    
        if am_pm:
            if am_pm in ("下午", "晚上") and hour != 12:
                hour += 12
            elif am_pm == "上午" and hour == 12:
                hour = 0
        dt = datetime.combine(date, datetime.min.time().replace(hour=hour))
        if dt < now:
            dt += timedelta(days=1)
        return event_name, dt
    return None, None

def add_calendar_event(event_name, dt):
    cal = Calendar()
    file_path = "schedule.ics"
    try:
        with open(file_path, 'r') as f:
            cal = Calendar(f.read())
    except FileNotFoundError:
        pass
    
    e = Event()
    e.name = event_name
    e.begin = dt
    e.uid = str(uuid.uuid4())
    cal.events.add(e)

    with open(file_path, 'w') as f:
        f.write(str(cal))
    
    delta = (dt - datetime.now()).total_seconds()
    if delta > 0:
        remind_seconds = delta - 600
        if remind_seconds > 0:
            threading.Timer(
                remind_seconds,
                lambda: notifier.show_toast("日程提醒", f"{event_name} 将在10分钟后开始")
            ).start()
        else:
            notifier.show_toast("日程提醒", f"{event_name} 即将开始")
    return f"已添加进日程: {event_name} 在 {dt.strftime('%Y-%m-%d %H:%M')}"

def set_reminder(minutes: int, message: str):
    def remind():
        notifier.show_toast("提醒", message, duration=5)
    timer = threading.Timer(minutes * 60, remind)
    timer.daemon = True
    timer.start()
    habit_recorder.record_event("set_reminder", f"{minutes}分钟后: {message}")
    return f"已设置 {minutes} 分钟后提醒: {message}"

# 增强的工具定义：描述中加入强制调用指令
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_cpu_usage",
            "description": "获取当前CPU使用率百分比。当用户询问CPU、处理器使用率时必须调用。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_usage",
            "description": "获取当前内存使用率百分比。当用户询问内存、RAM使用率时必须调用。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前真实的系统日期和时间。当用户询问时间、几点、日期时必须调用此工具，严禁自己猜测或编造。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_notepad",
            "description": "打开记事本应用程序。当用户要求打开记事本时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气。当用户询问天气时必须调用此工具，禁止自己编造。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，默认是长沙"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "设置一个定时提醒。当用户要求设置提醒、闹钟、定时时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "integer", "description": "多少分钟后提醒"},
                    "message": {"type": "string", "description": "提醒内容"}
                },
                "required": ["minutes", "message"]
            }
        }
    }
]

def execute_tool_call(tool_call):
    func_name = tool_call["function"]["name"]
    raw_args = tool_call["function"].get("arguments", {})
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return f"工具参数解析失败: {raw_args[:50]}..."
    else:
        args = raw_args
    try:
        if func_name == "get_cpu_usage":
            result = get_cpu_usage()
            return f"{result}% (真实CPU使用率，请直接采用)"
        elif func_name == "get_memory_usage":
            result = get_memory_usage()
            return f"{result}% (真实内存使用率，请直接采用)"
        elif func_name == "open_notepad":
            return open_notepad()
        elif func_name == "get_current_time":
            result = get_current_time()
            # 添加强制后缀，确保LLM完全采纳
            return f"{result} 【请严格使用上述日期和时间回答，不得编造或省略。】"
        elif func_name == "set_reminder":
            return set_reminder(args["minutes"], args["message"])
        elif func_name == "get_weather":
            result = get_weather(args.get("city", "长沙"))
            return f"{result} 【请直接告知用户上述天气，不要添加额外描述。】"
    except Exception as e:
        logging.error(f"工具执行异常 {func_name}: {e}")
        return f"执行{func_name}时出错"

def rule_based_dispatch(user_input):
    text = user_input.lower()
    if "cpu" in text:
        return get_cpu_usage, None
    if "内存" in text or "memory" in text:
        return get_memory_usage, None
    if "记事本" in text or "notepad" in text:
        return open_notepad, None
    if "时间" in text or "几点" in text:
        return get_current_time, None
    if "天气" in text:
        return get_weather, None
    if "提醒" in text or "分钟后" in text:
        match = re.search(r'(\d+)\s*分钟', text)
        if match:
            minutes = int(match.group(1))
            pattern = r'(?:提醒(?:我|一下)?\s*)?\d+\s*分钟后?\s*(?:提醒|喊|叫|说)?\s*(.*?)(?:$|[。！\.;\n])'
            msg_match = re.search(pattern, text)
            message = msg_match.group(1).strip() if msg_match and msg_match.group(1) else "时间到了"
            return set_reminder, (minutes, message)
    return None, None

client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(
    name="conversation_memory",
    embedding_function=embedding_fn
)

def retrieve_relevant_memories(query: str, n_results: int = 3) -> list:
    with chromadb_lock:
        if collection.count() == 0:
            return []
        try:
            results = collection.query(query_texts=[query], n_results=n_results)
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logging.error(f"记忆检索失败: {e}")
            return []

def add_memory(user_input: str, ai_response: str):
    memory_text = f"用户说: {user_input}\nAI回答: {ai_response}"
    if len(memory_text) > 1000:
        memory_text = memory_text[:1000] + "..."
    timestamp = datetime.now().timestamp()
    with chromadb_lock:
        try:
            collection.add(
                documents=[memory_text],
                ids=[f"mem_{timestamp}"],
                metadatas=[{"timestamp": datetime.now().isoformat()}]
            )
            if collection.count() > 500:
                all_ids = collection.get()['ids']
                mem_ids = [id_ for id_ in all_ids if id_.startswith("mem_")]
                if mem_ids:        
                    sorted_ids = sorted(mem_ids, key=lambda x: float(x.split('_')[1]))
                    collection.delete(ids=sorted_ids[:100])
        except Exception as e:
            logging.error(f"添加记忆失败: {e}")

class Monitor:
    def __init__(self, on_auto_message_callback=None):
        self.last_activity_time = datetime.now()
        self.on_auto_message = on_auto_message_callback
        self._last_battery_alert = 0
        self._last_idle_alert = 0
        self._last_cpu_alert = 0
        self._cpu_checking = False
        self._cpu_lock = threading.Lock()
        self.active_window_start_time = time.time()
        self.last_window_title = ""
        self._last_meeting_suggest = 0
        self._foreground_check_lock = threading.Lock()
        self._start_listeners()
    def _update_activity(self):
        self.last_activity_time = datetime.now()
    def _start_listeners(self):
        def on_move(x, y): self._update_activity()
        def on_click(x, y, button, pressed): self._update_activity()
        def on_scroll(x, y, dx, dy): self._update_activity()
        def on_press(key): self._update_activity()

        self.mouse_listener = MouseListener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
        self.keyboard_listener = KeyboardListener(on_press=on_press)
        self.mouse_listener.daemon = True
        self.keyboard_listener.daemon = True
        self.mouse_listener.start()
        self.keyboard_listener.start()
    def stop(self):
        if hasattr(self, 'mouse_listener'):
            self.mouse_listener.stop()
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
    def _get_active_window_title(self):
        try:
            win = gw.getActiveWindow()
            if win:
                return win.title
        except Exception as e:
            logging.error(f"获取窗口失败: {e}")
        return ""
    def _check_foreground_app(self):
        with self._foreground_check_lock:
            title = self._get_active_window_title()
            if not title:
                return
            now = time.time()
            if title != self.last_window_title:
                self.last_window_title = title
                self.active_window_start_time = now
                return
            duration_seconds = now - self.active_window_start_time
            if duration_seconds > 7200:
                msg = f"您已经连续使用「{title}」超过2小时，建议休息一下。"
                auto_queue.put(msg)
                notifier.show_toast("健康提醒", msg, duration=8)
                self.active_window_start_time = now
                return
            meeting_keywords = ["会议", "腾讯会议", "Zoom", "Meeting", "Teams"]
            if any(kw in title for kw in meeting_keywords):
                if now - self._last_meeting_suggest > 3600:
                    self._last_meeting_suggest = now
                    msg = "检测到您可能在开会, 需要我帮忙记录日程吗？"
                    auto_queue.put(msg)
                    notifier.show_toast("会议提醒", msg, duration=8)
    def _check_battery(self):
        battery = psutil.sensors_battery()
        if battery is None:
            return
        percent = battery.percent
        if percent < BATTERY_LOW_THRESHOLD and not battery.power_plugged:
            now = time.time()
            if now - self._last_battery_alert > 300:
                self._last_battery_alert = now
                msg = f"当前电量仅剩 {percent}% 请及时连接电源。"
                notifier.show_toast("电量警告", msg, duration=10)
                if self.on_auto_message:
                    self.on_auto_message(f"[系统提示] {msg}")
    def _check_cpu_load(self):
        with self._cpu_lock:
            if self._cpu_checking:
                return
            self._cpu_checking = True
        def _sample():
            try:
                samples = [psutil.cpu_percent(interval=1) for _ in range(CPU_SAMPLE_DURATION)]
                avg = sum(samples) / len(samples)
                if avg > CPU_HIGH_THRESHOLD:
                    now = time.time()
                    if now - self._last_cpu_alert > 600:
                        self._last_cpu_alert = now
                        procs = [p for p in psutil.process_iter(['name', 'cpu_percent'])
                                 if p.info['cpu_percent'] is not None]
                        procs.sort(key=lambda p: p.info['cpu_percent'], reverse=True)
                        top = procs[0].info['name'] if procs else "未知进程"
                        msg = f"过去 {CPU_SAMPLE_DURATION//60} 分钟 CPU 平均 {avg:.1f}%, {top} 占用较高。"
                        notifier.show_toast("CPU高负载", msg, duration=10)
                        if self.on_auto_message:
                            self.on_auto_message(f"[系统提示] {msg}")
            finally:
                with self._cpu_lock:
                    self._cpu_checking = False    
        threading.Thread(target=_sample, daemon=True).start()
    def _check_idle(self):
        idle_sec = (datetime.now() - self.last_activity_time).total_seconds()
        if idle_sec > IDLE_MINUTES * 60:
            now = time.time()
            if now - self._last_idle_alert > 600:
                self._last_idle_alert = now
                msg = f"你已经 {IDLE_MINUTES} 分钟没有操作电脑了，是否需要休息或锁定屏幕?"
                notifier.show_toast("空闲提醒", msg, duration=10)
                if self.on_auto_message:
                    self.on_auto_message(f"[系统提示] {msg}")
    def start(self):
        def loop():
            last_cpu = 0
            last_app_check = 0
            while True:
                self._check_battery()
                now = time.time()
                if now - last_cpu >= CPU_CHECK_INTERVAL:
                    self._check_cpu_load()
                    last_cpu = now
                self._check_idle()
                if now - last_app_check >= 300:
                    self._check_foreground_app()
                    last_app_check = now
                time.sleep(MONITOR_INTERVAL)
        threading.Thread(target=loop, daemon=True).start()

def call_ollama(messages, tools=None, max_retries=2):
    payload = {"model": MODEL_NAME, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logging.warning(f"Ollama超时 {attempt+1}")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama连接异常: {e}")
            return {"message": {"content": "抱歉，模型服务暂不可用，请检查 Ollama 是否运行。"}}
    return {"message": {"content": "未知错误"}}

def generate_templated_response(func, result):
    """根据性格生成事实查询的回答模板，确保准确性"""
    with persona_lock:
        persona = PERSONA
    # 定义不同性格的回复模板
    if "温柔" in persona:
        template_map = {
            get_current_time: "现在的时间是 {result} 呢~",
            get_weather: "为您查询到天气：{result} 呢~",
            get_cpu_usage: "当前 CPU 使用率为 {result} 呢~",
            get_memory_usage: "当前内存使用率为 {result} 呢~",
        }
    elif "冷酷" in persona:
        template_map = {
            get_current_time: "时间：{result}。",
            get_weather: "天气：{result}。",
            get_cpu_usage: "CPU：{result}。",
            get_memory_usage: "内存：{result}。",
        }
    elif "活泼" in persona:
        template_map = {
            get_current_time: "主上，现在是 {result} 哦 (=^▽^=)",
            get_weather: "报告主上！天气是 {result} (=^▽^=)",
            get_cpu_usage: "CPU正在以 {result} 的强度工作呢~",
            get_memory_usage: "内存占用 {result} 啦~",
        }
    else:
        template_map = {
            get_current_time: "现在是 {result}。",
            get_weather: "天气：{result}。",
            get_cpu_usage: "CPU使用率：{result}。",
            get_memory_usage: "内存使用率：{result}。",
        }
    template = template_map.get(func, "{result}")
    return template.format(result=result)
def process_input(user_input: str, conversation_messages: list):
    global PERSONA
    habit_recorder.record_event("user_query", user_input)

    # 性格切换检查
    for keyword in PERSONA_PRESETS:
        if f"变{keyword}" in user_input:
            switch_msg = switch_persona(keyword)
            temp_messages = [
                msg for msg in conversation_messages
                if msg["role"] in ("user", "assistant", "tool")
            ]
            temp_messages.append({"role": "user", "content": user_input})
            temp_messages.append({"role": "assistant", "content": switch_msg})
            add_memory(user_input, switch_msg)
            if len(temp_messages) > MAX_HISTORY:
                temp_messages = temp_messages[-MAX_HISTORY:]
            return switch_msg, temp_messages
    
    # 日程解析检查
    event_name, event_dt = parse_schedule(user_input)
    if event_name:
        result = add_calendar_event(event_name, event_dt)
        temp_messages = [
            msg for msg in conversation_messages
            if msg["role"] in ("user", "assistant", "tool")
        ]
        temp_messages.append({"role": "user", "content": user_input})
        temp_messages.append({"role": "assistant", "content": result})
        add_memory(user_input, result)
        if len(temp_messages) > MAX_HISTORY:
            temp_messages = temp_messages[-MAX_HISTORY:]
        return result, temp_messages

    # ========== 前置拦截：纯事实查询，完全绕过 LLM ==========
    fact_func, fact_args = rule_based_dispatch(user_input)
    if fact_func and fact_func in (get_current_time, get_weather, get_cpu_usage, get_memory_usage):
        # 执行工具，获取真实数据
        result = fact_func(*fact_args) if fact_args else fact_func()
        answer = generate_templated_response(fact_func, result)
        # 维护对话历史
        temp_messages = [msg for msg in conversation_messages if msg["role"] in ("user", "assistant", "tool")]
        temp_messages.append({"role": "user", "content": user_input})
        temp_messages.append({"role": "assistant", "content": answer})
        add_memory(user_input, answer)
        if len(temp_messages) > MAX_HISTORY:
            temp_messages = temp_messages[-MAX_HISTORY:]
        return answer, temp_messages
    # =======================================================

    # 对于其他输入（包括提醒、记事本、闲聊等），走正常的 LLM 流程
    current_messages = []
    with persona_lock:
        current_messages.append({"role": "system", "content": PERSONA})
    memories = retrieve_relevant_memories(user_input)
    if memories:
        mem_context = "相关历史对话:\n" + "\n".join(memories)
        current_messages.append({"role": "system", "content": mem_context})
    
    # 强制工具调用指令（对非事实查询仍然有效）
    force_tool_prompt = (
        "当用户询问时间、日期、天气、CPU使用率、内存使用率、打开记事本、设置提醒时，"
        "你必须调用对应的工具（get_current_time, get_weather, get_cpu_usage, get_memory_usage, open_notepad, set_reminder）获取真实信息，"
        "绝对禁止自己编造。如果没有这些工具可供调用，请告知用户暂时无法提供准确信息。"
    )
    current_messages.append({"role": "system", "content": force_tool_prompt})
    
    history_no_system = [
        msg for msg in conversation_messages
        if msg["role"] in ("user", "assistant", "tool")
    ]
    current_messages.extend(history_no_system)
    current_messages.append({"role": "user", "content": user_input})

    resp = call_ollama(current_messages, tools=TOOLS)
    print("[DEBUG] LLM 原始响应:", resp)
    answer = None

    if "message" in resp and "tool_calls" in resp["message"]:
        # 处理 function calling
        assistant_message = resp["message"]
        current_messages.append(assistant_message)
        for tc in assistant_message["tool_calls"]:
            result = execute_tool_call(tc)
            current_messages.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": tc.get("id", "unknown")
            })
        resp2 = call_ollama(current_messages)
        answer = resp2["message"]["content"]
        current_messages.append({"role": "assistant", "content": answer})
    else:
        # 没有 tool_calls，尝试规则分发（主要处理 open_notepad、set_reminder 等）
        func, args = rule_based_dispatch(user_input)
        if func:
            try:
                result = func(*args) if args else func()
                # 对于提醒、记事本等，直接返回结果，不再调 LLM
                answer = str(result)
                current_messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                logging.error(f"规则执行错误: {e}")
                answer = "抱歉，执行操作时出错了。"
                current_messages.append({"role": "assistant", "content": answer})
        else:
            answer = resp["message"]["content"]
            current_messages.append({"role": "assistant", "content": answer})

    cleaned = [msg for msg in current_messages if msg["role"] in ("user", "assistant", "tool")]
    if len(cleaned) > MAX_HISTORY:
        cleaned = cleaned[-MAX_HISTORY:]
    add_memory(user_input, answer)
    return answer, cleaned
monitor = None

def auto_callback(msg):
    auto_queue.put(msg)

def safe_speak(speaker, text):
    speaker.speak(text)

def main():
    global monitor
    print("智能管家已启动。输入'exit'退出。")
    print("后台监控: 电量,CPU,空闲状态。")
    logging.info("管家启动")

    use_voice = input("是否启用语音输入？(y/n): ").strip().lower() == 'y'

    monitor = Monitor(on_auto_message_callback=auto_callback)
    monitor.start()
    threading.Thread(target=habit_analysis_loop, daemon=True).start()
    if use_voice:
        threading.Thread(target=_load_whisper_background, daemon=True).start()
    history = []
    try:
        while True:
            try:
                while True:
                    msg = auto_queue.get_nowait()
                    print(f"\n[系统主动] {msg}")
            except queue.Empty:
                pass
        
            if use_voice:
                user_input = listen_whisper()
                if not user_input:
                    continue
            else:
                try:
                    user_input = input("\n你:")
                except EOFError:
                    break
            if user_input.lower() == 'exit':
                break
            if not user_input.strip():
                continue
            answer, history = process_input(user_input, history)
            print(f"AI: {answer}")
            speaker.speak(answer)

    except KeyboardInterrupt:
        print("\n正在停止监控...")
    finally:
        if monitor:
            monitor.stop()
        speaker.stop()
        print("智能管家已退出")

if __name__ == "__main__":
    main()