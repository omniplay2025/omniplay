import pygame
import os
import random
import pickle
import numpy as np
import base64
import mimetypes
try:
    import faiss
except ImportError:
    faiss = None
    print("Warning: FAISS library not found. Audio retrieval for musical notes will use placeholders.")
try:
    from openai import OpenAI, InternalServerError, RateLimitError # Using OpenAI client
except ImportError:
    OpenAI = None
    print("Warning: OpenAI library not found. Audio retrieval will use placeholders.")


# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED_COLOR = (255, 0, 0) # Renamed to avoid conflict with NOTE_RED if used
BLUE_COLOR = (0, 0, 255) # Renamed
YELLOW_COLOR = (255, 255, 0) # Renamed
PURPLE_COLOR = (128, 0, 128) # Renamed
ORANGE_COLOR = (255, 165, 0) # Renamed
GREY_COLOR = (128, 128, 128) # Renamed

LIGHT_GREEN = (170, 255, 170)
LIGHT_RED = (255, 170, 170)
LIGHT_BLUE = (173, 216, 230) # For UI elements
DARK_GREY = (50, 50, 50)     # For button backgrounds or text
BUTTON_HOVER_COLOR = (100, 100, 150)
BUTTON_SELECTED_COLOR = (50, 150, 50) # Greenish tint for selected

# --- Asset Paths ---
# 修改路径，使用相对路径
MAIN_DIR = "game"
ASSETS_DIR = os.path.join(MAIN_DIR, "assets-necessay")
KENNEY_DIR = os.path.join(MAIN_DIR, "assets-necessay", "kenney")
KENNEY_AUDIO_DIR = os.path.join(KENNEY_DIR, "Audio")
# ... other KENNEY paths ...
SFX_DIR = os.path.join(ASSETS_DIR, "sfx")
TEXTURES_DIR = os.path.join(ASSETS_DIR, "textures")

# --- RAG Paths & Config (from rag_fps_pipeline.py) ---
INDEX_PATH = "game/assets-necessay/rag-pipeline/kenney.index"
META_PATH = "game/assets-necessay/rag-pipeline/kenney_meta.pkl"

# IMPORTANT: In a real application, use environment variables for API keys.
API_BASE = "" # From rag_fps_pipeline.py
API_KEY = "" # From rag_fps_pipeline.py
USER_TAG = "sound_alchemist_game_rag" # Custom user tag
MODEL_EMB = "text-embedding-3-small" # From rag_fps_pipeline.py

# --- Game States ---
MENU = "menu"
PLAYING = "playing" # General exploration/hub state
PUZZLE_1 = "puzzle_1" # The original placeholder puzzle
PUZZLE_MELODY = "puzzle_melody"
PUZZLE_COMPLETE = "puzzle_complete" # New state for celebrating puzzle completion

# --- Musical Notes Definition ---
NOTE_DO = "do"
NOTE_RE = "re"
NOTE_MI = "mi"
NOTE_FA = "fa"
NOTE_SO = "so"
NOTE_LA = "la" 
NOTE_TI = "si" # 更改为si代替以前的ti，以与文件命名一致

# Full scale for RAG retrieval and potential future use
FULL_MUSICAL_SCALE_NOTES = [NOTE_DO, NOTE_RE, NOTE_MI, NOTE_FA, NOTE_SO, NOTE_LA, NOTE_TI]

# Notes that are currently interactive in the puzzle (mapped to colored blocks)
INTERACTIVE_MUSICAL_NOTES = [NOTE_DO, NOTE_RE, NOTE_MI, NOTE_FA, NOTE_SO, NOTE_LA, NOTE_TI]


NOTE_DISPLAY_NAMES = {
    NOTE_DO: "Do",
    NOTE_RE: "Re",
    NOTE_MI: "Mi",
    NOTE_FA: "Fa",
    NOTE_SO: "Sol", # 更改为Sol以更符合标准表示
    NOTE_LA: "La",
    NOTE_TI: "Si", # 更改为Si
}

# 音符文件路径
MUSIC_DIR = os.path.join(MAIN_DIR, "assets-necessay", "kenney", "music")

# 直接使用指定的音符文件
MUSIC_FILE_PATHS = {
    NOTE_DO: os.path.join(MUSIC_DIR, "note-do.mp3"),
    NOTE_RE: os.path.join(MUSIC_DIR, "note-re.mp3"), 
    NOTE_MI: os.path.join(MUSIC_DIR, "note-mi.mp3"),
    NOTE_FA: os.path.join(MUSIC_DIR, "note-f.mp3"),    # 注意文件名为f而非fa
    NOTE_SO: os.path.join(MUSIC_DIR, "note-salt.mp3"), # 注意文件名为salt而非so/sol
    NOTE_LA: os.path.join(MUSIC_DIR, "note-la.mp3"),
    NOTE_TI: os.path.join(MUSIC_DIR, "note-c.mp3"),    # 注意文件名为c而非si
}

# 移除预检索声音，直接使用我们指定的音乐文件
PRE_RETRIEVED_SOUNDS = {
    NOTE_DO: MUSIC_FILE_PATHS[NOTE_DO],
    NOTE_RE: MUSIC_FILE_PATHS[NOTE_RE],
    NOTE_MI: MUSIC_FILE_PATHS[NOTE_MI],
    NOTE_FA: MUSIC_FILE_PATHS[NOTE_FA],
    NOTE_SO: MUSIC_FILE_PATHS[NOTE_SO],
    NOTE_LA: MUSIC_FILE_PATHS[NOTE_LA],
    NOTE_TI: MUSIC_FILE_PATHS[NOTE_TI],
}

# --- Difficulty Levels ---
DIFFICULTY_EASY = "Easy"
DIFFICULTY_MEDIUM = "Medium"
DIFFICULTY_HARD = "Hard"

DIFFICULTY_SETTINGS = {
    DIFFICULTY_EASY: {"sequence_length": 3, "score_multiplier": 1, "name": "Easy"},
    DIFFICULTY_MEDIUM: {"sequence_length": 5, "score_multiplier": 2, "name": "Medium"},
    DIFFICULTY_HARD: {"sequence_length": 7, "score_multiplier": 3, "name": "Hard"} # Allows note repetition
}

# --- Pygame Setup ---
pygame.init()
pygame.mixer.init() # For sound
# Allow more sound channels for simultaneous playback
pygame.mixer.set_num_channels(16)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("The Sound Alchemist's Secret Chamber") # Changed to English
clock = pygame.time.Clock()

# --- Game Variables ---
current_state = MENU
running = True
player_melody_input = [] # Stores the player's current melody sequence
last_puzzle_solved = "" # To know which puzzle led to PUZZLE_COMPLETE
current_difficulty = DIFFICULTY_MEDIUM # Default difficulty
correct_melody_sequence = [] # Will be generated dynamically
melody_puzzle_attempts = 0
player_score = 0
mouse_pos = (0,0) # To store mouse position for hover effects
auto_start_enabled = True
# 新增：是否启用自动开始模式

# Initialize sprite groups
all_game_sprites = pygame.sprite.Group()
note_elements = pygame.sprite.Group()
particles_group = pygame.sprite.Group() 

# Define note_size here, as it's used in PUZZLE_MELODY setup below
note_size = (100, 100)

# --- RAG Setup ---
faiss_index = None
meta_data = None
openai_client = None

if OpenAI:
    try:
        openai_client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        print("OpenAI client initialized for RAG.")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}. Audio retrieval will use placeholders.")
        openai_client = None
else:
    print("OpenAI library not available. Using placeholder sounds for RAG.")

def get_text_embedding_openai(text):
    """
    Generates a text embedding using the OpenAI API.
    """
    if openai_client is None:
        print("Error: OpenAI client not initialized. Cannot generate embeddings.")
        return None
    try:
        # Uses the global openai_client and MODEL_EMB
        response = openai_client.embeddings.create(model=MODEL_EMB, input=[text], user=USER_TAG)
        embedding_vector = np.asarray(response.data[0].embedding, "float32")
        return np.expand_dims(embedding_vector, axis=0) # Ensure 2D for FAISS
    except Exception as e:
        print(f"Error generating text embedding for '{text}' via OpenAI: {e}")
        return None

def retrieve_sound_path_from_rag(query_text, index, metadata_list, top_k=1):
    """
    Retrieves the full path to a sound file based on a text query using FAISS and OpenAI embeddings.
    Assumes metadata_list stores paths relative to KENNEY_DIR.
    """
    if index is None or openai_client is None or not metadata_list:
        return None

    query_embedding = get_text_embedding_openai(query_text)
    if query_embedding is None:
        return None
    
    try:
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        if top_k == 1 and len(indices[0]) > 0:
            retrieved_idx = indices[0][0]
            if 0 <= retrieved_idx < len(metadata_list):
                relative_path = metadata_list[retrieved_idx] 
                full_path = os.path.join(KENNEY_DIR, relative_path)

                if os.path.exists(full_path):
                    print(f"Retrieved for '{query_text}': {full_path} (Score: {distances[0][0]})")
                    return full_path
                else:
                    print(f"Retrieved metadata for '{query_text}' but file not found: {full_path}")
                    return None
            else:
                print(f"Retrieved index {retrieved_idx} out of bounds for metadata list.")
                return None
        else:
            return None
    except Exception as e:
        print(f"Error during FAISS search or metadata lookup for '{query_text}': {e}")
        return None

# Load RAG components at startup
if faiss and openai_client: 
    if os.path.exists(INDEX_PATH):
        try:
            faiss_index = faiss.read_index(INDEX_PATH)
            print(f"FAISS index loaded from {INDEX_PATH}. Index size: {faiss_index.ntotal}")
        except Exception as e:
            print(f"Failed to load FAISS index: {e}. Audio retrieval will use placeholders.")
            faiss_index = None
    else:
        print(f"FAISS index file not found at {INDEX_PATH}. Audio retrieval will use placeholders.")
        faiss_index = None

    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "rb") as f:
                meta_data = pickle.load(f) 
            print(f"Metadata loaded from {META_PATH}. Number of entries: {len(meta_data)}")
        except Exception as e:
            print(f"Failed to load metadata: {e}. Audio retrieval will use placeholders.")
            meta_data = None
    else:
        print(f"Metadata file not found at {META_PATH}. Audio retrieval will use placeholders.")
        meta_data = None
else:
    if not faiss:
        print("FAISS not available. Using placeholder sounds.")
    if not openai_client:
         print("OpenAI client not initialized. Using placeholder sounds.")


# --- Helper Functions ---
sound_file_paths = {}

def load_sound(name, volume=1.0, directory=SFX_DIR):
    """加载声音文件并记录文件路径"""
    global sound_file_paths
    
    if directory is None: # name is expected to be a full path
        path = name
    else:
        path = os.path.join(directory, name)
    
    # 添加调试信息
    #print(f"尝试加载音频文件: {path}")
    #print(f"当前工作目录: {os.getcwd()}")
    #print(f"文件是否存在: {os.path.exists(path)}")
    
    # 如果文件不存在，尝试从项目根目录开始查找
    if not os.path.exists(path) and path.startswith("game/"):
        # 尝试计算基于项目根目录的路径
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alternative_path = os.path.join(base_dir, path)
        print(f"尝试替代路径: {alternative_path}")
        print(f"替代路径是否存在: {os.path.exists(alternative_path)}")
        if os.path.exists(alternative_path):
            path = alternative_path
    
    try:
        # 尝试加载音频文件，处理可能的MP3 ID3标签问题
        sound = pygame.mixer.Sound(path)
        sound.set_volume(volume)
        # 将声音对象和路径添加到映射字典中，而不是直接添加属性
        sound_file_paths[sound] = path
        print(f"成功加载音频: {path}")
        return sound
    except FileNotFoundError: # Specifically catch if the file isn't there
        print(f"Warning: Sound file not found at {path}")
        return None
    except pygame.error as e: # Catch Pygame-specific errors (e.g., unsupported format)
        # 如果是MP3文件的ID3标签问题，尝试忽略警告继续加载
        if "id3" in str(e).lower() or "comment" in str(e).lower():
            print(f"Warning: MP3 ID3 tag issue (continuing anyway): {path}")
            try:
                # 尝试强制加载，忽略ID3标签警告
                sound = pygame.mixer.Sound(path)
                sound.set_volume(volume)
                # 将声音对象和路径添加到映射字典中
                sound_file_paths[sound] = path
                return sound
            except Exception as inner_e:
                print(f"Failed to load sound even after ignoring ID3 warnings: {path} - Error: {inner_e}")
                return None
        else:
            print(f"Warning: Cannot load sound (pygame error): {path} - {e}")
            return None
    except Exception as e: # Catch any other unexpected errors during loading
        print(f"Warning: An unexpected error occurred while loading sound {path}: {e}")
        return None


def play_sound(sound):
    """播放声音并记录播放的音频文件路径"""
    if sound:
        sound.play()
        # 记录播放的音频文件路径（使用映射字典）
        if sound in sound_file_paths:
            pygame.mixer._last_played_sound_file = sound_file_paths[sound]
        else:
            pygame.mixer._last_played_sound_file = None

def get_last_played_audio_data():
    """获取最近播放的音频数据（转换为numpy数组）"""
    # 检查是否有最近播放的声音记录
    if not hasattr(pygame.mixer, '_last_played_sound_file') or pygame.mixer._last_played_sound_file is None:
        # 没有最近播放的音频文件记录，检查是否有音符被点击
        for note_sprite in note_elements:
            if note_sprite.is_highlighted and note_sprite.sound:
                # 返回当前高亮音符的声音
                if note_sprite.sound in sound_file_paths:
                    print(f"返回当前高亮音符的声音: {sound_file_paths[note_sprite.sound]}")
                    return note_sprite.sound
        # 如果没有高亮音符，返回None
        print("没有最近播放的音频，也没有高亮音符")
        return None
    
    try:
        import librosa
        import soundfile as sf
        
        sound_file_path = pygame.mixer._last_played_sound_file
        
        # 使用librosa加载音频文件并转换为16kHz单声道
        audio_data, sample_rate = librosa.load(sound_file_path, sr=16000, mono=True)
        
        # 确保音频长度符合环境要求（1秒 = 16000采样点）
        target_length = 16000
        if len(audio_data) > target_length:
            # 如果音频太长，截取前1秒
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            # 如果音频太短，用零填充到1秒
            padding = target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant', constant_values=0)
        
        # 清除记录
        pygame.mixer._last_played_sound_file = None
        
        return audio_data.astype(np.float32)
        
    except ImportError:
        print("Warning: librosa not available, cannot convert audio file to numpy array")
        return None
    except Exception as e:
        print(f"Error loading audio data from {sound_file_path}: {e}")
        return None

def load_image(name, directory=TEXTURES_DIR, convert_alpha=True):
    """Loads an image file from the specified directory."""
    path = os.path.join(directory, name)
    try:
        image = pygame.image.load(path)
        if convert_alpha:
            image = image.convert_alpha()
        else:
            image = image.convert()
        return image
    except pygame.error as e:
        print(f"Cannot load image: {path} - {e}")
        return None

# --- UI Button Class ---
class Button:
    def __init__(self, text, x, y, width, height, font, text_color, base_color, hover_color, selected_color, value):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.text_color = text_color
        self.base_color = base_color
        self.hover_color = hover_color
        self.selected_color = selected_color
        self.value = value 
        self.is_hovered = False

    def draw(self, screen, is_selected=False):
        current_color = self.base_color
        if is_selected:
            current_color = self.selected_color
        elif self.is_hovered:
            current_color = self.hover_color
        
        pygame.draw.rect(screen, current_color, self.rect, border_radius=5)
        pygame.draw.rect(screen, WHITE, self.rect, width=2, border_radius=5) 

        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def update_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)

# --- Game Objects ---
class PuzzleElement(pygame.sprite.Sprite):
    def __init__(self, image, x, y, sound_name=None, element_id=None, original_color=None):
        super().__init__()
        self.original_image = image 
        self.image = image.copy() 
        self.rect = self.image.get_rect(topleft=(x,y))
        self.sound = None
        self.element_id = element_id
        self.original_color = original_color if original_color else image.get_at((0,0))
        self.is_animating = False
        self.animation_timer = 0
        self.animation_duration = 15 
        
        self.is_highlighted = False
        self.highlight_color_tuple = None
        self.highlight_timer = 0
        self.highlight_duration = 20 

        if sound_name and SFX_DIR: 
            self.sound = load_sound(sound_name)
        elif sound_name: 
             self.sound = load_sound(sound_name, directory=os.path.join(os.path.dirname(__file__), "assets", "sfx"))


    def interact(self):
        print(f"Interacted with element {self.element_id} at {self.rect.topleft}")
        if self.sound:
            play_sound(self.sound)
        
        self.start_click_animation()

        if current_state == PUZZLE_MELODY and self.element_id:
            handle_melody_input(self.element_id, self)

    def start_click_animation(self):
        self.is_animating = True
        self.animation_timer = self.animation_duration
        current_center = self.rect.center
        self.image = pygame.transform.smoothscale(self.original_image, 
                                             (int(self.original_image.get_width() * 1.2),
                                              int(self.original_image.get_height() * 1.2)))
        self.rect = self.image.get_rect(center=current_center)


    def highlight(self, color_tuple, duration=20):
        self.is_highlighted = True
        self.highlight_color_tuple = color_tuple
        self.highlight_timer = duration
        self.image = self.original_image.copy() 
        self.image.fill(self.highlight_color_tuple, special_flags=pygame.BLEND_RGB_MULT)


    def update(self): 
        if self.is_animating:
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self.is_animating = False
                self.image = self.original_image.copy() 
                self.rect = self.image.get_rect(center=self.rect.center) 

        if self.is_highlighted:
            self.highlight_timer -= 1
            if self.highlight_timer > 0:
                if not self.is_animating or (self.is_animating and self.animation_timer <=0) : 
                    temp_image = self.original_image.copy()
                    temp_image.fill(self.highlight_color_tuple, special_flags=pygame.BLEND_RGB_MULT)
                    current_center = self.rect.center
                    self.image = temp_image
                    self.rect = self.image.get_rect(center=current_center)
            else: 
                self.is_highlighted = False
                self.highlight_color_tuple = None
                if not self.is_animating or (self.is_animating and self.animation_timer <=0):
                    self.image = self.original_image.copy()
                    self.rect = self.image.get_rect(center=self.rect.center)
        
        if not self.is_animating and not self.is_highlighted:
            if self.image is not self.original_image: 
                 current_center = self.rect.center
                 self.image = self.original_image.copy()
                 self.rect = self.image.get_rect(center=current_center)


# --- Particle System ---
class Particle(pygame.sprite.Sprite):
    def __init__(self, x, y, color, size_range=(3, 8), vel_range_x=(-2,2), vel_range_y=(-4, -1), grav=0.1, lifespan_frames=60):
        super().__init__()
        size = random.randint(size_range[0], size_range[1])
        self.image = pygame.Surface((size, size))
        self.image.fill(BLACK) 
        pygame.draw.circle(self.image, color, (size//2, size//2), size//2) 
        self.image.set_colorkey(BLACK) 
        self.rect = self.image.get_rect(center=(x, y))
        self.velocity = pygame.math.Vector2(random.uniform(vel_range_x[0], vel_range_x[1]), 
                                           random.uniform(vel_range_y[0], vel_range_y[1]))
        self.gravity = grav
        self.lifespan = lifespan_frames
        self.initial_lifespan = lifespan_frames 

    def update(self):
        self.velocity.y += self.gravity
        self.rect.x += self.velocity.x
        self.rect.y += self.velocity.y
        self.lifespan -= 1
        if self.initial_lifespan > 0:
            alpha = max(0, int(255 * (self.lifespan / self.initial_lifespan)))
            self.image.set_alpha(alpha)
        else:
            self.image.set_alpha(0)

        if self.lifespan <= 0:
            self.kill()


def create_success_particles(num_particles=70, position=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
                             colors=[(255,215,0), (255,255,224), (255,193,37)]): 
    for _ in range(num_particles):
        color = random.choice(colors)
        particle = Particle(position[0] + random.randint(-20,20), 
                            position[1] + random.randint(-20,20), 
                            color, lifespan_frames=random.randint(40,80))
        particles_group.add(particle)

# --- Puzzle Specific Assets & Logic ---

kenney_digital_audio_placeholder_dir = os.path.join(KENNEY_AUDIO_DIR, "Digital Audio", "Audio") 
kenney_ui_audio = os.path.join(KENNEY_AUDIO_DIR, "UI Audio", "Audio")
kenney_jingles_sax = os.path.join(KENNEY_AUDIO_DIR, "Music Jingles", "Audio (Saxophone)")

# Change all feedback messages to English
melody_feedback_sounds = {
    "success": load_sound(MUSIC_FILE_PATHS[NOTE_SO], directory=None, volume=0.7)
}

if melody_feedback_sounds["success"] is None:
    print("Warning: Success sound effect could not be loaded. Game will continue without it.")

placeholder_sounds_map = {
    NOTE_DO: ("powerUp1.ogg", kenney_digital_audio_placeholder_dir),
    NOTE_RE: ("laser1.ogg", kenney_digital_audio_placeholder_dir),
    NOTE_MI: ("highUp.ogg", kenney_digital_audio_placeholder_dir),
    NOTE_FA: ("phaserUp6.ogg", kenney_digital_audio_placeholder_dir),
    NOTE_SO: ("pepSound1.ogg", kenney_digital_audio_placeholder_dir),
    NOTE_LA: ("pepSound2.ogg", kenney_digital_audio_placeholder_dir), 
    NOTE_TI: ("powerUp3.ogg", kenney_digital_audio_placeholder_dir),  
}

# 更新音符声音加载逻辑
melody_note_sounds = {}

for note_id in FULL_MUSICAL_SCALE_NOTES:
    note_name_for_query = NOTE_DISPLAY_NAMES.get(note_id, "musical sound")
    sound_path = PRE_RETRIEVED_SOUNDS[note_id]
    
    # 暂时禁用pygame的音频警告输出
    import os
    old_stderr = os.dup(2)
    os.close(2)
    os.open(os.devnull, os.O_RDWR)
    
    try:
        sound = load_sound(sound_path, directory=None)
        if sound:
            melody_note_sounds[note_id] = sound
            print(f"Successfully loaded sound effect for note {note_name_for_query}: {sound_path}")
        else:
            print(f"Warning: Could not load sound effect for note {note_name_for_query}: {sound_path}")
            # 如果MP3加载失败，尝试使用原有的占位音效
            if note_id in placeholder_sounds_map:
                placeholder_file, placeholder_dir = placeholder_sounds_map[note_id]
                melody_note_sounds[note_id] = load_sound(placeholder_file, directory=placeholder_dir)
                if melody_note_sounds[note_id] is None:
                    print(f"Critical Warning: Placeholder sound effect for {note_name_for_query} ({placeholder_file}) also could not be loaded.")
    finally:
        # 恢复stderr
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

# Create custom note buttons without text labels
def create_note_button(color):
    """Creates a custom button surface for musical notes without text"""
    img = pygame.Surface(note_size, pygame.SRCALPHA)
    
    # Fill with base color but slightly transparent
    base_color = (*color[:3], 220)  # RGB + alpha
    pygame.draw.rect(img, base_color, (0, 0, note_size[0], note_size[1]), border_radius=10)
    
    # Add a highlight effect
    highlight_color = (min(color[0] + 50, 255), min(color[1] + 50, 255), min(color[2] + 50, 255), 180)
    pygame.draw.rect(img, highlight_color, (3, 3, note_size[0] - 6, 15), border_radius=7)
    
    # Add border
    border_color = (max(color[0] - 50, 0), max(color[1] - 50, 0), max(color[2] - 50, 0), 255)
    pygame.draw.rect(img, border_color, (0, 0, note_size[0], note_size[1]), width=2, border_radius=10)
    
    return img

# Create note images programmatically - no text labels
note_images = {}
note_colors = {
    "Do": RED_COLOR,
    "Re": ORANGE_COLOR,
    "Mi": YELLOW_COLOR,
    "Fa": GREEN,
    "Sol": BLUE_COLOR,
    "La": PURPLE_COLOR,
    "Si": GREY_COLOR,
}

for note_name, color in note_colors.items():
    note_images[note_name] = create_note_button(color)
    print(f"Created custom button image for note: {note_name}")

# 存储当前游戏中的颜色-音符映射关系
current_note_color_mapping = {}

# 所有可用颜色列表
ALL_COLORS = {
    "Red": RED_COLOR,
    "Orange": ORANGE_COLOR,
    "Yellow": YELLOW_COLOR,
    "Green": GREEN,
    "Blue": BLUE_COLOR,
    "Purple": PURPLE_COLOR,
    "Grey": GREY_COLOR,
}

# Create initial note elements with custom graphics based on difficulty
def create_note_elements(difficulty=None):
    global note_elements
    
    if difficulty is None:
        difficulty = current_difficulty
    
    # Get the number of notes to display based on difficulty
    settings = DIFFICULTY_SETTINGS[difficulty]
    note_count = settings["sequence_length"]
    
    # Clear existing note elements
    note_elements.empty()  
    all_game_sprites.remove(note_elements)
    
    # Layout parameters
    base_y = 250  # Center vertically
    
    # Only create positions for the correct number of note blocks
    positions = []
    
    # Calculate total width needed for all blocks with spacing
    total_width = note_count * note_size[0] + (note_count - 1) * 20
    start_x = (SCREEN_WIDTH - total_width) // 2
    
    # Create positions based on the actual number of blocks needed
    for i in range(note_count):
        positions.append((start_x + i * (note_size[0] + 20), base_y))
    
    # Randomize positions
    random.shuffle(positions)
    
    # 使用当前的音符-颜色映射
    # 如果映射为空，创建一个默认映射
    if not current_note_color_mapping:
        default_colors = {
            NOTE_DO: RED_COLOR,
            NOTE_RE: ORANGE_COLOR,
            NOTE_MI: YELLOW_COLOR,
            NOTE_FA: GREEN,
            NOTE_SO: BLUE_COLOR,
            NOTE_LA: PURPLE_COLOR,
            NOTE_TI: GREY_COLOR,
        }
        for note_id, color in default_colors.items():
            current_note_color_mapping[note_id] = color
    
    # Only create sprites for the notes in the current sequence
    for i, note_id in enumerate(correct_melody_sequence):
        if i < len(positions):
            note_name = NOTE_DISPLAY_NAMES.get(note_id, "?")
            color = current_note_color_mapping.get(note_id, (200, 200, 200))
            
            # 根据当前颜色创建自定义按钮图像
            img = create_note_button(color)
            
            element = PuzzleElement(
                img, 
                positions[i][0], 
                positions[i][1], 
                sound_name=None,
                element_id=note_id, 
                original_color=color
            )
            
            if note_id in melody_note_sounds: 
                element.sound = melody_note_sounds[note_id]
            else:
                print(f"Warning: Could not find sound for note {note_id}")
            
            note_elements.add(element)
            all_game_sprites.add(element)

def generate_new_melody_puzzle(difficulty):
    global correct_melody_sequence, player_melody_input, melody_puzzle_attempts, current_note_color_mapping
    
    settings = DIFFICULTY_SETTINGS[difficulty]
    seq_length = settings["sequence_length"]
    
    if not INTERACTIVE_MUSICAL_NOTES:
        print("Error: INTERACTIVE_MUSICAL_NOTES is not defined or empty.")
        correct_melody_sequence = []
        return

    # Use notes in the strict do-re-mi-fa-sol-la-si order
    ordered_notes = [NOTE_DO, NOTE_RE, NOTE_MI, NOTE_FA, NOTE_SO, NOTE_LA, NOTE_TI]
    
    # Randomly select a starting position in the ordered notes
    if seq_length < len(ordered_notes):
        max_start_idx = len(ordered_notes) - seq_length
        start_idx = random.randint(0, max_start_idx)
        correct_melody_sequence = ordered_notes[start_idx:start_idx + seq_length]
    else:
        correct_melody_sequence = ordered_notes[:seq_length]
    
    # 随机化颜色与音符的对应关系 - 更改这部分逻辑
    # 创建音符和颜色的副本列表用于随机分配
    available_notes = INTERACTIVE_MUSICAL_NOTES.copy()
    available_colors = list(ALL_COLORS.values())
    
    # 随机打乱两个列表
    random.shuffle(available_notes)
    random.shuffle(available_colors)
    
    # 创建新的随机音符-颜色映射
    current_note_color_mapping = {}
    for i, note_id in enumerate(available_notes):
        if i < len(available_colors):
            current_note_color_mapping[note_id] = available_colors[i]
    
    # Reset player state
    player_melody_input = []
    melody_puzzle_attempts = 0
    
    # Create note elements based on the new melody
    create_note_elements(difficulty)
    
    debug_sequence_names = [NOTE_DISPLAY_NAMES.get(n, n) for n in correct_melody_sequence]
    print(f"New melody puzzle ({difficulty}, {len(correct_melody_sequence)} notes): Sequence is {debug_sequence_names}")
    print("新的随机音符-颜色映射关系:")
    color_name_mapping = {v: k for k, v in ALL_COLORS.items()}
    current_note_color_mapping_name = {}
    
    for note, color in current_note_color_mapping.items():
        color_name = color_name_mapping.get(color, f"RGB{color}")
        print(f"{NOTE_DISPLAY_NAMES.get(note, note)}: {color_name}")
        current_note_color_mapping_name[f"{NOTE_DISPLAY_NAMES.get(note, note)}"] = color_name
    print(f"当前音符-颜色映射关系: {current_note_color_mapping_name}")

    # Modified scoring system with more detailed rules
def calculate_score(difficulty, mistakes, completion_time=None):
    """Calculate score with more sophisticated rules"""
    settings = DIFFICULTY_SETTINGS[difficulty]
    seq_length = settings["sequence_length"]
    base_score = 1000
    
    # Base multiplier from difficulty
    multiplier = settings["score_multiplier"]
    
    # Mistake penalties
    mistake_penalty = 150 * multiplier  # Harder difficulties have higher penalties
    mistake_deduction = mistakes * mistake_penalty
    
    # Perfect play bonus
    perfect_bonus = 0
    if mistakes == 0:
        perfect_bonus = 500 * multiplier
    
    # Sequential bonus - rewards completing longer sequences
    sequence_bonus = seq_length * 50 * multiplier
    
    # Calculate final score
    final_score = (base_score * multiplier) + sequence_bonus + perfect_bonus - mistake_deduction
    
    # Ensure score doesn't go negative
    final_score = max(0, final_score)
    
    # Log score calculation details
    print(f"Score Details:")
    print(f"  Base Score: {base_score} × {multiplier} = {base_score * multiplier}")
    print(f"  Sequence Bonus: {sequence_bonus}")
    print(f"  Perfect Play Bonus: {perfect_bonus}")
    print(f"  Mistake Penalty: -{mistake_deduction}")
    print(f"  Final Score: {final_score}")
    
    return final_score

def handle_melody_input(note_id, clicked_element: PuzzleElement):
    global player_melody_input, current_state, last_puzzle_solved, melody_puzzle_attempts, player_score
    
    # 安全检查：如果没有传入有效的元素，不播放声音但继续处理逻辑
    if clicked_element and hasattr(clicked_element, 'sound') and clicked_element.sound:
        play_sound(clicked_element.sound)
    elif note_id in melody_note_sounds:
        # 如果没有元素但有对应的音效，直接播放
        play_sound(melody_note_sounds[note_id])
    
    if note_id not in INTERACTIVE_MUSICAL_NOTES:
        if clicked_element:
            clicked_element.highlight(LIGHT_RED, duration=30)
        player_melody_input = [] 
        melody_puzzle_attempts += 1
        return

    player_melody_input.append(note_id)
    
    correct_so_far = True
    for i in range(len(player_melody_input)):
        if i >= len(correct_melody_sequence) or player_melody_input[i] != correct_melody_sequence[i]:
            correct_so_far = False
            break
    
    if correct_so_far:
        if clicked_element:
            clicked_element.highlight(LIGHT_GREEN, duration=20)

        if len(player_melody_input) == len(correct_melody_sequence):
            print("Melody correct!")
            if clicked_element:
                create_success_particles(position=clicked_element.rect.center)
            else:
                create_success_particles(position=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            
            # Use the enhanced scoring system
            player_score = calculate_score(current_difficulty, melody_puzzle_attempts)
            print(f"Score: {player_score}, Attempts: {melody_puzzle_attempts}")

            player_melody_input = [] 
            
            last_puzzle_solved = PUZZLE_MELODY
            current_state = PUZZLE_COMPLETE
        
    else:
        # Don't play error sound, just highlight in red if element exists
        if clicked_element:
            clicked_element.highlight(LIGHT_RED, duration=30)
        #player_melody_input = [] 
        melody_puzzle_attempts += 1


# --- UI Elements & Fonts ---
try:
    default_font_name = pygame.font.get_default_font()
    font_large = pygame.font.Font(default_font_name, 74)
    font_medium = pygame.font.Font(default_font_name, 50)
    font_small = pygame.font.Font(default_font_name, 30)
    font_tiny = pygame.font.Font(default_font_name, 24)
except Exception as e:
    print(f"Font loading error: {e}. Using pygame.font.Font(None, size).")
    font_large = pygame.font.Font(None, 74)
    font_medium = pygame.font.Font(None, 50)
    font_small = pygame.font.Font(None, 30)
    font_tiny = pygame.font.Font(None, 24)

# --- Menu Buttons ---
difficulty_buttons = []
def setup_menu_buttons():
    global difficulty_buttons
    difficulty_buttons = [] 
    button_width = 150
    button_height = 50
    button_y_start = 240
    button_spacing = 20 
    button_x = SCREEN_WIDTH // 2 - button_width // 2

    difficulty_options_config = [
        (DIFFICULTY_SETTINGS[DIFFICULTY_EASY]["name"], DIFFICULTY_EASY),
        (DIFFICULTY_SETTINGS[DIFFICULTY_MEDIUM]["name"], DIFFICULTY_MEDIUM),
        (DIFFICULTY_SETTINGS[DIFFICULTY_HARD]["name"], DIFFICULTY_HARD)
    ]

    for i, (text, value) in enumerate(difficulty_options_config):
        button = Button(
            text=text,
            x=button_x,
            y=button_y_start + i * (button_height + button_spacing),
            width=button_width,
            height=button_height,
            font=font_small,
            text_color=WHITE,
            base_color=DARK_GREY,
            hover_color=BUTTON_HOVER_COLOR,
            selected_color=BUTTON_SELECTED_COLOR,
            value=value
        )
        difficulty_buttons.append(button)

setup_menu_buttons() 

# --- Game Logic Placeholder & Other Assets ---
# 创建自定义背景而不是加载不存在的图片
menu_background_image = None

# 创建一个自定义的音乐主题背景
def create_custom_background():
    bg_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    # 创建从深蓝色到深紫色的渐变
    for i in range(SCREEN_HEIGHT):
        # 渐变颜色
        color = (20, max(10, min(20 + i // 30, 40)), max(40, min(60 + i // 10, 120)))
        pygame.draw.line(bg_surface, color, (0, i), (SCREEN_WIDTH, i))
    
    # 添加音乐主题装饰元素
    for _ in range(30):
        # 随机位置的音符符号
        x = random.randint(0, SCREEN_WIDTH)
        y = random.randint(0, SCREEN_HEIGHT)
        size = random.randint(5, 15)
        alpha = random.randint(30, 100)
        
        note_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        color = (255, 255, 255, alpha)  # 半透明白色
        
        # 简单的音符形状
        pygame.draw.circle(note_surf, color, (size//2, size//2), size//2)
        pygame.draw.line(note_surf, color, (size-2, size//2), (size-2, size//4), 2)
        
        bg_surface.blit(note_surf, (x, y))
    
    return bg_surface

# 生成自定义背景
menu_background_image = create_custom_background()

# 不尝试加载不存在的图片
placeholder_puzzle_image = pygame.Surface((50, 50))
placeholder_puzzle_image.fill(GREEN)
clickable_element = PuzzleElement(placeholder_puzzle_image, 100, 100, None, element_id="green_box") 
# 为绿色方块设置音效（使用音符声音而非不存在的音效文件）
if melody_note_sounds and NOTE_DO in melody_note_sounds:
    clickable_element.sound = melody_note_sounds[NOTE_DO]
puzzle1_sprites = pygame.sprite.Group(clickable_element)
all_game_sprites.add(clickable_element)

def start_melody_puzzle_directly(difficulty=None):
    """直接开始旋律谜题，无需按键交互"""
    global current_state, current_difficulty, player_score, melody_puzzle_attempts
    
    if difficulty:
        current_difficulty = difficulty
    
    print(f"Auto-starting Melody Puzzle (Difficulty: {current_difficulty})...")
    generate_new_melody_puzzle(current_difficulty) 
    player_score = 0 
    melody_puzzle_attempts = 0 
    current_state = PUZZLE_MELODY
    return True

def set_auto_start_mode(enabled=True):
    """设置自动开始模式"""
    global auto_start_enabled
    auto_start_enabled = enabled
    print(f"Auto-start mode: {'enabled' if enabled else 'disabled'}")

def get_current_color_note_mapping():
    """获取当前颜色-音符映射关系（用于智能体）"""
    color_note_mapping = {}
    color_name_mapping = {v: k for k, v in ALL_COLORS.items()}
    
    for note_id, color in current_note_color_mapping.items():
        color_name = color_name_mapping.get(color, f"RGB{color}")
        color_note_mapping[color_name.lower()] = NOTE_DISPLAY_NAMES.get(note_id, note_id)
    
    return color_note_mapping

def get_current_color_to_note_id_mapping():
    """获取当前颜色名称到音符ID的映射（供环境使用）"""
    color_to_note = {}
    color_name_mapping = {v: k for k, v in ALL_COLORS.items()}
    
    for note_id, rgb_color in current_note_color_mapping.items():
        color_name = color_name_mapping.get(rgb_color)
        if color_name:
            color_to_note[color_name.upper()] = note_id
    
    return color_to_note

def get_game_state():
    """获取当前游戏状态信息"""
    return {
        "state": current_state,
        "difficulty": current_difficulty,
        "score": player_score,
        "attempts": melody_puzzle_attempts,
        "sequence_length": len(correct_melody_sequence),
        "input_length": len(player_melody_input),
        "game_over": current_state == PUZZLE_COMPLETE,
        "current_note_color_mapping": current_note_color_mapping.copy(),  # 添加当前映射信息
        "correct_sequence": correct_melody_sequence.copy(),  # 添加正确序列信息
        "current_color_to_note_mapping": get_current_color_to_note_id_mapping()  # 添加颜色到音符的映射
    }

def encode_audio(audio_path):
    # 获取文件的 MIME 类型
    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        # 根据文件扩展名设置默认 MIME 类型
        if audio_path.lower().endswith('.wav'):
            mime_type = "audio/wav"
        elif audio_path.lower().endswith('.mp3'):
            mime_type = "audio/mpeg"
        else:
            mime_type = "audio/wav"  # 默认使用 wav
    
    with open(audio_path, "rb") as audio_file:
        base64_data = base64.b64encode(audio_file.read()).decode("utf-8")
    
    return f"data:{mime_type};base64,{base64_data}"

def get_last_played_audio_data():
    if player_melody_input[-1]:
        _note_id = player_melody_input[-1]
        _sound_path = PRE_RETRIEVED_SOUNDS[_note_id]
        audio_data_ = encode_audio(_sound_path)
        return audio_data_
    try:
        import librosa
        import soundfile as sf
        
        sound_file_path = pygame.mixer._last_played_sound_file
        
        # 使用librosa加载音频文件并转换为16kHz单声道
        audio_data, sample_rate = librosa.load(sound_file_path, sr=16000, mono=True)
        
        # 确保音频长度符合环境要求（1秒 = 16000采样点）
        target_length = 16000
        if len(audio_data) > target_length:
            # 如果音频太长，截取前1秒
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            # 如果音频太短，用零填充到1秒
            padding = target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant', constant_values=0)
        
        # 清除记录
        pygame.mixer._last_played_sound_file = None
        
        return audio_data.astype(np.float32)
        
    except ImportError:
        print("Warning: librosa not available, cannot convert audio file to numpy array")
        return None
    except Exception as e:
        print(f"Error loading audio data from {sound_file_path}: {e}")
        return None

# --- Main Game Loop ---
# 只有当这个文件被直接执行时才运行游戏循环
if __name__ == "__main__":
    while running:
        mouse_pos = pygame.mouse.get_pos() 
        
        # 如果启用自动开始模式且在菜单状态，自动开始游戏
        if auto_start_enabled and current_state == MENU:
            start_melody_puzzle_directly()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    if current_state == MENU:
                        for button in difficulty_buttons:
                            if button.is_clicked(mouse_pos):
                                current_difficulty = button.value
                                print(f"Difficulty set to {DIFFICULTY_SETTINGS[current_difficulty]['name']} via button click")
                                if melody_feedback_sounds and melody_feedback_sounds.get("correct_input"):
                                    play_sound(melody_feedback_sounds["correct_input"])
                                # 如果启用自动开始，选择难度后立即开始游戏
                                if auto_start_enabled:
                                    start_melody_puzzle_directly()
                                break 
                    elif current_state == PUZZLE_1:
                        for sprite in puzzle1_sprites:
                            if sprite.rect.collidepoint(event.pos):
                                sprite.interact()
                    elif current_state == PUZZLE_MELODY:
                        for note_sprite in note_elements: 
                            if note_sprite.rect.collidepoint(event.pos):
                                note_sprite.interact() 

            elif event.type == pygame.KEYDOWN:
                # 只有在非自动模式下才响应按键
                if not auto_start_enabled:
                    if current_state == MENU:
                        if event.key == pygame.K_p: 
                            start_melody_puzzle_directly()
                        elif event.key == pygame.K_g: 
                             print("Starting Green Box Puzzle from Menu...")
                             current_state = PUZZLE_1
                    elif current_state == PUZZLE_COMPLETE:
                        # Return to menu when puzzle is completed
                        current_state = MENU
                    elif current_state == PUZZLE_MELODY:
                         if event.key == pygame.K_m: 
                            # Return to menu
                            current_state = MENU
                         elif event.key == pygame.K_r:
                            # Reset current puzzle with the same difficulty
                            generate_new_melody_puzzle(current_difficulty)
                            player_score = 0
                            melody_puzzle_attempts = 0
    
        screen.fill(BLACK) 

        if current_state == MENU:
            if menu_background_image:
                screen.blit(menu_background_image, (0,0))
            else:
                # 使用渐变背景代替图片
                for i in range(SCREEN_HEIGHT):
                    # 创建从深蓝色到稍浅蓝色的渐变
                    color = (20, 20, max(40, min(40 + i // 3, 90)))
                    pygame.draw.line(screen, color, (0, i), (SCREEN_WIDTH, i))
            
            # 优化标题和按钮布局，确保文字适配
            title_text = font_large.render("Sound Alchemist's Chamber", True, WHITE)
            title_width = title_text.get_width()
            
            # 如果标题太宽，缩小字体
            if title_width > SCREEN_WIDTH - 40:
                scale_factor = (SCREEN_WIDTH - 40) / title_width
                new_size = int(74 * scale_factor)
                font_large_adjusted = pygame.font.Font(default_font_name, new_size)
                title_text = font_large_adjusted.render("Sound Alchemist's Chamber", True, WHITE)
            
            screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 80))
            
            difficulty_title_text = font_medium.render("Select Difficulty:", True, WHITE)
            screen.blit(difficulty_title_text, (SCREEN_WIDTH // 2 - difficulty_title_text.get_width() // 2, 180))

            for button in difficulty_buttons:
                button.update_hover(mouse_pos) 
                button.draw(screen, is_selected=(button.value == current_difficulty))
            
            puzzle_prompts_y_start = (difficulty_buttons[-1].rect.bottom + 40) if difficulty_buttons else 380
            play_prompt_melody = font_small.render("Press 'P' to attempt the Melody Puzzle", True, WHITE)
            screen.blit(play_prompt_melody, (SCREEN_WIDTH // 2 - play_prompt_melody.get_width() // 2, puzzle_prompts_y_start))

            play_prompt_green = font_small.render("Press 'G' for the Green Box (placeholder)", True, WHITE)
            screen.blit(play_prompt_green, (SCREEN_WIDTH // 2 - play_prompt_green.get_width() // 2, puzzle_prompts_y_start + 50))

        elif current_state == PLAYING: 
            screen.fill(WHITE)
            text = font_medium.render("Exploring the Chamber...", True, BLACK) 
            screen.blit(text, (50, 50))
            
        elif current_state == PUZZLE_1:
            screen.fill((50, 50, 50)) 
            text = font_medium.render("Puzzle 1: Click the Green Box", True, WHITE) 
            screen.blit(text, (50, 50))
            puzzle1_sprites.draw(screen) 
            
        elif current_state == PUZZLE_MELODY: 
            screen.fill((30, 30, 70)) 
            
            # 优化文字显示确保适配
            title_text = font_medium.render("The Alchemist's Melody", True, WHITE)
            if title_text.get_width() > SCREEN_WIDTH - 40:
                font_medium_adjusted = pygame.font.Font(default_font_name, 40)  # 缩小字体
                title_text = font_medium_adjusted.render("The Alchemist's Melody", True, WHITE)
                
            screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 30))

            difficulty_text_str = f"Difficulty: {DIFFICULTY_SETTINGS[current_difficulty]['name']}"
            difficulty_text_surf = font_small.render(difficulty_text_str, True, LIGHT_BLUE)
            screen.blit(difficulty_text_surf, (SCREEN_WIDTH // 2 - difficulty_text_surf.get_width() // 2, 80))

            instruction_text_str = "Click the colored blocks in the correct musical order"
            instruction_text = font_small.render(instruction_text_str, True, YELLOW_COLOR)
            
            # 如果文本太长，减小字体大小
            if instruction_text.get_width() > SCREEN_WIDTH - 40:
                max_width = SCREEN_WIDTH - 40
                adjusted_font_size = int(font_small.get_height() * (max_width / instruction_text.get_width()))
                adjusted_font = pygame.font.Font(default_font_name, adjusted_font_size)
                instruction_text = adjusted_font.render(instruction_text_str, True, YELLOW_COLOR)
                
            screen.blit(instruction_text, (SCREEN_WIDTH // 2 - instruction_text.get_width() // 2, 120))
            
            instruction_reset = font_tiny.render("Press 'R' to reset puzzle", True, WHITE)
            screen.blit(instruction_reset, (SCREEN_WIDTH // 2 - instruction_reset.get_width() // 2, 150))
            
            instruction_exit = font_tiny.render("Press 'M' to return to menu", True, WHITE) 
            screen.blit(instruction_exit, (SCREEN_WIDTH // 2 - instruction_exit.get_width() // 2, SCREEN_HEIGHT - 30))

            note_elements.draw(screen) 

            feedback_y_pos = SCREEN_HEIGHT - 70
            input_display_parts = []
            for pid in player_melody_input:
                if "distractor" in pid:
                    input_display_parts.append("X") 
                else:
                    input_display_parts.append(NOTE_DISPLAY_NAMES.get(pid, "?"))
            
            input_text_str = "Your input: " + " - ".join(input_display_parts) 
            
            # 如果文本太长，减小字体大小
            input_text_render = font_small.render(input_text_str, True, WHITE)
            if input_text_render.get_width() > SCREEN_WIDTH - 40:
                max_width = SCREEN_WIDTH - 40
                adjusted_font_size = int(font_small.get_height() * (max_width / input_text_render.get_width()))
                adjusted_font = pygame.font.Font(default_font_name, adjusted_font_size)
                input_text_render = adjusted_font.render(input_text_str, True, WHITE)
                
            screen.blit(input_text_render, (SCREEN_WIDTH // 2 - input_text_render.get_width() // 2, feedback_y_pos))

            attempts_text_str = f"Mistakes: {melody_puzzle_attempts}"
            attempts_text_surf = font_tiny.render(attempts_text_str, True, LIGHT_RED)
            screen.blit(attempts_text_surf, (20, SCREEN_HEIGHT - 30))

        elif current_state == PUZZLE_COMPLETE:
            screen.fill((20, 80, 20)) 
            
            main_message = ""
            if last_puzzle_solved == PUZZLE_MELODY:
                main_message = "Melody Puzzle Solved!" 
            elif last_puzzle_solved == PUZZLE_1:
                 main_message = "Green Box Puzzle Solved!" 
            else:
                main_message = "Puzzle Complete!" 

            # 确保文字适配
            main_message_text = font_large.render(main_message, True, WHITE)
            if main_message_text.get_width() > SCREEN_WIDTH - 40:
                adjusted_font_size = int(font_large.get_height() * ((SCREEN_WIDTH - 40) / main_message_text.get_width()))
                adjusted_font = pygame.font.Font(default_font_name, adjusted_font_size)
                main_message_text = adjusted_font.render(main_message, True, WHITE)
                
            screen.blit(main_message_text, (SCREEN_WIDTH // 2 - main_message_text.get_width() // 2, 
                                             SCREEN_HEIGHT // 2 - main_message_text.get_height() // 2 - 70))
            
            sub_text_str = "Well Done, Alchemist!" 
            if last_puzzle_solved == PUZZLE_MELODY:
                score_text_str = f"Score: {player_score}"
                score_surf = font_medium.render(score_text_str, True, YELLOW_COLOR)
                screen.blit(score_surf, (SCREEN_WIDTH // 2 - score_surf.get_width() // 2, SCREEN_HEIGHT // 2 + 0))

                rating = ""
                if player_score >= DIFFICULTY_SETTINGS[current_difficulty]["score_multiplier"] * 900 : 
                    rating = "Excellent!"
                elif player_score >= DIFFICULTY_SETTINGS[current_difficulty]["score_multiplier"] * 600: 
                    rating = "Great Job!"
                elif player_score > 0:
                    rating = "Good Effort!"
                else:
                    rating = "Keep Practicing!"
                
                rating_surf = font_small.render(rating, True, WHITE)
                screen.blit(rating_surf, (SCREEN_WIDTH // 2 - rating_surf.get_width() // 2, SCREEN_HEIGHT // 2 + 50))
                sub_text_str = "You have a keen ear!"

            sub_text = font_medium.render(sub_text_str, True, YELLOW_COLOR)
            screen.blit(sub_text, (SCREEN_WIDTH // 2 - sub_text.get_width() // 2, SCREEN_HEIGHT // 2 + 90))

            text_continue = font_small.render("Press any key to return to the Menu.", True, WHITE) 
            screen.blit(text_continue, (SCREEN_WIDTH // 2 - text_continue.get_width() // 2, SCREEN_HEIGHT - 100))
            
        all_game_sprites.update() 
        particles_group.update()  
        particles_group.draw(screen) 

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()