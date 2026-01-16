"""
Device-Aware Agent Configuration
Optimized for edge devices: Pi3, Pi5, older laptops
Minimal memory footprint, local-first operation
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any
import os
import psutil


class DeviceProfile(Enum):
    """Hardware profiles for edge operation"""
    PI3 = "pi3"              # 512MB RAM, ARMv7
    PI5 = "pi5"              # 8GB RAM, ARM64
    PI5_CLUSTER = "pi5_cluster"  # Multiple Pi5s
    LAPTOP_OLD = "laptop_old"    # 4-8GB RAM, i5
    LAPTOP_NEW = "laptop_new"    # 16GB+ RAM, M1/i7+
    DESKTOP = "desktop"         # Server-class hardware


class ModelSize(Enum):
    """Model size for device capability"""
    TINY = "tiny"           # TinyLLaMA 1.1B (600MB q8)
    SMALL = "small"         # Phi-2 2.7B (1.6GB q4)
    MEDIUM = "medium"       # Mistral 7B (4GB q4)
    LARGE = "large"         # Llama 2 13B (8GB q4)


@dataclass
class DeviceConfig:
    """Configuration for specific device capabilities"""
    name: str
    profile: DeviceProfile
    model_size: ModelSize
    
    # Memory constraints
    max_model_memory_mb: int
    max_inference_memory_mb: int
    
    # Performance expectations
    tokens_per_sec: int
    max_context_length: int
    batch_size: int
    
    # Hardware capabilities
    supports_training: bool
    supports_fine_tuning: bool
    can_run_background_tasks: bool
    
    # Storage (MB)
    available_storage_mb: int
    
    # Optimization flags
    use_quantization: bool
    quantization_bits: int  # 4 or 8
    use_streaming_inference: bool
    enable_cache_compression: bool
    
    # API usage (for setup/updates only)
    allow_api_calls_during_setup: bool = True
    allow_api_calls_after_setup: bool = False


# DEVICE PROFILES

PI3_CONFIG = DeviceConfig(
    name="Raspberry Pi 3",
    profile=DeviceProfile.PI3,
    model_size=ModelSize.TINY,
    max_model_memory_mb=400,
    max_inference_memory_mb=100,
    tokens_per_sec=2,
    max_context_length=512,
    batch_size=1,
    supports_training=False,
    supports_fine_tuning=False,
    can_run_background_tasks=False,
    available_storage_mb=32000,  # 32GB SD card
    use_quantization=True,
    quantization_bits=8,
    use_streaming_inference=True,
    enable_cache_compression=True,
)

PI5_CONFIG = DeviceConfig(
    name="Raspberry Pi 5",
    profile=DeviceProfile.PI5,
    model_size=ModelSize.SMALL,
    max_model_memory_mb=2000,
    max_inference_memory_mb=500,
    tokens_per_sec=6,
    max_context_length=2048,
    batch_size=2,
    supports_training=False,
    supports_fine_tuning=True,  # Slow but possible
    can_run_background_tasks=True,
    available_storage_mb=128000,  # 128GB SD card
    use_quantization=True,
    quantization_bits=4,
    use_streaming_inference=True,
    enable_cache_compression=True,
)

LAPTOP_OLD_CONFIG = DeviceConfig(
    name="Older Laptop",
    profile=DeviceProfile.LAPTOP_OLD,
    model_size=ModelSize.MEDIUM,
    max_model_memory_mb=3500,
    max_inference_memory_mb=1000,
    tokens_per_sec=12,
    max_context_length=4096,
    batch_size=4,
    supports_training=False,
    supports_fine_tuning=True,
    can_run_background_tasks=True,
    available_storage_mb=256000,  # 256GB typical
    use_quantization=True,
    quantization_bits=4,
    use_streaming_inference=True,
    enable_cache_compression=True,
)

LAPTOP_NEW_CONFIG = DeviceConfig(
    name="Modern Laptop",
    profile=DeviceProfile.LAPTOP_NEW,
    model_size=ModelSize.MEDIUM,
    max_model_memory_mb=6000,
    max_inference_memory_mb=2000,
    tokens_per_sec=25,
    max_context_length=4096,
    batch_size=8,
    supports_training=True,
    supports_fine_tuning=True,
    can_run_background_tasks=True,
    available_storage_mb=512000,  # 512GB typical
    use_quantization=True,
    quantization_bits=4,
    use_streaming_inference=False,
    enable_cache_compression=False,
)

DESKTOP_CONFIG = DeviceConfig(
    name="Desktop/Server",
    profile=DeviceProfile.DESKTOP,
    model_size=ModelSize.LARGE,
    max_model_memory_mb=12000,
    max_inference_memory_mb=4000,
    tokens_per_sec=50,
    max_context_length=8192,
    batch_size=16,
    supports_training=True,
    supports_fine_tuning=True,
    can_run_background_tasks=True,
    available_storage_mb=1000000,  # 1TB typical
    use_quantization=False,
    quantization_bits=32,
    use_streaming_inference=False,
    enable_cache_compression=False,
)


class EdgeAgentConfig:
    """Master configuration for edge-optimized agent"""
    
    def __init__(self, device_profile: Optional[str] = None):
        """Initialize with auto-detection or specified profile"""
        
        if device_profile:
            self.device_config = self._get_device_config(device_profile)
        else:
            self.device_config = self._auto_detect_device()
        
        # Load device-specific settings
        self._apply_device_settings()
    
    def _auto_detect_device(self) -> DeviceConfig:
        """Detect device capabilities and select best config"""
        
        # Get system info
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # Try to detect Pi
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().strip()
                if "Pi 5" in model:
                    return PI5_CONFIG
                elif "Pi 3" in model:
                    return PI3_CONFIG
        except FileNotFoundError:
            pass
        
        # Detect by RAM
        if total_memory_gb < 1:
            return PI3_CONFIG
        elif total_memory_gb < 4:
            return LAPTOP_OLD_CONFIG
        elif total_memory_gb < 8:
            return LAPTOP_OLD_CONFIG
        elif total_memory_gb < 16:
            return LAPTOP_NEW_CONFIG
        else:
            return DESKTOP_CONFIG
    
    def _get_device_config(self, profile: str) -> DeviceConfig:
        """Get config for named profile"""
        configs = {
            "pi3": PI3_CONFIG,
            "pi5": PI5_CONFIG,
            "laptop": LAPTOP_OLD_CONFIG,
            "laptop_prod": LAPTOP_NEW_CONFIG,
            "desktop": DESKTOP_CONFIG,
        }
        return configs.get(profile.lower(), LAPTOP_OLD_CONFIG)
    
    def _apply_device_settings(self):
        """Apply device-specific configuration"""
        
        dc = self.device_config
        
        # Model selection
        self.MODEL_SELECTION = {
            ModelSize.TINY: "TinyLLaMA 1.1B",
            ModelSize.SMALL: "Phi-2 2.7B",
            ModelSize.MEDIUM: "Mistral 7B",
            ModelSize.LARGE: "Llama 2 13B",
        }[dc.model_size]
        
        # Model path and quantization
        if dc.use_quantization:
            self.MODEL_QUANT = f"q{dc.quantization_bits}"
            self.MODEL_FORMAT = "gguf"  # GGML format for quantized
        else:
            self.MODEL_QUANT = "f32"
            self.MODEL_FORMAT = "safetensors"
        
        # Inference settings
        self.MAX_TOKENS = dc.max_context_length
        self.BATCH_SIZE = dc.batch_size
        self.STREAM_OUTPUT = dc.use_streaming_inference
        
        # Cache settings
        self.CACHE_TYPE = "compressed_sqlite" if dc.enable_cache_compression else "sqlite"
        self.CACHE_MAX_SIZE_mb = 50 if dc.profile == DeviceProfile.PI3 else 200
        
        # API restrictions (local-first philosophy)
        self.ALLOW_API_CALLS_SETUP = dc.allow_api_calls_during_setup
        self.ALLOW_API_CALLS_PRODUCTION = dc.allow_api_calls_after_setup
        self.API_CALL_INTERVAL_HOURS = 24 * 7  # Weekly
        
        # Feature enablement
        self.ENABLE_WEB_SYNC = True  # Fetch updates
        self.ENABLE_FINE_TUNING = dc.supports_fine_tuning
        self.ENABLE_TRAINING = dc.supports_training
        self.ENABLE_BACKGROUND_TASKS = dc.can_run_background_tasks
        
        # Knowledge base settings
        self.KNOWLEDGE_BASE_TYPE = "embedded_onnx"  # No external DB
        self.KNOWLEDGE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Tiny, efficient
        self.KNOWLEDGE_SEARCH_TOP_K = 3  # Fewer results for fast search
        
        # Device-specific optimization
        self.DEVICE_PROFILE = dc.profile.value
        self.TOKENS_PER_SEC_EXPECTED = dc.tokens_per_sec
        self.MAX_INFERENCE_MEMORY_MB = dc.max_inference_memory_mb
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary"""
        return {
            "device": self.device_config.profile.value,
            "model": self.MODEL_SELECTION,
            "quantization": self.MODEL_QUANT,
            "max_tokens": self.MAX_TOKENS,
            "batch_size": self.BATCH_SIZE,
            "stream_output": self.STREAM_OUTPUT,
            "api_calls_allowed": self.ALLOW_API_CALLS_PRODUCTION,
            "fine_tuning_enabled": self.ENABLE_FINE_TUNING,
            "training_enabled": self.ENABLE_TRAINING,
            "cache_type": self.CACHE_TYPE,
        }
    
    def print_summary(self):
        """Print device configuration summary"""
        dc = self.device_config
        print(f"""
╔════════════════════════════════════════════════════════════╗
║          EDGE AGENT CONFIGURATION                          ║
╚════════════════════════════════════════════════════════════╝

Device:        {dc.name}
Model:         {self.MODEL_SELECTION} ({self.MODEL_QUANT})
Format:        {self.MODEL_FORMAT}

MEMORY BUDGET:
  Model:       {dc.max_model_memory_mb} MB
  Inference:   {dc.max_inference_memory_mb} MB
  Total:       {dc.max_model_memory_mb + dc.max_inference_memory_mb} MB

PERFORMANCE:
  Tokens/sec:  {dc.tokens_per_sec}
  Context:     {dc.max_context_length} tokens
  Batch size:  {dc.batch_size}

CAPABILITIES:
  Fine-tuning: {dc.supports_fine_tuning}
  Training:    {dc.supports_training}
  Background:  {dc.can_run_background_tasks}

OPTIMIZATION:
  Quantized:   {dc.use_quantization} ({dc.quantization_bits}-bit)
  Streaming:   {dc.use_streaming_inference}
  Compression: {dc.enable_cache_compression}

OPERATION MODE:
  API (setup):      {self.ALLOW_API_CALLS_SETUP}
  API (production): {self.ALLOW_API_CALLS_PRODUCTION}
  Sync interval:    {self.API_CALL_INTERVAL_HOURS}h
  Cache type:       {self.CACHE_TYPE}

STORAGE:
  Available:   {dc.available_storage_mb} MB
  Model:       ~{int(dc.max_model_memory_mb * 1.5)} MB
  Knowledge:   ~100 MB
  Cache:       ~{self.CACHE_MAX_SIZE_mb} MB
        """)


# Auto-detect and create default config
config = EdgeAgentConfig()

if __name__ == "__main__":
    config.print_summary()
    print("\nConfig Dictionary:")
    print(config.to_dict())
