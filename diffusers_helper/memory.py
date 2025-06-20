# By lllyasviel


import torch
import gc


cpu = torch.device('cpu')
gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
gpu_complete_modules = []

# Add a global registry to track model states and prevent disk reloading
_model_registry = {}
_model_preserved_states = {}

# Smart model management system optimized for container runtime
class SmartModelManager:
    def __init__(self):
        self.models = {}
        self.current_gpu_models = set()
        self.vram_mode = 'low'  # 'low', 'mid', 'high'
        self.persistent_gpu_models = set()  # Models that should stay on GPU in mid-vram mode
        self.free_vram_gb = 0
    
    def register_model(self, model, name):
        """Register a model for smart management"""
        self.models[name] = model
        preserve_model_in_memory(model, name)
        print(f'Registered model: {name}')
    
    def set_vram_mode(self, free_vram_gb):
        """Set VRAM mode based on available memory"""
        self.free_vram_gb = free_vram_gb
        if free_vram_gb > 40:
            self.vram_mode = 'high'
            print(f'High VRAM mode: {free_vram_gb}GB - keeping all models on GPU')
        elif 20 <= free_vram_gb <= 40:
            self.vram_mode = 'mid'
            # In mid-VRAM mode (24GB), keep smaller models on GPU permanently
            self.persistent_gpu_models = {'text_encoder_2', 'vae', 'image_encoder'}
            print(f'Mid VRAM mode: {free_vram_gb}GB - keeping small models on GPU, smart swap large models')
            print(f'Persistent GPU models: {self.persistent_gpu_models}')
        else:
            self.vram_mode = 'low'
            print(f'Low VRAM mode: {free_vram_gb}GB - aggressive memory management')
    
    def initialize_persistent_models(self, target_device=None):
        """Initialize persistent models on GPU for mid-VRAM mode"""
        if target_device is None:
            target_device = gpu
        
        if self.vram_mode == 'mid':
            print('Initializing persistent models on GPU for mid-VRAM mode...')
            for name in self.persistent_gpu_models:
                if name in self.models:
                    self.models[name].to(target_device)
                    self.current_gpu_models.add(name)
                    print(f'Moved {name} to GPU (persistent)')
            torch.cuda.empty_cache()
    
    def load_for_inference(self, model_names, target_device=None, preserved_memory_gb=2):
        """Container-optimized loading: efficient for 24GB GPU + 32GB RAM"""
        if target_device is None:
            target_device = gpu
        
        if self.vram_mode == 'high':
            # High VRAM: Keep everything on GPU
            for name in model_names:
                if name in self.models and name not in self.current_gpu_models:
                    self.models[name].to(target_device)
                    self.current_gpu_models.add(name)
        
        elif self.vram_mode == 'mid':
            # Mid VRAM (24GB): Smart management optimized for container
            # Persistent models stay on GPU, only swap transformer
            for name in model_names:
                if name in self.models and name not in self.current_gpu_models:
                    if name in self.persistent_gpu_models:
                        # This should already be on GPU, but ensure it
                        if name not in self.current_gpu_models:
                            self.models[name].to(target_device)
                            self.current_gpu_models.add(name)
                    else:
                        # For non-persistent models (mainly transformer), check memory
                        current_free = get_cuda_free_memory_gb(target_device)
                        if current_free > preserved_memory_gb:
                            # Move non-persistent models to CPU first to make room
                            for gpu_model in list(self.current_gpu_models):
                                if gpu_model not in self.persistent_gpu_models and gpu_model not in model_names:
                                    move_model_to_cpu_preserve_memory(self.models[gpu_model], gpu_model)
                                    self.current_gpu_models.discard(gpu_model)
                            
                            # Now load the requested model
                            self.models[name].to(target_device)
                            self.current_gpu_models.add(name)
                        else:
                            print(f'Not enough memory to load {name} (need {preserved_memory_gb}GB, have {current_free}GB)')
        
        else:  # low VRAM
            # Low VRAM: Aggressive swapping like before
            for name in list(self.current_gpu_models):
                if name not in model_names:
                    move_model_to_cpu_preserve_memory(self.models[name], name)
                    self.current_gpu_models.discard(name)
            
            for name in model_names:
                if name in self.models and name not in self.current_gpu_models:
                    if get_cuda_free_memory_gb(target_device) > preserved_memory_gb:
                        self.models[name].to(target_device)
                        self.current_gpu_models.add(name)
                    else:
                        print(f'Not enough memory to load {name}, keeping on CPU')
        
        torch.cuda.empty_cache()
    
    def unload_all_but_preserve(self):
        """Move all models to CPU but preserve in memory"""
        for name in list(self.current_gpu_models):
            move_model_to_cpu_preserve_memory(self.models[name], name)
        self.current_gpu_models.clear()
        torch.cuda.empty_cache()
    
    def get_model(self, name):
        """Get a model by name"""
        return self.models.get(name)
    
    def get_memory_status(self):
        """Get current memory status for debugging"""
        return {
            'vram_mode': self.vram_mode,
            'free_vram_gb': self.free_vram_gb,
            'current_gpu_models': self.current_gpu_models,
            'persistent_gpu_models': self.persistent_gpu_models,
            'available_vram_gb': get_cuda_free_memory_gb(gpu)
        }

# Global smart model manager instance
smart_manager = SmartModelManager()

class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return


def preserve_model_in_memory(model: torch.nn.Module, model_name: str):
    """Preserve model state in memory to prevent disk reloading"""
    global _model_preserved_states
    
    if model_name not in _model_preserved_states:
        print(f'Preserving {model_name} state in memory to prevent disk reloading')
        # Create a lightweight state preservation - just keep the model reference alive
        _model_preserved_states[model_name] = {
            'model_ref': model,
            'preserved': True
        }


def move_model_to_cpu_preserve_memory(model: torch.nn.Module, model_name: str = None):
    """Move model to CPU while preserving it in memory to prevent disk reloading"""
    if model_name:
        preserve_model_in_memory(model, model_name)
    
    # Move to CPU but keep model object alive
    model.to(device=cpu)
    return model


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    """Fixed version that properly moves the entire model, not just one layer"""
    print(f'Moving {model.__class__.__name__} to {target_device} (fake_diffusers_current_device)')
    
    # Special handling for models with scale_shift_table
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
    
    # Move the entire model, not just the first layer
    model.to(target_device)
    
    # Ensure all parameters are on the target device
    for name, param in model.named_parameters():
        if param.device != target_device:
            param.data = param.data.to(target_device)
    
    return


def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024 ** 3)


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    # Check if we have enough memory before starting
    if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
        torch.cuda.empty_cache()
        print(f'Insufficient memory, aborting move of {model.__class__.__name__}')
        return

    # Move the entire model at once instead of module by module
    model.to(device=target_device)
    torch.cuda.empty_cache()
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    # Check if we need to offload
    if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
        torch.cuda.empty_cache()
        print(f'Sufficient memory available, skipping offload of {model.__class__.__name__}')
        return

    # Preserve model in memory before moving to CPU
    model_name = f"{model.__class__.__name__}_{id(model)}"
    move_model_to_cpu_preserve_memory(model, model_name)
    torch.cuda.empty_cache()
    return


def unload_complete_models(*args):
    """Improved version that preserves models in memory to prevent disk reloading"""
    for m in gpu_complete_modules + list(args):
        model_name = f"{m.__class__.__name__}_{id(m)}"
        move_model_to_cpu_preserve_memory(m, model_name)
        print(f'Unloaded {m.__class__.__name__} as complete (preserved in memory).')

    gpu_complete_modules.clear()
    torch.cuda.empty_cache()
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return


def cleanup_preserved_states():
    """Call this periodically to clean up preserved states if needed"""
    global _model_preserved_states
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'Cleaned up {len(_model_preserved_states)} preserved model states')
    _model_preserved_states.clear()


def register_models_for_smart_management(text_encoder=None, text_encoder_2=None, vae=None, image_encoder=None, transformer=None, free_vram_gb=0):
    """Register all models for smart management - optimized for container runtime"""
    smart_manager.set_vram_mode(free_vram_gb)
    
    if text_encoder is not None:
        smart_manager.register_model(text_encoder, 'text_encoder')
    if text_encoder_2 is not None:
        smart_manager.register_model(text_encoder_2, 'text_encoder_2')
    if vae is not None:
        smart_manager.register_model(vae, 'vae')
    if image_encoder is not None:
        smart_manager.register_model(image_encoder, 'image_encoder')
    if transformer is not None:
        smart_manager.register_model(transformer, 'transformer')
    
    print('All models registered for smart management')
    
    # Initialize persistent models for mid-VRAM mode (24GB GPU)
    smart_manager.initialize_persistent_models()


def smart_load_for_text_encoding(preserved_memory_gb=2):
    """Smart loading for text encoding phase - container optimized"""
    smart_manager.load_for_inference(['text_encoder', 'text_encoder_2'], preserved_memory_gb=preserved_memory_gb)


def smart_load_for_vae_encoding(preserved_memory_gb=2):
    """Smart loading for VAE encoding phase - container optimized"""
    smart_manager.load_for_inference(['vae'], preserved_memory_gb=preserved_memory_gb)


def smart_load_for_image_encoding(preserved_memory_gb=2):
    """Smart loading for image encoding phase - container optimized"""
    smart_manager.load_for_inference(['image_encoder'], preserved_memory_gb=preserved_memory_gb)


def smart_load_for_sampling(preserved_memory_gb=2):
    """Smart loading for sampling phase - container optimized"""
    smart_manager.load_for_inference(['transformer'], preserved_memory_gb=preserved_memory_gb)


def smart_load_for_vae_decoding(preserved_memory_gb=2):
    """Smart loading for VAE decoding phase - container optimized"""
    smart_manager.load_for_inference(['vae'], preserved_memory_gb=preserved_memory_gb)


def smart_load_for_multi_step(model_names, preserved_memory_gb=2):
    """Load multiple models for combined operations - container optimized"""
    smart_manager.load_for_inference(model_names, preserved_memory_gb=preserved_memory_gb)


def get_smart_manager_status():
    """Get current smart manager status for debugging"""
    return smart_manager.get_memory_status()


def smart_unload_all():
    """Unload all models but preserve in memory"""
    smart_manager.unload_all_but_preserve()
