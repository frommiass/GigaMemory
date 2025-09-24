# src/submit/core/container.py

from typing import Dict, Any, Type, Optional
import threading

class DependencyContainer:
    """Контейнер для управления зависимостями"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self._services = {}
            self._singletons = {}
            self._factories = {}
            self.initialized = True
    
    def register_singleton(self, interface: Type, implementation: Any):
        """Регистрирует singleton сервис"""
        self._singletons[interface] = implementation
    
    def register_factory(self, interface: Type, factory: callable):
        """Регистрирует фабрику для создания сервиса"""
        self._factories[interface] = factory
    
    def get(self, interface: Type) -> Any:
        """Получает реализацию интерфейса"""
        # Проверяем singleton
        if interface in self._singletons:
            return self._singletons[interface]
        
        # Проверяем фабрику
        if interface in self._factories:
            instance = self._factories[interface]()
            self._singletons[interface] = instance
            return instance
        
        raise ValueError(f"No implementation registered for {interface}")
    
    def configure(self, config: Dict[str, Any]):
        """Конфигурирует контейнер"""
        for service_name, service_config in config.items():
            if 'interface' in service_config and 'implementation' in service_config:
                interface = service_config['interface']
                impl_class = service_config['implementation']
                params = service_config.get('params', {})
                
                # Создаем фабрику
                def factory(cls=impl_class, p=params):
                    return cls(**p)
                
                self.register_factory(interface, factory)

# Глобальный контейнер
container = DependencyContainer()