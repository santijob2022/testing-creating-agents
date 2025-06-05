# base_store_interface.py

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseStore(ABC):
    @abstractmethod
    def put(self, namespace: Tuple[str, ...], key: str, value: Any) -> None:
        pass

    @abstractmethod
    def get(self, namespace: Tuple[str, ...], key: str) -> Any:
        pass

    @abstractmethod
    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        pass
