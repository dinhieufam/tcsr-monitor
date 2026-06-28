"""Simple name→class registry for Hydra instantiation."""

from __future__ import annotations

from typing import Any, Type


_REGISTRY: dict[str, dict[str, Type]] = {}


def register(group: str, name: str):
    def decorator(cls: Type) -> Type:
        _REGISTRY.setdefault(group, {})[name] = cls
        return cls
    return decorator


def lookup(group: str, name: str) -> Type:
    if group not in _REGISTRY or name not in _REGISTRY[group]:
        available = list(_REGISTRY.get(group, {}).keys())
        raise KeyError(f"Unknown {group!r}: {name!r}. Available: {available}")
    return _REGISTRY[group][name]


def instantiate(group: str, name: str, **kwargs: Any) -> Any:
    cls = lookup(group, name)
    return cls(**kwargs)
