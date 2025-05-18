import os
import yaml
import logging
from typing import Dict, Any, Optional
from ..core.omr_exceptions import ConfigurationError

from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar, Union
import yaml
import os

T = TypeVar("T")


class ConfigManager:
    """
    Advanced configuration management for OMR Pipeline

    Supports multiple configuration sources:
    1. Default configuration (via dataclasses)
    2. YAML configuration file
    3. Environment variables
    4. Runtime overrides

    Features:
    - Type-safe configuration via dataclasses
    - Automatic validation of config values
    - Environment variable overrides
    - Nested configuration support
    - Configuration persistence
    """

    def __init__(
        self,
        config_class: Type[T],
        config_path: Optional[
            Union[str, Path]
        ] = "/ml/models/resources/tf-deep-omr/config/config.yml",
        env_prefix: str = "SMM",
    ) -> None:
        """
        Initialize configuration manager with type-safe configuration.

        Args:
            config_class: Dataclass type for configuration
            config_path: Optional path to YAML configuration file
            env_prefix: Prefix for environment variables
        """
        self.logger = logging.getLogger(__name__)

        if not is_dataclass(config_class):
            raise ValueError(f"{config_class.__name__} must be a dataclass")

        # Ensure config path is absolute
        if config_path:
            config_path = str(Path(config_path).resolve())

        # Store configuration class
        self._config_class = config_class
        self._env_prefix = env_prefix

        # Initialize with defaults from dataclass
        self._config = config_class()

        # Load configuration file if provided
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                self._load_yaml_config(config_path)
            else:
                self.logger.warning(f"Config file not found: {config_path}")

        # Load environment variables
        self._load_env_vars()

        # Validate final configuration
        self._validate_config()

    def _load_yaml_config(self, config_path: Path) -> None:
        """Load configuration from YAML file with type validation.

        Args:
            config_path: Path to YAML configuration file
        """
        try:
            yaml_config = yaml.safe_load(config_path.read_text())
            if yaml_config:
                # Validate against dataclass fields
                for key, value in yaml_config.items():
                    if hasattr(self._config_class, key):
                        if is_dataclass(value):
                            self._config.__setattr__(key, self._config_class(**value))
                        else:
                            self._config.__setattr__(key, value)
                    else:
                        self.logger.warning(f"Unknown config key: {key}")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML configuration: {e}", context={"config_path": config_path}
            )

    def _load_env_vars(self) -> None:
        """Load configuration from environment variables with type conversion."""
        prefix = f"{self._env_prefix}_"

        for key, field_type in self._config_class.__annotations__.items():
            env_key = f"{prefix}{key.upper()}"
            if env_key in os.environ:
                try:
                    # Convert environment variable to correct type
                    value = os.environ[env_key]
                    if field_type == bool:
                        value = value.lower() in ("true", "1", "yes")
                    elif field_type in (int, float):
                        value = field_type(value)
                    elif field_type == tuple:
                        value = tuple(eval(value))  # Safe for numeric tuples

                    self._config[key] = value
                except Exception as e:
                    self.logger.error(
                        f"Failed to convert {env_key}={value} to {field_type}: {e}"
                    )

    def _validate_config(self) -> None:
        """Validate configuration against dataclass fields."""
        config_dict = asdict(self._config)
        for key, field_type in self._config_class.__annotations__.items():
            if key not in config_dict:
                self.logger.warning(f"Missing config key: {key}")
                continue

            value = config_dict[key]
            if is_dataclass(field_type):
                # For nested dataclasses, validate all fields
                for nested_key, nested_type in field_type.__annotations__.items():
                    if nested_key not in value:
                        self.logger.warning(
                            f"Missing nested config key: {key}.{nested_key}"
                        )
                        continue

                    # Handle nested tuple types
                    if (
                        hasattr(nested_type, "__origin__")
                        and nested_type.__origin__ is tuple
                    ):
                        if not isinstance(value[nested_key], tuple):
                            self.logger.warning(
                                f"Type mismatch for {key}.{nested_key}: expected tuple, got {type(value[nested_key])}"
                            )
                        else:
                            if len(value[nested_key]) != len(nested_type.__args__):
                                self.logger.warning(
                                    f"Type mismatch for {key}.{nested_key}: expected tuple of length {len(nested_type.__args__)}, got {len(value[nested_key])}"
                                )
                            else:
                                for i, (elem, expected_type) in enumerate(
                                    zip(value[nested_key], nested_type.__args__)
                                ):
                                    if not isinstance(elem, expected_type):
                                        self.logger.warning(
                                            f"Type mismatch for {key}.{nested_key}[{i}]: expected {expected_type}, got {type(elem)}"
                                        )
                    else:
                        if not isinstance(value[nested_key], nested_type):
                            self.logger.warning(
                                f"Type mismatch for {key}.{nested_key}: expected {nested_type}, got {type(value[nested_key])}"
                            )
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is tuple:
                # Handle Tuple types
                if not isinstance(value, tuple):
                    self.logger.warning(
                        f"Type mismatch for {key}: expected tuple, got {type(value)}"
                    )
                else:
                    # Check tuple length and element types
                    if len(value) != len(field_type.__args__):
                        self.logger.warning(
                            f"Type mismatch for {key}: expected tuple of length {len(field_type.__args__)}, got {len(value)}"
                        )
                    else:
                        for i, (elem, expected_type) in enumerate(
                            zip(value, field_type.__args__)
                        ):
                            if not isinstance(elem, expected_type):
                                self.logger.warning(
                                    f"Type mismatch for {key}[{i}]: expected {expected_type}, got {type(elem)}"
                                )
            else:
                if not isinstance(value, field_type):
                    self.logger.warning(
                        f"Type mismatch for {key}: expected {field_type}, got {type(value)}"
                    )

    def _deep_merge(
        self, base: Dict[str, Any], update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries

        Args:
            base (Dict[str, Any]): Base configuration
            update (Dict[str, Any]): Configuration to merge

        Returns:
            Merged configuration dictionary
        """
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = base.get(key, {})
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value with type safety.

        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found

        Returns:
            Configuration value of correct type
        """
        try:
            value = self._config
            for key in key_path.split("."):
                value = value[key]

            # Validate type if possible
            if hasattr(self._config_class, key_path):
                expected_type = self._config_class.__annotations__[key_path]
                if not isinstance(value, expected_type):
                    self.logger.warning(
                        f"Type mismatch for {key_path}: expected {expected_type}, got {type(value)}"
                    )

            return value
        except (KeyError, TypeError):
            return default

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with type validation.

        Args:
            updates: Configuration updates to apply
        """
        # If updates is a dataclass instance, convert to dict
        if is_dataclass(updates):
            updates = asdict(updates)

        # Validate updates against dataclass fields
        validated_updates = {}
        for key, value in updates.items():
            if hasattr(self._config_class, key):
                field_type = self._config_class.__annotations__[key]

                # Handle nested dataclass fields
                if is_dataclass(field_type) and isinstance(value, dict):
                    value = field_type(**value)

                if isinstance(value, field_type):
                    validated_updates[key] = (
                        asdict(value) if is_dataclass(value) else value
                    )
                else:
                    self.logger.warning(
                        f"Type mismatch for {key}: expected {field_type}, got {type(value)}"
                    )
            else:
                self.logger.warning(f"Unknown config key: {key}")

        self._config = self._deep_merge(self._config, validated_updates)

    def get_config(self) -> T:
        """
        Get the current configuration as a dataclass instance.

        Returns:
            Dataclass instance containing the current configuration
        """
        return self._config

    def generate_config_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive configuration report

        Returns:
            Dict[str, Any]: Configuration report with validation and insights
        """
        import time

        # Validate key configuration parameters
        validation_results = {
            "preprocessing": {
                "min_image_size_valid": all(
                    isinstance(x, int) and x > 0
                    for x in self.get("preprocessing.min_image_size", (0, 0))
                ),
                "max_image_size_valid": all(
                    isinstance(x, int) and x > 0
                    for x in self.get("preprocessing.max_image_size", (0, 0))
                ),
                "color_mode_valid": self.get("preprocessing.color_mode")
                in ["rgb", "grayscale"],
            },
            "segmentation": {
                "min_symbol_size_valid": isinstance(
                    self.get("segmentation.min_symbol_size"), int
                )
                and self.get("segmentation.min_symbol_size") > 0,
                "max_symbol_size_valid": isinstance(
                    self.get("segmentation.max_symbol_size"), int
                )
                and self.get("segmentation.max_symbol_size") > 0,
            },
            "recognition": {
                "confidence_threshold_valid": 0
                <= self.get("recognition.confidence_threshold", 0)
                <= 1,
                "model_path_exists": os.path.exists(
                    self.get("recognition.model_path", "")
                ),
            },
        }

        # Compute overall configuration health
        config_health = all(
            all(val for val in category.values())
            for category in validation_results.values()
        )

        return {
            "timestamp": time.time(),
            "config_version": "1.0",
            "config_health": config_health,
            "validation_results": validation_results,
            "full_configuration": self.to_dict(),
        }
