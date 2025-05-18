import os
from dataclasses import dataclass, asdict
import yaml
import logging


@dataclass
class OMRSettings:
    """Configuration for Optical Music Recognition"""

    pdf_dpi: int = 300
    preprocessing_kernel_size: tuple = (5, 5)
    staff_line_detection_threshold: float = 0.7
    symbol_recognition_confidence: float = 0.8


@dataclass
class ExportSettings:
    """Configuration for MusicXML Export"""

    output_directory: str = "exports"
    export_format: str = "musicxml"


@dataclass
class MLModelSettings:
    """Machine Learning Model Configuration"""

    symbol_recognition_model_path: str = (
        "resources/tf-deep-omr/Data/Models/symbol_recognition.h5"
    )
    training_data_dir: str = "resources/training_data/music_symbols"


class ConfigManager:
    def __init__(self, config_path=None):
        """
        Manage application configuration

        Args:
            config_path (str, optional): Path to custom configuration file
        """
        self.logger = logging.getLogger("config_manager")
        self.config_path = config_path or self._default_config_path()

        self.omr_settings = OMRSettings()
        self.export_settings = ExportSettings()
        self.ml_model_settings = MLModelSettings()

        self._load_config()

    @property
    def config(self):
        """
        Return the configuration as a dict for pipeline compatibility.
        """
        return {
            "omr": asdict(self.omr_settings),
            "export": asdict(self.export_settings),
            "ml_models": asdict(self.ml_model_settings),
        }

    def _default_config_path(self):
        """Generate default configuration path"""
        return os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "config.yml"
        )

    def _load_config(self):
        """Load configuration from YAML file"""
        is_default_path = self.config_path == self._default_config_path()
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                if config_data:  # Ensure config_data is not None or empty
                    self._update_settings(config_data)
                    self.logger.info(f"Loaded configuration from {self.config_path}")
                else:
                    # If explicitly provided config file is empty or invalid YAML
                    if not is_default_path:
                        self.logger.error(
                            f"Empty or invalid YAML in specified config file: {self.config_path}"
                        )
                        raise ConfigurationError(
                            f"Empty or invalid YAML in specified config file: {self.config_path}",
                            context={"config_path": self.config_path},
                        )
                    else:
                        self.logger.warning(
                            f"Empty or invalid YAML in default config file: {self.config_path}. Using defaults."
                        )
            else:
                # If config_path was explicitly provided and file doesn't exist
                if not is_default_path:
                    self.logger.error(
                        f"Specified config file not found: {self.config_path}"
                    )
                    raise ConfigurationError(
                        f"Specified config file not found: {self.config_path}",
                        context={"config_path": self.config_path},
                    )
                else:
                    self.logger.warning(
                        f"Default config file not found at {self.config_path}. Using defaults."
                    )
        except (
            ConfigurationError
        ):  # Re-raise ConfigurationError to be caught by OMRPipeline
            raise
        except Exception as e:
            self.logger.error(
                f"Configuration load error: {e} for path {self.config_path}"
            )
            # If it's an explicitly provided path that caused a general error, also raise ConfigurationError
            if not is_default_path:
                raise ConfigurationError(
                    f"Error loading specified config file: {e}",
                    context={"config_path": self.config_path, "original_error": str(e)},
                )
            # For default path errors, it might be less critical to halt everything,
            # but consistency could argue for raising ConfigurationError here too.
            # For now, let OMRPipeline's own checks handle issues with default/empty configs.

    def _update_settings(self, config):
        """Update settings from loaded configuration"""
        if "omr" in config:
            for key, value in config["omr"].items():
                setattr(self.omr_settings, key, value)

        if "export" in config:
            for key, value in config["export"].items():
                setattr(self.export_settings, key, value)

        if "ml_models" in config:
            for key, value in config["ml_models"].items():
                setattr(self.ml_model_settings, key, value)

    def save_config(self, path=None):
        """
        Save current configuration to YAML file

        Args:
            path (str, optional): Path to save configuration
        """
        save_path = path or self.config_path
        config = {
            "omr": asdict(self.omr_settings),
            "export": asdict(self.export_settings),
            "ml_models": asdict(self.ml_model_settings),
        }

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            self.logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Configuration save error: {e}")


def main():
    config_manager = ConfigManager()

    # Example of accessing settings
    print("PDF DPI:", config_manager.omr_settings.pdf_dpi)
    print("Export Directory:", config_manager.export_settings.output_directory)

    # Save current configuration
    config_manager.save_config()


if __name__ == "__main__":
    main()
