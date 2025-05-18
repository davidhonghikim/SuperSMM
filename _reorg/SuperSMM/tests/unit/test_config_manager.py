import os
import pytest
import yaml

from config.settings import ConfigManager


class TestConfigManager:
    def test_default_configuration(self):
        """Test default configuration initialization"""
        config_manager = ConfigManager()

        # Check default configuration keys
        assert config_manager.get('preprocessing.min_image_size') is not None
        assert config_manager.get('segmentation.min_symbol_size') is not None
        assert config_manager.get(
            'recognition.confidence_threshold') is not None

    def test_yaml_configuration_loading(self, config_path):
        """Test loading configuration from YAML file"""
        config_manager = ConfigManager(config_path=config_path)

        # Verify YAML configuration is loaded
        assert config_manager.get('preprocessing.color_mode') == 'rgb'
        assert config_manager.get(
            'segmentation.segmentation_strategy') == 'connected_components'

    def test_environment_variable_override(self, monkeypatch):
        """Test configuration override via environment variables"""
        monkeypatch.setenv('SMM_PREPROCESSING_COLOR_MODE', 'grayscale')

        config_manager = ConfigManager()

        # Verify environment variable overrides default
        assert config_manager.get('preprocessing.color_mode') == 'grayscale'

    def test_nested_configuration_access(self, config_path):
        """Test accessing nested configuration values"""
        config_manager = ConfigManager(config_path=config_path)

        # Test nested value access
        assert config_manager.get(
            'preprocessing.preprocessing_methods') is not None
        assert 'grayscale' in config_manager.get(
            'preprocessing.preprocessing_methods')

    def test_configuration_conversion(self):
        """Test value type conversion"""
        config_manager = ConfigManager()

        # Test various type conversions
        assert isinstance(config_manager.get(
            'segmentation.min_symbol_size'), int)
        assert isinstance(config_manager.get(
            'recognition.confidence_threshold'), float)
