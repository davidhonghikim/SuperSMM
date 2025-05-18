import logging
import os
import time
import datetime
import traceback
import json
import cv2
import numpy as np
import psutil
import multiprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from src.core.omr_pipeline import OMRPipeline
from src.logging_config import log_performance, setup_logging
from src.performance.performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_diagnostics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PerformanceDiagnostics:
    def __init__(self, config_path=None):
        self.logger = logging.getLogger(__name__)
        self.pipeline = OMRPipeline(config_path=config_path)

        # Performance tracking
        self.performance_metrics = {
            'total_processing_time': 0.0,
            'total_memory_usage': 0.0,
            'pages_processed': 0,
            'errors_encountered': 0
        }

    @PerformanceMonitor.track_performance
    def analyze_image(self, image_path: str) -> dict:
        """Analyze single image performance

        Args:
            image_path (str): Path to image file

        Returns:
            dict: Processing results
        """
        try:
            result = self.pipeline.process_sheet_music(image_path)

            # Update performance metrics
            self.performance_metrics['total_processing_time'] += result['metadata'].get(
                'processing_time', 0.0)
            self.performance_metrics['pages_processed'] += 1

            return result
        except Exception as e:
            self.logger.error(f"Failed to process image {image_path}: {e}")
            self.performance_metrics['errors_encountered'] += 1
            return {}

    def parallel_performance_test(self, image_paths: list) -> list:
        """Run performance tests in parallel

        Args:
            image_paths (list): List of image file paths

        Returns:
            list: Processing results
        """
        with multiprocessing.Pool() as pool:
            results = [
                r for r in pool.map(
                    self.analyze_image,
                    image_paths) if r]

        return results

    def generate_performance_report(self) -> dict:
        """Generate comprehensive performance report

        Returns:
            dict: Performance metrics
        """
        metrics = self.performance_metrics.copy()
        metrics['average_processing_time'] = (
            metrics['total_processing_time'] / metrics['pages_processed']
            if metrics['pages_processed'] > 0 else 0.0
        )
        metrics['system_config'] = PerformanceMonitor.adaptive_resource_allocation()

        return metrics

    def advanced_performance_report(
            self, test_directory: str) -> Dict[str, Any]:
        """
        Generate an advanced performance report with detailed insights

        Args:
            test_directory (str): Directory containing test images

        Returns:
            Dict[str, Any]: Comprehensive performance report
        """
        import datetime

        # Run comprehensive test
        test_results = self.run_comprehensive_test(test_directory)

        # Performance metrics
        performance_metrics = self.generate_performance_report()

        # System resource analysis
        system_resources = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'total_memory_gb': psutil.virtual_memory().total / (1024 ** 3),
            'available_memory_gb': psutil.virtual_memory().available / (1024 ** 3)
        }

        # TensorFlow and GPU details
        tf_details = {
            'tf_version': tf.__version__,
            'cuda_available': tf.test.is_built_with_cuda(),
            'gpu_devices': [
                device.name for device in tf.config.list_physical_devices('GPU')]}

        # Analyze test results
        result_analysis = {
            'total_images_processed': len(test_results),
            'successful_processing_rate': len(
                [
                    r for r in test_results if r]) /
            len(test_results) if test_results else 0,
            'processing_times': [
                r['metadata']['processing_time'] for r in test_results if 'metadata' in r and 'processing_time' in r['metadata']]}

        # Compute advanced statistics
        if result_analysis['processing_times']:
            result_analysis.update(
                {
                    'min_processing_time': min(
                        result_analysis['processing_times']), 'max_processing_time': max(
                        result_analysis['processing_times']), 'avg_processing_time': sum(
                        result_analysis['processing_times']) / len(
                        result_analysis['processing_times'])})

        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'performance_metrics': performance_metrics,
            'system_resources': system_resources,
            'tensorflow_details': tf_details,
            'result_analysis': result_analysis,
            'test_directory': test_directory
        }

    def run_comprehensive_test(self, test_directory: str) -> list:
        """Run comprehensive performance test on all images in directory

        Args:
            test_directory (str): Directory containing test images

        Returns:
            list: Processing results
        """
        # Validate test directory
        if not os.path.isdir(test_directory):
            raise ValueError(f"Invalid test directory: {test_directory}")

        # Find all image files
        image_paths = [
            os.path.join(test_directory, f)
            for f in os.listdir(test_directory)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.pdf'))
        ]

        if not image_paths:
            self.logger.warning(
                f"No images found in test directory: {test_directory}")
            return []

        # Parallel processing
        return self.parallel_performance_test(image_paths)

    def visualize_performance_metrics(
            self, report: Dict[str, Any], output_dir: str = None) -> Dict[str, str]:
        """
        Visualize performance metrics from advanced performance report

        Args:
            report (Dict[str, Any]): Advanced performance report
            output_dir (str, optional): Directory to save visualization outputs

        Returns:
            Dict[str, str]: Paths to generated visualization files
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        # Set default output directory
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(__file__),
                'outputs',
                'performance_viz')
        os.makedirs(output_dir, exist_ok=True)

        # Visualization outputs
        viz_outputs = {}

        # 1. Processing Time Distribution
        if report.get('result_analysis', {}).get('processing_times'):
            plt.figure(figsize=(10, 6))
            sns.histplot(report['result_analysis']
                         ['processing_times'], kde=True)
            plt.title('Processing Time Distribution')
            plt.xlabel('Processing Time (seconds)')
            plt.ylabel('Frequency')
            processing_time_path = os.path.join(
                output_dir, 'processing_time_distribution.png')
            plt.savefig(processing_time_path)
            plt.close()
            viz_outputs['processing_time_distribution'] = processing_time_path

        # 2. System Resource Utilization
        if 'system_resources' in report:
            resources = report['system_resources']
            plt.figure(figsize=(10, 6))
            plt.bar(resources.keys(), resources.values())
            plt.title('System Resource Utilization')
            plt.xlabel('Resource')
            plt.ylabel('Percentage')
            plt.xticks(rotation=45)
            resource_util_path = os.path.join(
                output_dir, 'system_resource_utilization.png')
            plt.savefig(resource_util_path)
            plt.close()
            viz_outputs['system_resource_utilization'] = resource_util_path

        # 3. Performance Metrics Radar Chart
        metrics = report.get('performance_metrics', {})
        metrics_to_plot = {
            'Total Processing Time': metrics.get(
                'total_processing_time', 0), 'Pages Processed': metrics.get(
                'pages_processed', 0), 'Average Processing Time': metrics.get(
                'average_processing_time', 0), 'Errors Encountered': metrics.get(
                    'errors_encountered', 0)}

        plt.figure(figsize=(8, 8))
        plt.subplot(polar=True)
        angles = [
            n /
            float(
                len(metrics_to_plot)) *
            2 *
            np.pi for n in range(
                len(metrics_to_plot))]
        angles += angles[:1]

        values = list(metrics_to_plot.values())
        values += values[:1]

        plt.polar(angles[:-1], values[:-1], 'o-', linewidth=2)
        plt.fill(angles, values, alpha=0.25)
        plt.xticks(angles[:-1], list(metrics_to_plot.keys()))
        plt.title('Performance Metrics Overview')
        performance_metrics_path = os.path.join(
            output_dir, 'performance_metrics_radar.png')
        plt.savefig(performance_metrics_path)
        plt.close()
        viz_outputs['performance_metrics_radar'] = performance_metrics_path

        # 4. Generate JSON summary
        import json
        summary_path = os.path.join(
            output_dir, 'performance_visualization_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': report.get('timestamp'),
                'visualizations': viz_outputs,
                'metrics_summary': metrics_to_plot
            }, f, indent=4)

        viz_outputs['summary'] = summary_path

        return viz_outputs


def main():
    logger = logging.getLogger(__name__)

    try:
        # Test image directory
        test_directory = os.path.join(
            os.path.dirname(__file__), 'tests', 'test_images')

        # Validate test directory
        if not os.path.exists(test_directory):
            logger.error(f"Test directory does not exist: {test_directory}")
            return None, None

        # Initialize diagnostics
        diagnostics = PerformanceDiagnostics()

        # Generate advanced performance report
        report = diagnostics.advanced_performance_report(test_directory)

        # Log performance report
        logger.info("Advanced Performance Diagnostics Report:")
        for section, details in report.items():
            logger.info(f"{section.upper()}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {details}")

        return report.get(
            'result_analysis', {}).get(
            'test_results', []), report

    except Exception as e:
        logger.critical(f"Critical error in performance diagnostics: {e}")
        logger.critical(traceback.format_exc())
        return None, None


if __name__ == '__main__':
    setup_logging()
    main()
