from export.musicxml_converter import MusicXMLConverter
import os
import sys
import unittest
import numpy as np

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../src')))


class TestMusicXMLConverter(unittest.TestCase):
    def setUp(self):
        """Initialize MusicXML converter for testing"""
        self.converter = MusicXMLConverter()

        # Sample symbol recognition results
        self.sample_symbols = [
            {
                'label': 'quarter_note',
                'confidence': 0.95,
                'raw_image': np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
                'position': {'x': 100, 'y': 200}
            },
            {
                'label': 'half_note',
                'confidence': 0.85,
                'raw_image': np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
                'position': {'x': 200, 'y': 250}
            },
            {
                'label': 'quarter_rest',
                'confidence': 0.65,
                'raw_image': np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
                'position': {'x': 300, 'y': 300}
            }
        ]

    def test_note_element_creation(self):
        """Test creation of individual note elements"""
        for symbol in self.sample_symbols:
            note_elem = self.converter._create_note_element(
                symbol_info=symbol,
                staff_context={'clef': 'treble'},
                advanced_processing={'source': 'test_generation'}
            )

            # Basic assertions
            self.assertIsNotNone(note_elem)
            self.assertEqual(note_elem.tag, 'note')

            # Check note type
            type_elem = note_elem.find('type')
            self.assertIsNotNone(type_elem)

            # Check duration
            duration_elem = note_elem.find('duration')
            self.assertIsNotNone(duration_elem)

            # Check confidence annotation for low confidence symbols
            if symbol['confidence'] < 0.7:
                annotation_elem = note_elem.find('annotation')
                self.assertIsNotNone(annotation_elem)

            # Check metadata
            metadata_elem = note_elem.find('metadata')
            if symbol['raw_image'] is not None:
                self.assertIsNotNone(metadata_elem)

    def test_full_conversion(self):
        """Test full conversion of symbol set to MusicXML"""
        # Simulate OMR processing results
        omr_results = {
            'symbols': self.sample_symbols,
            'metadata': {
                'title': 'Test Conversion',
                'composer': 'Test Composer'
            }
        }

        # Perform conversion
        output_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'exports',
            'test_conversion.mxl'
        )

        self.converter.convert(
            omr_results,
            output_path=output_path
        )

        # Verify file was created
        self.assertTrue(os.path.exists(output_path))

        # Optional: Basic XML validation could be added here

    def test_advanced_processing(self):
        """Test advanced processing configurations"""
        advanced_config = {
            'pitch_correction': True,
            'noise_reduction': 0.8,
            'symbol_enhancement': 'adaptive'
        }

        note_elem = self.converter._create_note_element(
            symbol_info=self.sample_symbols[0],
            advanced_processing=advanced_config
        )

        # Check for advanced processing elements
        for key in advanced_config:
            adv_elem = note_elem.find(f'advanced_{key}')
            self.assertIsNotNone(adv_elem)


def main():
    """Run tests and print debug information"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMusicXMLConverter)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Create a debug log
    debug_dir = os.path.join(os.path.dirname(
        __file__), '..', '..', 'debug_logs')
    os.makedirs(debug_dir, exist_ok=True)

    with open(os.path.join(debug_dir, 'musicxml_test_results.log'), 'w') as f:
        f.write(f"Tests Run: {result.testsRun}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Failures: {len(result.failures)}\n")


if __name__ == '__main__':
    main()
