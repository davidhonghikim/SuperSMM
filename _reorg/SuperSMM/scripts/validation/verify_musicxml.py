import os
import xml.etree.ElementTree as ET
import logging
import argparse # Added for command-line arguments
import zipfile

def verify_musicxml(musicxml_path):
    """Verify the generated MusicXML file"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Added format
    logger = logging.getLogger(__name__)

    if not os.path.exists(musicxml_path):
        logger.error(f"MusicXML file not found: {musicxml_path}")
        return False

    try:
        # Handle compressed .mxl file
        if musicxml_path.endswith('.mxl'):
            with zipfile.ZipFile(musicxml_path, 'r') as zip_ref:
                # Extract the first XML file (typically the main score)
                xml_files = [f for f in zip_ref.namelist() if f.endswith('.xml') and not f.startswith('META-INF/')]
                if not xml_files:
                    # Look for files in subdirectories if any, like 'score.xml'
                    xml_files = [f for f in zip_ref.namelist() if f.endswith('.xml') and '/' in f and not f.startswith('META-INF/')]
                    if not xml_files:
                        logger.error("No primary XML file found in .mxl archive (expected .xml not in META-INF)")
                        return False
                
                # Prefer specific names if multiple XMLs exist, e.g. score.xml, or the one not in META-INF/container.xml
                # For now, just take the first valid one found.
                xml_to_parse = xml_files[0]
                logger.info(f"Parsing {xml_to_parse} from MXL archive.")
                with zip_ref.open(xml_to_parse) as xml_file:
                    tree = ET.parse(xml_file)
        else:
            # Regular XML file
            tree = ET.parse(musicxml_path)
        
        root = tree.getroot()

        # Namespace for MusicXML (optional, as some files might not use explicit namespace)
        # Common MusicXML namespaces
        namespaces = {
            '': 'urn:iso:std:iso:15924:music:musicxml40', # Default for MusicXML 4.0
            'musicxml': 'http://www.musicxml.org/xsd/musicxml.xsd' # Older/generic
        }
        
        # Try to find elements with and without namespace prefixes
        parts = root.findall('.//part') # Common case with no explicit namespace in findall
        if not parts:
             parts = root.findall('.//{urn:iso:std:iso:15924:music:musicxml40}part')
        if not parts: # Fallback for older namespace
            parts = root.findall('.//musicxml:part', namespaces)
            
        measures = root.findall('.//measure')
        if not measures:
            measures = root.findall('.//{urn:iso:std:iso:15924:music:musicxml40}measure')
        if not measures:
            measures = root.findall('.//musicxml:measure', namespaces)

        notes = root.findall('.//note')
        if not notes:
            notes = root.findall('.//{urn:iso:std:iso:15924:music:musicxml40}note')
        if not notes:    
            notes = root.findall('.//musicxml:note', namespaces)

        logger.info(f"MusicXML Analysis for: {os.path.basename(musicxml_path)}")
        logger.info(f"  Total Parts: {len(parts)}")
        logger.info(f"  Total Measures: {len(measures)}")
        logger.info(f"  Total Notes: {len(notes)}")

        # Basic validation
        if len(parts) == 0:
            logger.warning("MusicXML validation issue: No <part> elements found.")
            return False
        if len(measures) == 0:
            logger.warning("MusicXML validation issue: No <measure> elements found.")
            return False
        if len(notes) == 0:
            logger.warning("MusicXML validation issue: No <note> elements found.")
            # Allowing this to pass as some simple scores might not have notes (e.g. percussion guides) - adjust if needed
            # return False 

        logger.info("MusicXML basic structure appears valid.")
        return True

    except zipfile.BadZipFile:
        logger.error(f"Invalid .mxl file (bad zip archive): {musicxml_path}")
        return False
    except ET.ParseError as e:
        logger.error(f"XML Parsing Error in {musicxml_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during verification of {musicxml_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify a MusicXML file.")
    parser.add_argument("musicxml_path", help="Path to the MusicXML file (.xml or .mxl)")
    args = parser.parse_args()

    is_valid = verify_musicxml(args.musicxml_path)
    print(f"MusicXML Validation for {os.path.basename(args.musicxml_path)}: {'PASSED' if is_valid else 'FAILED'}")

if __name__ == "__main__":
    main()
