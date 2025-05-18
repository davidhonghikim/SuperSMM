#!/usr/bin/env python3
import os
import sys
import subprocess
import autopep8
import logging
import ast
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('project_repair.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProjectRepair:
    def __init__(self, project_root):
        self.project_root = project_root
        self.errors = []
        self.repairs = []

    def find_python_files(self):
        """Find all Python files in the project"""
        python_files = []
        for root, _, files in os.walk(self.project_root):
            python_files.extend([
                os.path.join(root, file)
                for file in files
                if file.endswith('.py')
            ])
        return python_files

    def fix_indentation(self, file_path):
        """Automatically fix indentation using autopep8"""
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()

            # Fix indentation
            fixed_content = autopep8.fix_code(
                content,
                options={'aggressive': 1}
            )

            # Write back to file
            with open(file_path, 'w') as f:
                f.write(fixed_content)

            self.repairs.append(f"Fixed indentation in {file_path}")
            logger.info(f"✅ Indentation fixed: {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to fix indentation in {file_path}: {e}")
            self.errors.append(f"Indentation error in {file_path}: {e}")

    def validate_syntax(self, file_path):
        """Validate Python syntax"""
        try:
            with open(file_path, 'r') as f:
                ast.parse(f.read())
            logger.info(f"✅ Syntax valid: {file_path}")
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in {file_path}: {e}")
            self.errors.append(f"Syntax error in {file_path}: {e}")
            self.fix_indentation(file_path)

    def repair_project(self):
        """Comprehensive project repair"""
        logger.info("🔧 Starting Project Repair")

        # Find and repair Python files
        python_files = self.find_python_files()

        for file_path in python_files:
            self.validate_syntax(file_path)

        # Generate repair report
        report = self.generate_report()
        return report

    def generate_report(self):
        """Generate repair report"""
        report = {
            'total_files_checked': len(self.find_python_files()),
            'errors': self.errors or [],
            'repairs': self.repairs or [],
            'status': 'PASS' if not self.errors else 'FAIL'
        }

        # Write report
        with open('project_repair_report.json', 'w') as f:
            import json
            json.dump(report, f, indent=2)

        logger.info("📋 Repair Report Generated")
        return report


def main():
    """Main repair script entry point"""
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Initialize and run repair
    repair_tool = ProjectRepair(project_root)
    repair_result = repair_tool.repair_project()

    # Print report for debugging
    print(f"Repair Status: {repair_result['status']}")
    print(f"Errors: {repair_result['errors']}")
    print(f"Repairs: {repair_result['repairs']}")

    # Exit with appropriate status
    sys.exit(0 if repair_result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
