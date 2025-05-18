#!/usr/bin/env python3
"""Log management utility for maintaining and archiving logs.

This script:
1. Rotates logs based on size and age
2. Archives old logs
3. Generates log statistics and reports
4. Cleans up old log files
"""

import os
import sys
import shutil
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from src.utils.logging_utils import get_logger

logger = get_logger('log_manager', 'maintenance')

class LogManager:
    """Manages log files and directories."""
    
    def __init__(self, base_log_dir: str = "logs"):
        """Initialize the log manager.
        
        Args:
            base_log_dir: Base directory for logs
        """
        self.base_dir = Path(base_log_dir)
        self.categories = [
            'linting', 'testing', 'performance',
            'maintenance', 'conversion', 'validation'
        ]
        
    def rotate_logs(self, max_size: int = 200, max_age_days: int = 30):
        """Rotate logs based on size and age.
        
        Args:
            max_size: Maximum lines per log file
            max_age_days: Maximum age of log files in days
        """
        logger.info(f"Rotating logs (max_size={max_size}, max_age_days={max_age_days})")
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
        
        for category in self.categories:
            category_dir = self.base_dir / category
            if not category_dir.exists():
                continue
                
            for date_dir in category_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                    
                try:
                    dir_date = datetime.datetime.strptime(date_dir.name, '%Y-%m-%d')
                    if dir_date < cutoff_date:
                        self._archive_logs(date_dir)
                        continue
                except ValueError:
                    continue
                
                self._rotate_dir_logs(date_dir, max_size)
    
    def _rotate_dir_logs(self, dir_path: Path, max_size: int):
        """Rotate logs in a specific directory."""
        for log_file in dir_path.glob('*.log'):
            with open(log_file) as f:
                lines = f.readlines()
            
            if len(lines) > max_size:
                # Split into chunks
                chunks = [lines[i:i + max_size] 
                         for i in range(0, len(lines), max_size)]
                
                # Write chunks to new files
                stem = log_file.stem.rsplit('_', 1)[0]
                for i, chunk in enumerate(chunks, 1):
                    new_file = log_file.parent / f"{stem}_{i:03d}.log"
                    with open(new_file, 'w') as f:
                        f.writelines(chunk)
                
                # Remove original file
                log_file.unlink()
    
    def _archive_logs(self, date_dir: Path):
        """Archive logs from a date directory."""
        archive_dir = self.base_dir / 'archive'
        archive_dir.mkdir(exist_ok=True)
        
        archive_name = f"{date_dir.parent.name}_{date_dir.name}.tar.gz"
        archive_path = archive_dir / archive_name
        
        if not archive_path.exists():
            logger.info(f"Archiving {date_dir} to {archive_path}")
            shutil.make_archive(
                str(archive_path.with_suffix('')),
                'gztar',
                root_dir=str(date_dir.parent),
                base_dir=date_dir.name
            )
        
        shutil.rmtree(date_dir)
    
    def generate_stats(self) -> Dict:
        """Generate statistics about log files.
        
        Returns:
            Dictionary containing log statistics
        """
        stats = {
            'total_size': 0,
            'categories': {},
            'dates': {},
            'archived': {
                'count': 0,
                'size': 0
            }
        }
        
        # Current logs
        for category in self.categories:
            category_dir = self.base_dir / category
            if not category_dir.exists():
                continue
            
            cat_stats = {'size': 0, 'files': 0, 'dates': 0}
            for date_dir in category_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                
                dir_size = sum(f.stat().st_size for f in date_dir.glob('*.log'))
                cat_stats['size'] += dir_size
                cat_stats['files'] += len(list(date_dir.glob('*.log')))
                cat_stats['dates'] += 1
                
                stats['dates'][date_dir.name] = {
                    'size': dir_size,
                    'files': len(list(date_dir.glob('*.log')))
                }
            
            stats['categories'][category] = cat_stats
            stats['total_size'] += cat_stats['size']
        
        # Archived logs
        archive_dir = self.base_dir / 'archive'
        if archive_dir.exists():
            archives = list(archive_dir.glob('*.tar.gz'))
            stats['archived']['count'] = len(archives)
            stats['archived']['size'] = sum(f.stat().st_size for f in archives)
        
        return stats

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Log management utility')
    parser.add_argument('--rotate', action='store_true',
                       help='Rotate and archive logs')
    parser.add_argument('--stats', action='store_true',
                       help='Generate log statistics')
    parser.add_argument('--max-size', type=int, default=200,
                       help='Maximum lines per log file')
    parser.add_argument('--max-age', type=int, default=30,
                       help='Maximum age of log files in days')
    
    args = parser.parse_args()
    manager = LogManager()
    
    if args.rotate:
        manager.rotate_logs(args.max_size, args.max_age)
    
    if args.stats:
        stats = manager.generate_stats()
        print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
