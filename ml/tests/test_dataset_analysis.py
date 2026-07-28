"""
Unit tests for Dataset Analysis Module.
"""

import os
import csv
import shutil
import tempfile
import unittest
import numpy as np

from ml.analysis.dataset_analyzer import DatasetAnalyzer
from ml.analysis.visualizer import DatasetVisualizer
from ml.analysis.report_generator import ReportGenerator


class TestDatasetAnalysis(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "metadata.csv")
        self.output_dir = os.path.join(self.temp_dir, "reports")

        # Create sample CSV
        fieldnames = [
            'video_id', 'gesture', 'subject', 'video_path', 'duration',
            'frame_count', 'fps', 'resolution', 'quality',
            'lighting_condition', 'background', 'created_at'
        ]
        
        rows = [
            {'video_id': 'V01', 'gesture': 'HELLO', 'subject': 'S1', 'video_path': 'V01.mp4', 'duration': '2.0', 'frame_count': '60', 'fps': '30.0', 'resolution': '1920x1080', 'quality': 'high', 'lighting_condition': 'normal', 'background': 'indoor', 'created_at': '2026-01-01'},
            {'video_id': 'V02', 'gesture': 'HELLO', 'subject': 'S2', 'video_path': 'V02.mp4', 'duration': '3.0', 'frame_count': '90', 'fps': '30.0', 'resolution': '1920x1080', 'quality': 'high', 'lighting_condition': 'normal', 'background': 'indoor', 'created_at': '2026-01-01'},
            {'video_id': 'V03', 'gesture': 'NAMASTE', 'subject': 'S1', 'video_path': 'V03.mp4', 'duration': '1.5', 'frame_count': '45', 'fps': '30.0', 'resolution': '1280x720', 'quality': 'medium', 'lighting_condition': 'normal', 'background': 'indoor', 'created_at': '2026-01-01'},
        ]

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_analyzer_class_distribution(self):
        analyzer = DatasetAnalyzer(metadata_csv=self.csv_path)
        cls_metrics = analyzer.analyze_class_distribution()

        self.assertEqual(cls_metrics.total_classes, 2)
        self.assertEqual(cls_metrics.total_samples, 3)
        self.assertEqual(cls_metrics.class_counts['HELLO'], 2)
        self.assertEqual(cls_metrics.class_counts['NAMASTE'], 1)
        self.assertEqual(cls_metrics.max_samples_per_class, 2)
        self.assertEqual(cls_metrics.min_samples_per_class, 1)

    def test_analyzer_durations_and_frames(self):
        analyzer = DatasetAnalyzer(metadata_csv=self.csv_path)
        dur_metrics = analyzer.analyze_video_durations()
        frame_metrics = analyzer.analyze_frame_counts()

        self.assertAlmostEqual(dur_metrics.mean_duration, 2.1666, places=2)
        self.assertEqual(frame_metrics.total_frames, 195)
        self.assertAlmostEqual(frame_metrics.mean_frames, 65.0, places=1)

    def test_report_generator_and_visualizer(self):
        analyzer = DatasetAnalyzer(metadata_csv=self.csv_path)
        cls_metrics = analyzer.analyze_class_distribution()
        dur_metrics = analyzer.analyze_video_durations()
        frame_metrics = analyzer.analyze_frame_counts()
        missing_metrics = analyzer.analyze_missing_landmarks()
        quality_metrics = analyzer.generate_quality_report()

        reporter = ReportGenerator(self.output_dir)
        visualizer = DatasetVisualizer(self.output_dir)

        # Export CSVs
        path_cls = reporter.export_class_distribution_csv(cls_metrics)
        path_imb = reporter.export_class_imbalance_summary_csv(cls_metrics)
        path_dur = reporter.export_duration_stats_csv(dur_metrics)
        path_frm = reporter.export_frame_count_stats_csv(frame_metrics)
        path_mis = reporter.export_missing_landmarks_summary_csv(missing_metrics)
        path_qlt = reporter.export_dataset_quality_report_csv(quality_metrics)
        path_md = reporter.generate_markdown_report(cls_metrics, dur_metrics, frame_metrics, missing_metrics, quality_metrics)

        self.assertTrue(os.path.exists(path_cls))
        self.assertTrue(os.path.exists(path_imb))
        self.assertTrue(os.path.exists(path_dur))
        self.assertTrue(os.path.exists(path_frm))
        self.assertTrue(os.path.exists(path_mis))
        self.assertTrue(os.path.exists(path_qlt))
        self.assertTrue(os.path.exists(path_md))

        # Generate plots
        p1 = visualizer.plot_class_distribution(cls_metrics)
        p2 = visualizer.plot_class_imbalance(cls_metrics)
        p3 = visualizer.plot_missing_landmarks(missing_metrics)

        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))
        self.assertTrue(os.path.exists(p3))


if __name__ == '__main__':
    unittest.main()
