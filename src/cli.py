#!/usr/bin/env python3
"""
Command Line Interface for Fraud Detection System

This module provides a command-line interface for the fraud detection system,
allowing users to interact with all four models through simple commands.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.fraud_detection_system import FraudDetectionSystem


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Fraud Detection System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s website-fraud "http://example.com"
  %(prog)s mobile-risk "This app is great!" --metadata '{"rating": 4.5, "downloads": 1000}'
  %(prog)s apk-malware --features-file features.csv
  %(prog)s vision-brand --features-file image_features.csv
  %(prog)s status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Website fraud detection
    website_parser = subparsers.add_parser('website-fraud', help='Detect website fraud')
    website_parser.add_argument('url', help='URL to analyze')
    website_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    
    # Mobile risk assessment
    mobile_parser = subparsers.add_parser('mobile-risk', help='Assess mobile app risk')
    mobile_parser.add_argument('reviews', help='App reviews text')
    mobile_parser.add_argument('--metadata', '-m', help='PlayStore metadata as JSON string')
    mobile_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    
    # APK malware detection
    apk_parser = subparsers.add_parser('apk-malware', help='Detect APK malware')
    apk_parser.add_argument('--features-file', '-f', required=True, help='CSV file with APK features')
    apk_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    
    # Vision brand detection
    vision_parser = subparsers.add_parser('vision-brand', help='Detect brand impersonation')
    vision_parser.add_argument('--features-file', '-f', required=True, help='CSV file with image features')
    vision_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    
    # System status
    status_parser = subparsers.add_parser('status', help='Get system status')
    status_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Initialize the fraud detection system
        print("Initializing Fraud Detection System...")
        fraud_system = FraudDetectionSystem()
        
        if args.command == 'website-fraud':
            result = fraud_system.predict_website_fraud(args.url)
            print_result("Website Fraud Detection", result, args.output)
            
        elif args.command == 'mobile-risk':
            metadata = {}
            if args.metadata:
                try:
                    metadata = json.loads(args.metadata)
                except json.JSONDecodeError:
                    print("Error: Invalid JSON in metadata argument")
                    return 1
            
            result = fraud_system.predict_mobile_risk(args.reviews, metadata)
            print_result("Mobile Risk Assessment", result, args.output)
            
        elif args.command == 'apk-malware':
            import pandas as pd
            try:
                features_df = pd.read_csv(args.features_file)
                if features_df.empty:
                    print("Error: Features file is empty")
                    return 1
                
                features = features_df.iloc[0].to_numpy()
                result = fraud_system.predict_apk_malware(features)
                print_result("APK Malware Detection", result, args.output)
            except Exception as e:
                print(f"Error reading features file: {e}")
                return 1
                
        elif args.command == 'vision-brand':
            import pandas as pd
            try:
                features_df = pd.read_csv(args.features_file)
                if features_df.empty:
                    print("Error: Features file is empty")
                    return 1
                
                features = features_df.iloc[0].to_numpy()
                result = fraud_system.predict_vision_brand(features)
                print_result("Vision Brand Detection", result, args.output)
            except Exception as e:
                print(f"Error reading features file: {e}")
                return 1
                
        elif args.command == 'status':
            result = fraud_system.get_system_status()
            print_result("System Status", result, args.output)
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


def print_result(title: str, result: Dict[str, Any], output_file: str = None):
    """Print results in a formatted way and optionally save to file."""
    print(f"\n{title}")
    print("=" * len(title))
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    # Print results in a clean format
    for key, value in result.items():
        if key == 'model_info':
            continue
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Print model info if available
    if 'model_info' in result:
        print("\nModel Information:")
        for key, value in result['model_info'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Save to file if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")


if __name__ == '__main__':
    sys.exit(main())
