import argparse

from classpulse.cv import PrivacyPreservingAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Run the ClassPulse privacy-preserving classroom CV pipeline."
    )
    parser.add_argument(
        "--backend-url",
        default="http://127.0.0.1:5000/api/telemetry",
        help="Telemetry backend endpoint",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index to use",
    )
    args = parser.parse_args()

    analyzer = PrivacyPreservingAnalyzer(
        backend_url=args.backend_url,
    )
    analyzer.run(camera_index=args.camera_index)


if __name__ == "__main__":
    main()
