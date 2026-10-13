import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from .config import ROOT, Settings


def main():
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description='Local-model manhwa chapter cleaner')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare-models')
    commands.add_parser('doctor')
    clean = commands.add_parser('clean')
    clean.add_argument('image', type=Path)
    clean.add_argument('--output', type=Path, required=True)
    clean.add_argument('--mode', choices=['dialogue', 'all-text'], default='dialogue')
    clean.add_argument('--mask', type=Path, help='Optional reviewed binary text mask')
    commands.add_parser('bot')
    purge = commands.add_parser('purge')
    purge.add_argument('--days', type=int, default=7)
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.command == 'prepare-models':
        from .models import prepare
        prepare(settings.models)
        print('Model checksums verified')
    elif args.command == 'doctor':
        from .models import WEIGHTS, digest
        import os
        ready = {name: (settings.models / name).exists() and digest(settings.models / name, algorithm) == expected
                 for name, (_, algorithm, expected) in WEIGHTS.items()}
        print(json.dumps({'models': ready, 'discord_token_configured': bool(os.getenv('DISCORD_BOT_TOKEN')),
                          'drive_configured': all(os.getenv(k) for k in ('GDRIVE_CLIENT_ID', 'GDRIVE_CLIENT_SECRET', 'GDRIVE_REFRESH_TOKEN')),
                          'data_directory': str(settings.data)}, indent=2))
    elif args.command == 'clean':
        from .models import Detector, Inpainter
        from .pipeline import Pipeline
        import numpy as np
        from PIL import Image
        mask = np.asarray(Image.open(args.mask).convert('L')) if args.mask else None
        pipeline = Pipeline(settings, Detector(settings), Inpainter(settings))
        report = pipeline.clean(args.image.read_bytes(), args.output, args.mode, manual_mask=mask)
        print(json.dumps({k: v for k, v in report.items() if k not in {'boxes', 'after_boxes'}}, indent=2))
    elif args.command == 'bot':
        from .bot import run
        run(settings)
    elif args.command == 'purge':
        if args.days < 1: parser.error('Retention must be at least one day')
        from .jobs import Store
        print(f'Removed {Store(settings).expire(args.days)} expired jobs')

if __name__ == '__main__':
    main()
