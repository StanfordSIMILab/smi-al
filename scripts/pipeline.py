import argparse, yaml
from smi_al.active_loop import run_phase_a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    run_phase_a(cfg)

if __name__ == '__main__':
    main()
