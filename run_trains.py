import json
import sys

from trainers.trainer import get_trainer


DEFAULT_CONFIG = 'config/beta/mimic_sepsis_event.json'


def run_config(conf):
    print(conf)
    with open(conf) as f:
        config = json.load(f)
    config['cont'] = False
    trainer = get_trainer(config)
    trainer.train()


if __name__ == '__main__':
    configs = sys.argv[1:] or [DEFAULT_CONFIG]
    for conf in configs:
        run_config(conf)
