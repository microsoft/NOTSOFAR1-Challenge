from typing import TypeVar, Type
import argparse
from dataclasses import dataclass, field

from omegaconf import OmegaConf


ConfT = TypeVar('ConfT')

def get_conf(yaml_path: str, conf_type: Type[ConfT]) -> ConfT:
    schema = OmegaConf.structured(conf_type)
    conf = OmegaConf.load(yaml_path)
    merged = OmegaConf.merge(schema, conf)  # this will override schema with values from conf
    return OmegaConf.to_object(merged)


def _demo():
    @dataclass
    class CssConf:
        lr: float = 0.001
        epochs: int = 100

    @dataclass
    class Conf:
        css: CssConf = field(default_factory=CssConf)

    parser = argparse.ArgumentParser()
    parser.add_argument('--verb', choices=['show', 'write-default'], default='show')
    parser.add_argument('--yaml_path', default='../configs/conf_demo.yaml')
    args = parser.parse_args()

    if args.verb == 'show':
        c: Conf = get_conf(args.yaml_path, Conf)
        print(c)

    elif args.verb == 'write-default':
        schema = OmegaConf.structured(Conf)
        OmegaConf.save(config=schema, f=args.yaml_path)
        print(f'Default config was written to {args.yaml_path}')

    else:
        raise ValueError(f'Unknown verb: {args.verb}')


if __name__ == '__main__':
    _demo()
