from typing import TypeVar, Type, Dict
import argparse
from dataclasses import dataclass, field

from omegaconf import OmegaConf


ConfT = TypeVar('ConfT')

def load_yaml_to_dataclass(yaml_path: str, conf_type: Type[ConfT]) -> ConfT:
    """
    Load a YAML file and convert it to a dataclass object.

    Example:
        cfg: InferenceCfg = get_conf(conf_file, InferenceCfg)
    """
    schema = OmegaConf.structured(conf_type)
    conf = OmegaConf.load(yaml_path)
    merged = OmegaConf.merge(schema, conf)  # this will override schema with values from conf
    return OmegaConf.to_object(merged)


def update_dataclass(dataclass_obj: ConfT, updates: Dict) -> ConfT:
    """
    Update values in dataclass config using either dot-notation or brackets to denote sub-keys
    """
    schema = OmegaConf.structured(dataclass_obj)
    for k,v in updates.items():
        OmegaConf.update(schema, k, v)
    return OmegaConf.to_object(schema)


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
        c: Conf = load_yaml_to_dataclass(args.yaml_path, Conf)
        print(c)

    elif args.verb == 'write-default':
        schema = OmegaConf.structured(Conf)
        OmegaConf.save(config=schema, f=args.yaml_path)
        print(f'Default config was written to {args.yaml_path}')

    else:
        raise ValueError(f'Unknown verb: {args.verb}')


if __name__ == '__main__':
    _demo()
