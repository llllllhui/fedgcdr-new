import json
import os

from domain_config import AMAZON_PRESET_DOMAIN_NAMES


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_amazon_domain_names(dataset_dir, num_domain):
    metadata_path = os.path.join(dataset_dir, "domain_names.json")
    if os.path.exists(metadata_path):
        return _load_json(metadata_path)

    if num_domain in AMAZON_PRESET_DOMAIN_NAMES:
        return list(AMAZON_PRESET_DOMAIN_NAMES[num_domain])

    raise FileNotFoundError(
        f"Missing domain metadata: {metadata_path}. "
        f"Please regenerate the dataset or add domain_names.json."
    )


def set_dataset(args):
    if args.dataset == "amazon":
        dataset_dir = os.path.join(PROJECT_ROOT, "data", f"{args.num_domain}domains")
        domain_user = _load_json(os.path.join(dataset_dir, "domain_user.json"))
        dic = _load_json(os.path.join(dataset_dir, "implicit.json"))
        domain_names = _load_amazon_domain_names(dataset_dir, args.num_domain)
        args.num_users = len(dic["user_dic"])
        return domain_user, dic, domain_names

    if args.dataset == "douban":
        domain_names = ["Book", "Movie", "Music"]
        dataset_dir = os.path.join(PROJECT_ROOT, "data", "douban_oldver")
        domain_user = _load_json(os.path.join(dataset_dir, "domain_user.json"))
        dic = _load_json(os.path.join(dataset_dir, "implicit.json"))
        args.num_users = len(dic["user_dic"])
        return domain_user, dic, domain_names

    raise ValueError(f"Unsupported dataset: {args.dataset}")
