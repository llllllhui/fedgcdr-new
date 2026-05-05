import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


BOOKS_FILE = "Books.csv"
CDS_FILE = "CDs_and_Vinyl.csv"
DOMAIN_NAMES = ["Books", "CDs"]
LEGACY_USER_CORE = {"Books": 96, "CDs": 24}
LEGACY_ITEM_CORE = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Amazon-Dual with randomly sampled overlapping users."
    )
    parser.add_argument("--ratings_dir", type=str, default="ratings-full")
    parser.add_argument("--output_dir", type=str, default="data/amazon_dual_2500")
    parser.add_argument("--sample_size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="raw_overlap",
        choices=["raw_overlap", "legacy_filtered_overlap"],
        help="raw_overlap samples from raw overlapping users; legacy_filtered_overlap matches the old project's core filtering first.",
    )
    parser.add_argument("--num_negatives", type=int, default=99)
    return parser.parse_args()


def load_ratings(csv_path):
    return pd.read_csv(
        csv_path,
        header=None,
        names=["item", "user", "rating", "timestamp"],
        usecols=[0, 1, 2, 3],
        dtype={
            "item": "string",
            "user": "string",
            "rating": "float32",
            "timestamp": "int64",
        },
    )


def filter_domain_by_legacy_core(df, user_core, item_core):
    user_counts = df.groupby("user").size()
    keep_users = set(user_counts[user_counts >= user_core].index)

    item_counts = df.groupby("item").size()
    keep_items = set(item_counts[item_counts >= item_core].index)

    return df[df["user"].isin(keep_users) & df["item"].isin(keep_items)].reset_index(drop=True)


def sample_overlap_users(users_a, users_b, sample_size, seed):
    overlap = sorted(set(users_a) & set(users_b))
    if len(overlap) < sample_size:
        raise ValueError(
            f"Only {len(overlap)} overlapping users are available, fewer than sample_size={sample_size}."
        )

    rng = random.Random(seed)
    sampled = rng.sample(overlap, sample_size)
    sampled.sort()
    return sampled


def select_dual_domain_frames(books_df, cds_df, sample_size, seed, sampling_mode="raw_overlap"):
    if sampling_mode == "legacy_filtered_overlap":
        books_df = filter_domain_by_legacy_core(
            books_df,
            user_core=LEGACY_USER_CORE["Books"],
            item_core=LEGACY_ITEM_CORE,
        )
        cds_df = filter_domain_by_legacy_core(
            cds_df,
            user_core=LEGACY_USER_CORE["CDs"],
            item_core=LEGACY_ITEM_CORE,
        )

    sampled_users = sample_overlap_users(books_df["user"], cds_df["user"], sample_size, seed)
    selected_books = books_df[books_df["user"].isin(sampled_users)].reset_index(drop=True)
    selected_cds = cds_df[cds_df["user"].isin(sampled_users)].reset_index(drop=True)
    return sampled_users, selected_books, selected_cds


def split_single_domain(df, sampled_users):
    user_order = {user_id: idx for idx, user_id in enumerate(sampled_users)}
    item_ids = sorted(df["item"].unique())
    item_order = {item_id: idx for idx, item_id in enumerate(item_ids)}

    encoded = df.copy()
    encoded["user_idx"] = encoded["user"].map(user_order)
    encoded["item_idx"] = encoded["item"].map(item_order)
    encoded = encoded.sort_values(["user_idx", "timestamp", "item_idx"]).reset_index(drop=True)

    train_rows = {}
    test_rows = {}
    for user_id in sampled_users:
        user_idx = user_order[user_id]
        user_data = encoded[encoded["user_idx"] == user_idx]
        if user_data.empty:
            raise ValueError(f"Sampled user {user_id} has no interactions after filtering.")

        test_row = user_data.iloc[-1]
        train_rows[user_idx] = user_data.iloc[:-1]["item_idx"].tolist()
        test_rows[user_idx] = int(test_row["item_idx"])

    return train_rows, test_rows, len(item_order)


def build_implicit_artifacts(domain_dfs, domain_names, sampled_users, num_negatives, seed):
    rng = np.random.default_rng(seed)
    sampled_users = list(sampled_users)

    client_train_data = [[[] for _ in range(len(domain_names))] for _ in range(len(sampled_users))]
    server_evaluate_data = [[] for _ in range(len(domain_names))]
    num_items = []
    num_users = []
    user_dic = [{name: idx for name in domain_names} for idx in range(len(sampled_users))]

    for domain_idx, df in enumerate(domain_dfs):
        train_rows, test_rows, item_count = split_single_domain(df, sampled_users)
        num_items.append(item_count)
        num_users.append(len(sampled_users))

        for user_idx in range(len(sampled_users)):
            train_items = train_rows[user_idx]
            test_item = test_rows[user_idx]
            observed = set(train_items + [test_item])
            negatives = []
            while len(negatives) < num_negatives:
                negative = int(rng.integers(0, item_count))
                if negative not in observed and negative not in negatives:
                    negatives.append(negative)

            client_train_data[user_idx][domain_idx] = train_items
            for item_id in negatives:
                server_evaluate_data[domain_idx].append([user_idx, item_id, 0])
            server_evaluate_data[domain_idx].append([user_idx, test_item, 1])

    domain_user = {name: list(range(len(sampled_users))) for name in domain_names}
    return {
        "client_train_data": client_train_data,
        "server_evaluate_data": server_evaluate_data,
        "num_items": num_items,
        "num_users": num_users,
        "user_dic": user_dic,
        "domain_user": domain_user,
    }


def save_outputs(output_dir, sampled_users, artifacts):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "domain_names.json", "w", encoding="utf-8") as f:
        json.dump(DOMAIN_NAMES, f)

    with open(output_path / "sampled_overlap_users.json", "w", encoding="utf-8") as f:
        json.dump(sampled_users, f)

    with open(output_path / "domain_user.json", "w", encoding="utf-8") as f:
        json.dump(artifacts["domain_user"], f)

    implicit = {
        "client_train_data": artifacts["client_train_data"],
        "server_evaluate_data": artifacts["server_evaluate_data"],
        "num_items": artifacts["num_items"],
        "num_users": artifacts["num_users"],
        "user_dic": artifacts["user_dic"],
    }
    with open(output_path / "implicit.json", "w", encoding="utf-8") as f:
        json.dump(implicit, f)


def main():
    args = parse_args()
    ratings_dir = Path(args.ratings_dir)

    print(f"Loading {ratings_dir / BOOKS_FILE} ...")
    books_df = load_ratings(ratings_dir / BOOKS_FILE)
    print(f"Loading {ratings_dir / CDS_FILE} ...")
    cds_df = load_ratings(ratings_dir / CDS_FILE)
    print(f"Sampling mode: {args.sampling_mode}")

    sampled_users, selected_books, selected_cds = select_dual_domain_frames(
        books_df,
        cds_df,
        sample_size=args.sample_size,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
    )

    artifacts = build_implicit_artifacts(
        [selected_books, selected_cds],
        DOMAIN_NAMES,
        sampled_users,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )
    save_outputs(args.output_dir, sampled_users, artifacts)

    print(f"Built Amazon-Dual at {args.output_dir}")
    print(f"sampled_users={len(sampled_users)}")
    print(f"books_items={artifacts['num_items'][0]}, cds_items={artifacts['num_items'][1]}")


if __name__ == "__main__":
    main()
