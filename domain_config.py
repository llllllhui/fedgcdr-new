"""Shared domain configuration for Amazon cross-domain recommendation datasets."""

AMAZON_DOMAIN_CATALOG = {
    "Clothing": {
        "domain": "Clothing_Shoes_and_Jewelry",
        "core": 48,
    },
    "Books": {
        "domain": "Books",
        "core": 96,
    },
    "Home": {
        "domain": "Home_and_Kitchen",
        "core": 32,
    },
    "Electronics": {
        "domain": "Electronics",
        "core": 32,
    },
    "Sports": {
        "domain": "Sports_and_Outdoors",
        "core": 24,
    },
    "Cell": {
        "domain": "Cell_Phones_and_Accessories",
        "core": 16,
    },
    "Tools": {
        "domain": "Tools_and_Home_Improvement",
        "core": 16,
    },
    "CDs": {
        "domain": "CDs_and_Vinyl",
        "core": 24,
    },
    "Movies": {
        "domain": "Movies_and_TV",
        "core": 48,
    },
    "Toys": {
        "domain": "Toys_and_Games",
        "core": 32,
    },
    "Automotive": {
        "domain": "Automotive",
        "core": 32,
    },
    "Pet": {
        "domain": "Pet_Supplies",
        "core": 32,
    },
    "Kindle": {
        "domain": "Kindle_Store",
        "core": 48,
    },
    "Office": {
        "domain": "Office_Products",
        "core": 32,
    },
    "Patio": {
        "domain": "Patio_Lawn_and_Garden",
        "core": 32,
    },
    "Grocery": {
        "domain": "Grocery_and_Gourmet_Food",
        "core": 32,
    },
}

AMAZON_PRESET_DOMAIN_NAMES = {
    2: ["Books", "CDs"],
    4: ["Clothing", "Books", "Movies", "CDs"],
    8: ["Clothing", "Books", "Home", "Electronics", "Sports", "Cell", "Movies", "CDs"],
    16: [
        "Clothing",
        "Books",
        "Home",
        "Electronics",
        "Sports",
        "Cell",
        "Tools",
        "CDs",
        "Movies",
        "Toys",
        "Automotive",
        "Pet",
        "Kindle",
        "Office",
        "Patio",
        "Grocery",
    ],
}


def build_amazon_domain_config(num_domains, selected_domains=None):
    """Return domain filenames, user-core thresholds, and short names."""
    if selected_domains is None:
        if num_domains not in AMAZON_PRESET_DOMAIN_NAMES:
            raise ValueError(f"Unsupported Amazon domain count: {num_domains}")
        short_names = list(AMAZON_PRESET_DOMAIN_NAMES[num_domains])
    else:
        short_names = list(selected_domains)
        if len(short_names) != num_domains:
            raise ValueError(
                f"Expected {num_domains} domain names, got {len(short_names)}: {short_names}"
            )

    unknown = [name for name in short_names if name not in AMAZON_DOMAIN_CATALOG]
    if unknown:
        raise ValueError(
            f"Unknown Amazon domains: {unknown}. "
            f"Available domains: {sorted(AMAZON_DOMAIN_CATALOG)}"
        )

    if len(set(short_names)) != len(short_names):
        raise ValueError(f"Duplicate Amazon domains are not allowed: {short_names}")

    return {
        "domains": [AMAZON_DOMAIN_CATALOG[name]["domain"] for name in short_names],
        "cores": [AMAZON_DOMAIN_CATALOG[name]["core"] for name in short_names],
        "shorts": short_names,
    }
