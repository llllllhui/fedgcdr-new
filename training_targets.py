"""Helpers for running one or more target-domain training directions."""


def resolve_target_domains(target_domain, domain_names):
    if target_domain == -1:
        return list(range(len(domain_names)))

    if 0 <= target_domain < len(domain_names):
        return [target_domain]

    raise ValueError(
        f"target_domain must be -1 or in [0, {len(domain_names) - 1}], got {target_domain}"
    )


def replace_or_add_cli_arg(argv, option, value):
    updated = []
    i = 0
    replaced = False

    while i < len(argv):
        token = argv[i]
        if token == option:
            updated.extend([option, value])
            replaced = True
            i += 2
            continue
        updated.append(token)
        i += 1

    if not replaced:
        updated.extend([option, value])

    return updated
