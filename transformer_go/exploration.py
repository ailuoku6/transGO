"""Runtime self-play profiles; deliberately excluded from checkpoint Config."""

PROFILES = ("katago", "legacy")


def validate_profile(profile):
    if profile not in PROFILES:
        raise ValueError(f"selfplay_policy must be one of {PROFILES}")


def selfplay_temperature(move_number, board_size, profile, legacy_moves):
    validate_profile(profile)
    if profile == "legacy":
        return 1.0 if move_number < legacy_moves else 0.1
    # KataGo selfplay1_maxsize9.cfg and Search::interpolateEarly:
    # configured half-life 19 scales to board width on square boards.
    return 0.15 + (0.75 - 0.15) * 0.5 ** (move_number / board_size)


def exploration_settings(config, profile):
    validate_profile(profile)
    modern = profile == "katago"
    return {
        "profile": profile, "version": 1,
        "temperature_early": 0.75 if modern else 1.0,
        "temperature_late": 0.15 if modern else 0.1,
        "temperature_halflife_moves": config.board_size if modern else None,
        "temperature_switch_move": None if modern else config.search.temperature_moves,
        "root_noise": "policy-shaped-dirichlet" if modern else "symmetric-dirichlet",
        "root_noise_weight": config.search.dirichlet_epsilon,
        "root_noise_total_concentration": 10.83 if modern else None,
        "root_noise_alpha_per_move": None if modern else config.search.dirichlet_alpha,
        "policy_target": "retrospective-puct" if modern else "raw-visits",
        "policy_target_temperature": 1.0,
        "chosen_move_prune": 1.0 if modern else 0.0,
        "lcb_selection": False,
    }
