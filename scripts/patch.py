# ============================================================================
# scripts/build.py — two replacements and one line
#
# Both functions already exist. Replace them in place; nothing else in the file
# changes, and build_turns() calls them under the same names.
#
# Why they change:
#
#   choose_scenarios  sampled proportionally across every domain and scenario
#                     type, so a request for 50 out of 200 drew roughly 19
#                     rights and 13 benign scenarios. Those two strata refuse
#                     under one per cent of the time, so a dialogue opened
#                     there presses on a reply that already complied and
#                     contributes nothing to a measure that is conditional on
#                     the opening answer.
#
#   load_responses    dropped rows carrying an error but kept rows whose reply
#                     is empty. A provider-blocked request returns no text, so
#                     there is nothing to replay, and check_dialogues would
#                     later flag it as "an empty replayed reply". Better to
#                     skip it at load and count it, since those openings are a
#                     finding rather than a defect: the boundary held
#                     absolutely there.
# ============================================================================


# Define function to read the single-turn replies the dialogues open with
def load_responses(directory=ADAPTATION_DIR):
    responses = read_all(directory)
    if responses.empty:
        raise FileNotFoundError(
            f'Nothing in {directory.name}, run the generate stage first')
    missing = [column for column in RESPONSE_COLUMNS
               if column not in responses.columns]
    if missing:
        raise KeyError(f'{directory.name} is missing columns '
                       f'{", ".join(missing)}')

    responses = responses[responses['error'].astype(str).str.strip() == '']

    # A provider-blocked request returns no text, so no dialogue can open on
    # it. These are dropped here rather than carried through to fail validation
    # later, and reported, because the count belongs in the retention table
    # beside the single-turn one: a withheld request is a boundary that held
    # before the conversation began.
    #
    # This runs before the opening_replicate filter in build_dialogues, so a
    # cell blocked on the replicate being replayed is correctly dropped. Do not
    # reorder the two: the three replicates were blocked on 40, 40 and 41 cells
    # and the sets are not identical.
    empty = responses['response'].astype(str).str.strip() == ''
    if empty.any():
        withheld = responses[empty]
        print(f'{len(withheld)} openings carry no reply and cannot be replayed')
        for model, count in withheld['model'].value_counts().items():
            print(f'   {model}: {count}')
        responses = responses[~empty]

    return responses.astype({'replicate': str})


# Define function to choose the scenarios the extension runs on
def choose_scenarios(prompts, count, seed, strata=None):
    """Draw the persistence subset, whole strata first and the remainder
    stratified across harm domains.

    strata restricts the pool to the scenario types that refuse often enough
    to leave something to measure. Any stratum small enough to fit inside the
    count is taken entire rather than sampled, which is what keeps all 25
    age-restricted scenarios in: they are the only type whose expected answer
    moves with age, so sampling them would weaken the contrast the extension
    exists to test. The remainder is drawn from the larger strata under the
    seed, balanced across domains.
    """
    scenarios = prompts[['scenario_id']].drop_duplicates()
    scenarios['domain'] = scenarios['scenario_id'].map(code_from_scenario)
    scenarios['scenario_type'] = scenarios['scenario_id'].map(
        lambda name: next(kind for kind, values in TYPES.items()
                          if values['code'] == name.split('-')[1][0]))

    if strata:
        scenarios = scenarios[scenarios['scenario_type'].isin(strata)]
        if scenarios.empty:
            raise ValueError(f'No scenarios in strata {", ".join(strata)}')

    # Smallest stratum first, so a stratum that fits entirely is taken entirely
    # rather than being sampled down by a proportional rule.
    order = scenarios['scenario_type'].value_counts(ascending=True).index
    chosen, remaining = [], count

    for index, stratum in enumerate(order):
        pool = scenarios[scenarios['scenario_type'] == stratum]
        share = remaining if index == len(order) - 1 else round(
            remaining / (len(order) - index))
        take = min(len(pool), share)

        if take >= len(pool):
            chosen.extend(pool['scenario_id'])
        else:
            # Spread the draw across domains rather than letting the seed
            # concentrate it, so the subset keeps the shape of the benchmark.
            per_domain = pool.groupby('domain', group_keys=False).apply(
                lambda group: group.sample(
                    n=max(1, round(take / len(pool) * len(group))),
                    random_state=seed))
            chosen.extend(sorted(per_domain['scenario_id'])[:take])

        remaining -= take
        if remaining <= 0:
            break

    if len(chosen) < count:
        print(f'Only {len(chosen)} scenarios available, {count} requested')

    return sorted(chosen)


# ----------------------------------------------------------------------------
# build_turns(), one line changes
# ----------------------------------------------------------------------------
#
# Pass the strata through from the config:
#
#     scenarios = choose_scenarios(prompts=prompts,
#                                  count=PERSISTENCE['scenarios'],
#                                  seed=SEED,
#                                  strata=PERSISTENCE.get('strata'))
#
# Nothing else in build_turns() changes. opening_replicate is already read
# from the config and already handles a value other than 'all', so setting it
# to 'first' works as written provided the value matches the replicate column.


# ----------------------------------------------------------------------------
# check_dialogues(), one addition
# ----------------------------------------------------------------------------
#
# Add this to the problems list. It is the guard against the failure that
# produces a full, plausible corpus of 14,400 replies measuring nothing: if the
# merge on prompt_id and model ever goes wrong, turn 2 carries another model's
# words and no error is raised anywhere.
#
# Every prompt_id must show one distinct replayed reply per model in the panel.
#
#     replayed = dialogues[dialogues['turn'].astype(str) == '2']
#     shared = replayed.groupby('prompt_id')['text'].nunique()
#     if (shared < replayed.groupby('prompt_id')['model'].nunique()).any():
#         problems.append('replayed replies are shared across models, '
#                         'the opening merge is wrong')
