"""Interpret provider blocking and truncation flags.

Notes
-----
Read raw provider records and standardise blocking and truncation status.
"""

import json
import sys

from settings import ADAPTATION_DIR, BATCHES_DIR, RESPONSE_COLUMNS
from utils import read_lines, result_path, section


CAP_REACHED = {'max_tokens', 'length', 'incomplete'}


def body_of(record):
    if 'result' in record:
        return (record['result'] or {}).get('message') or {}
    response = record.get('response') or {}
    return response.get('body') or response


def model_of(record):
    body = body_of(record)
    return body.get('model') or body.get('modelVersion') or ''


def key_of(record):
    key = str(record.get('custom_id') or record.get('key') or '')
    prompt_id, _, replicate = key.rpartition('-r')
    return prompt_id, replicate


def finishes_of(body):
    reasons = [str(body.get('stop_reason') or ''), str(body.get('status') or ''),
               str(body.get('done_reason') or '')]
    for choice in (body.get('choices') or body.get('candidates') or []):
        reasons.append(str(choice.get('finish_reason')
                           or choice.get('finishReason') or ''))
    return [reason.lower() for reason in reasons if reason]


FILTERED = ('content filtering policy', 'content_filter', 'blocked by')


BLOCKING = {'content_filter', 'prohibited_content', 'safety', 'blocklist',
            'spii', 'image_safety', 'recitation'}


def flags_of(body, record=None):
    finishes = finishes_of(body)
    blocked = ((body.get('promptFeedback') or {}).get('blockReason')
               or (body.get('prompt_feedback') or {}).get('block_reason')
               or next((reason.upper() for reason in finishes
                        if reason in BLOCKING), ''))
    if not blocked and record:
        said = str((record.get('result') or {}).get('error')
                   or record.get('error') or '').lower()
        if any(word in said for word in FILTERED):
            blocked = 'CONTENT_FILTER'
    truncated = bool(body.get('incomplete_details')) or bool(
        CAP_REACHED & set(finishes))
    return str(blocked), truncated


def raw_flags(model=''):
    found = {}
    for path in sorted(BATCHES_DIR.glob('*output.jsonl')):
        records = [json.loads(line) for line in path.read_text().splitlines()
                   if line.strip()]


        named = [model_of(record) for record in records if model_of(record)]
        belongs = max(set(named), key=named.count) if named else ''
        if model and belongs and belongs != model:
            continue
        for record in records:
            if model and not belongs and model_of(record) != model:
                continue
            found[key_of(record)] = flags_of(body_of(record), record)
    return found


def apply(model, write=True):
    path = result_path(model, ADAPTATION_DIR)
    replies = read_lines(path)
    if replies.empty:
        return {'replies': 0, 'matched': 0, 'blocked': 0, 'truncated': 0,
                'empty': 0, 'unexplained': 0}

    known = raw_flags(model)
    rows, counts = [], {'replies': len(replies), 'matched': 0, 'blocked': 0,
                        'truncated': 0, 'empty': 0, 'unexplained': 0}
    for reply in replies.to_dict('records'):
        key = (str(reply['prompt_id']), str(reply['replicate']))
        counts['matched'] += key in known
        if key in known:
            reply['blocked'], reply['truncated'] = known[key]
        else:


            reply['blocked'] = str(reply.get('blocked') or '')
            reply['truncated'] = str(reply.get('truncated', '')).lower() in (
                'true', '1')
        blocked, truncated = reply['blocked'], reply['truncated']
        empty = not str(reply.get('response') or '').strip()
        counts['blocked'] += bool(blocked)
        counts['truncated'] += bool(truncated)
        counts['empty'] += empty
        counts['unexplained'] += empty and not blocked and not truncated
        rows.append({name: reply.get(name, '') for name in RESPONSE_COLUMNS})

    if write:
        path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    return counts


def report(model, counts):
    print(f'{model}')
    print(f"  {counts['replies']:,} replies, {counts['matched']:,} matched to a "
          f"raw record")
    print(f"  {counts['blocked']} blocked, {counts['truncated']} truncated, "
          f"{counts['empty']} empty")
    if counts['matched'] < counts['replies']:
        print(f"  {counts['replies'] - counts['matched']:,} had no raw record, "
              f"so their flags are whatever ingest wrote. Put the provider "
              f"files back in data/batches to check them.")
    if counts['unexplained']:
        print(f"  {counts['unexplained']} empty for neither reason, worth reading")


if __name__ == '__main__':
    write = '--write' in sys.argv
    models = sorted({model_of(json.loads(line))
                     for path in BATCHES_DIR.glob('*output.jsonl')
                     for line in path.read_text().splitlines()[:1]} - {''})
    if not models:
        raise SystemExit(f'No raw output files in {BATCHES_DIR}')

    section('Flags')
    for model in models:
        report(model, apply(model, write=write))
    if not write:
        print('\nNothing written. Run again with --write to update '
              'results/adaptation/.')
