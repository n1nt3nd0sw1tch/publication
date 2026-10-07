"""Audit external resources named in model replies.

Notes
-----
Extract, classify, and summarise resource mentions across both experiments.
"""

import argparse
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis
from analysis import MACRO, MACHINE, NAME, ORDER, SUPPLEMENT
from settings import (ADAPTATION_DIR, BENCHMARK_PATH, CONFIG_DIR, DATA_DIR,
                      MULTI_CLASSIFICATION_DIR, PROMPTS_PATH, RESULTS_DIR,
                      read_config)
from utils import read_lines, section

CONFIG = read_config(CONFIG_DIR / 'resources.yml')
SCHEMA = CONFIG['schema']
KINDS = SCHEMA['kinds']
SPECIFIC_KINDS = set(SCHEMA['specific_kinds'])
ENTITY_TYPES = SCHEMA['entity_types']

ENTITY_TYPE = {'helpline': 'Helpline', 'service': 'Service',
               'organisation': 'Organisation', 'emergency': 'Emergency',
               'generic': 'Other'}
AMBIGUOUS = {str(number): dict(readings) for number, readings
             in CONFIG.get('ambiguous_numbers', {}).items()}
CONTACT_KINDS = ['phone', 'shortcode', 'emergency', 'domain', 'email',
                 'organisation', 'candidate']
BEFORE = CONFIG['window']['before']
AFTER = CONFIG['window']['after']

RESOURCE_DIR = RESULTS_DIR / 'resources'
SHEET_PATH = DATA_DIR / 'resources.csv'


EMAIL = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')
URL = re.compile(
    r'(?:https?://|www\.)[^\s<>()\[\]"\'`,]+'
    r'|\b(?:[\w-]+\.)+(?:org\.uk|co\.uk|gov\.uk|ac\.uk|com\.au|org\.au|gov\.au'
    r'|co\.nz|org\.nz|org|com|net|gov|edu|int|info|help|life|ie|ca|us|uk|au|nz|in)'
    r'(?:/[^\s<>()\[\]"\'`,]*)?')


NUMBER = re.compile(r'(?<![\w@/.$£€%-])(?:\+?\d|\(\d{2,4}\))'
                    r'[\d \t.\u2010-\u2015-]{1,22}\d')


VANITY = re.compile(r'(?<![\w-])1?[ \t.-]?8(?:00|33|44|55|66|77|88)'
                    r'[ \t.-]?(?:[A-Z0-9]{1,10}[ \t.-]?){1,4}[A-Z]{2,10}\b')


YEAR = re.compile(r'^(19|20)\d{2}$')
DOT = re.compile(r'\d\.\d')


DOTTED_FLOOR = 7
UNIT = re.compile(
    r'^\s*(?:%|(?:mg|ml|mcg|g|kg|lb|lbs|oz|cm|mm|km|ft|kcal|cal|calories|'
    r'steps|reps|sets|hours|hour|hrs|minutes|minute|mins|seconds|days|day|'
    r'weeks|week|months|month|years|year|olds?|times|degrees|bpm|per)\b)',
    re.I)
MONEY = re.compile(r'[$£€]\s*$')


PUBLIC_SUFFIX = {'co.uk', 'org.uk', 'gov.uk', 'ac.uk', 'net.uk', 'nhs.uk',
                 'com.au', 'org.au', 'gov.au', 'net.au', 'co.nz', 'org.nz',
                 'govt.nz', 'co.in', 'org.in', 'gov.in'}


KEYPAD = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                       '22233344455566677778889999')


TRUNK = {'44': '0', '61': '0', '353': '0', '64': '0', '91': '0', '1': ''}


ACRONYM = re.compile(r'(?<![\w-])[A-Z][A-Z&-]{2,7}(?![\w-])')


KEYWORD = re.compile(r'(?:text|texting|send|message|reply|type|write|sms)'
                     r'(?:\s+\w+){0,2}\s*$', re.I)
STOPWORDS = {str(word).upper() for word in CONFIG['acronym_stopwords']}
ORGANISATION_CUE = re.compile(
    r'(?<![\w-])(?:organisation|organization|charity|charities|nonprofit|'
    r'non-profit|foundation|helpline|hotline|support|service|agency|'
    r'reach out|report to|guidance from)', re.I)


PASS = ['regex', 'dictionary', 'suffix', 'adjacent', 'acronym', 'ner']


NER = None

ARTICLE = re.compile(r'^(?:the|a|an|and|or|for|of|on|to|call|text|visit)\s+', re.I)
TAIL = re.compile(r'\s+(?:at|on|via|by|call|text|number|line)$', re.I)


def enable_ner(name):
    global NER
    import spacy
    NER = spacy.load(name, disable=['parser', 'lemmatizer', 'tagger',
                                    'attribute_ruler', 'textcat'])
    return NER


def normalise_number(value):
    digits = ''.join(value.upper().translate(KEYPAD).split())
    digits = re.sub(r'[^\d+]', '', digits)
    if digits.startswith('+'):
        rest = digits[1:]
        for code, trunk in sorted(TRUNK.items(), key=lambda pair: -len(pair[0])):
            if rest.startswith(code):
                return trunk + rest[len(code):]
        return rest
    if len(digits) == 11 and digits.startswith('1'):
        return digits[1:]
    return digits


def normalise_host(value):
    text = str(value).strip().rstrip('.,;:)')
    if '//' not in text:
        text = 'http://' + text
    host = urlsplit(text).netloc.lower()
    host = host.split('@')[-1].split(':')[0]
    return host[4:] if host.startswith('www.') else host


def trim(value):
    value = re.sub(r"^[\s\-\u2013\u2014*'\"(]+|[\s\-\u2013\u2014*'\":;,.)]+$",
                   '', str(value))
    return TAIL.sub('', ARTICLE.sub('', value)).strip()


def build_index():
    numbers, hosts, addresses, names, entries = {}, {}, {}, [], {}
    for entry in CONFIG['resources']:
        identifier = entry['id']
        entries[identifier] = {
            'canonical': entry['name'],
            'kind': entry.get('kind', 'organisation'),
            'jurisdiction': '|'.join(entry.get('jurisdiction', ['INT'])),
            'official': entry.get('official', ''),
            'seeded': True}
        for number in entry.get('numbers', []):
            numbers.setdefault(normalise_number(str(number)), identifier)
        for host in entry.get('domains', []):
            hosts.setdefault(normalise_host(host), identifier)
        for address in entry.get('emails', []):
            addresses.setdefault(str(address).lower(), identifier)
        for name in entry.get('names', []):
            names.append((str(name), identifier))

    for group in CONFIG['generic']:
        entries[group['id']] = {
            'canonical': group['name'], 'kind': 'generic',
            'jurisdiction': 'INT', 'official': '', 'seeded': True}

    cased = [(name, key) for name, key in names if len(name.split()) == 1]
    loose = [(name, key) for name, key in names if len(name.split()) > 1]
    lookup = {name: key for name, key in cased}
    lookup.update({name.lower(): key for name, key in loose})

    def alternation(pairs, flags=0):
        if not pairs:
            return None
        ordered = sorted({name for name, _ in pairs}, key=len, reverse=True)
        return re.compile(r'(?<![\w-])(?:' + '|'.join(re.escape(name)
                          for name in ordered) + r')(?![\w-])', flags)

    generic = [(alternation([(text, group['id']) for text in group['patterns']],
                            re.I), group['id'])
               for group in CONFIG['generic']]

    suffixes = '|'.join(re.escape(word)
                        for word in CONFIG['organisation_suffixes'])
    candidate = re.compile(
        r"\b(?:[A-Z][\w&.'\u2019-]*|of|for|and|the|on)"
        r"(?:\s+(?:[A-Z][\w&.'\u2019-]*|of|for|and|the|on)){0,4}"
        r'\s+(?:' + suffixes + r')\b')
    trailing = re.compile(
        r"(?:[A-Z][\w&.'\u2019-]*)(?:\s+(?:[A-Z][\w&.'\u2019-]*|of|for|and|the))"
        r'{0,4}\s*(?:[:\u2013\u2014-]|\bat\b|\bon\b)?\s*$')

    return {'numbers': numbers, 'hosts': hosts, 'emails': addresses,
            'entries': entries, 'lookup': lookup,
            'cased': alternation(cased), 'loose': alternation(loose, re.I),
            'generic': generic, 'candidate': candidate, 'trailing': trailing,
            'cues': re.compile(r'(?<![\w-])(?:' + '|'.join(
                re.escape(cue) for cue in CONFIG['contact_cues']) + r')', re.I)}


INDEX = build_index()


def build_jurisdiction():
    compiled = {}
    for code, terms in CONFIG['jurisdiction_terms'].items():
        cased = [term for term in terms
                 if len(term.split()) == 1 and len(term.replace('.', '')) < 5]
        loose = [term for term in terms if term not in cased]
        parts = []
        if cased:
            parts.append(re.compile(r'(?<![\w-])(?:' + '|'.join(
                re.escape(term) for term in cased) + r')(?![\w-])'))
        if loose:
            parts.append(re.compile(r'(?<![\w-])(?:' + '|'.join(
                re.escape(term) for term in loose) + r')(?![\w-])', re.I))
        compiled[code] = parts
    return compiled


JURISDICTION = build_jurisdiction()
HEDGE = re.compile('|'.join(re.escape(phrase)
                            for phrase in CONFIG['locality_hedges']), re.I)


def jurisdictions_in(text):
    return [code for code, patterns in JURISDICTION.items()
            if any(pattern.search(text) for pattern in patterns)]


def mask(text, spans):
    characters = list(text)
    for start, end in spans:
        for position in range(start, end):
            characters[position] = ' '
    return ''.join(characters)


def rejected(text, span, digits):
    start, end = span
    raw = text[start:end]
    if len(digits) < 3:
        return 'short'
    if DOT.search(raw) and len(digits) < DOTTED_FLOOR:
        return 'decimal'
    if YEAR.match(digits) and raw == digits:
        return 'year'
    if MONEY.search(text[max(0, start - 2):start]):
        return 'money'
    if UNIT.match(text[end:end + 12]):
        return 'unit'
    return ''


def extract(text, counts=None):
    text = str(text or '')
    if not text.strip():
        return []
    counts = counts if counts is not None else Counter()
    found, spans = [], []

    for match in EMAIL.finditer(text):
        spans.append(match.span())
        found.append(('email', match.group(0).lower(), match.group(0),
                      match.start(), 'regex'))
    working = mask(text, spans)

    address_spans = []
    for match in URL.finditer(working):
        host = normalise_host(match.group(0))
        if not host or '.' not in host or host in PUBLIC_SUFFIX:
            continue
        address_spans.append(match.span())


        found.append(('domain', host, match.group(0).rstrip('.,;:)'),
                      match.start(), 'regex'))
    working = mask(working, address_spans)
    body = working

    number_spans = []
    for pattern in (VANITY, NUMBER):
        working = mask(working, number_spans)
        number_spans = []
        for match in pattern.finditer(working):
            raw = match.group(0).strip()
            digits = normalise_number(raw)
            reason = rejected(working, match.span(), digits)
            if reason:
                counts[f'rejected {reason}'] += 1
                continue
            number_spans.append(match.span())
            if len(digits) >= 7:
                kind = 'phone'
            else:


                window = working[max(0, match.start() - BEFORE):
                                 match.end() + AFTER]
                if not INDEX['cues'].search(window):
                    counts['rejected uncued'] += 1
                    continue
                kind = 'shortcode'
            found.append((kind, digits, raw, match.start(), 'regex'))

    for pattern in (INDEX['cased'], INDEX['loose']):
        if pattern is None:
            continue
        for match in pattern.finditer(body):
            found.append(('organisation', match.group(0), match.group(0),
                          match.start(), 'dictionary'))

    for pattern, identifier in INDEX['generic']:
        if pattern is None:
            continue
        for match in pattern.finditer(body):
            found.append(('generic', identifier, match.group(0), match.start(),
                          'dictionary'))


    known = [(start, start + len(surface))
             for kind, _, surface, start, _source in found
             if kind == 'organisation']
    claimed = []

    def novel(start, end):
        for register, reason in ((known, 'overlapping a known name'),
                                 (claimed, 'overlapping an earlier candidate')):
            if any(start < right and left < end for left, right in register):
                counts[reason] += 1
                return False
        return True


    def claim(start, end):
        claimed.append((start, end))

    for match in sorted(INDEX['candidate'].finditer(body),
                        key=lambda hit: hit.start() - len(hit.group(0)) / 1e6):
        value = trim(match.group(0))
        if not value:
            continue
        if not novel(match.start(), match.end()):
            continue
        claim(match.start(), match.end())
        found.append(('candidate', value, match.group(0), match.start(),
                      'suffix'))


    for kind, value, surface, start, _source in list(found):
        if kind not in ('phone', 'shortcode', 'domain'):
            continue
        line = body.rfind('\n', 0, start) + 1
        head = INDEX['trailing'].search(body[line:start])
        if not head:
            continue
        value = trim(head.group(0))
        named = len(value.split()) > 1 or (len(value) >= 5 and value[:1].isupper())
        if not named:
            continue
        if not novel(line + head.start(), line + head.end()):
            continue
        claim(line + head.start(), line + head.end())
        found.append(('candidate', value, head.group(0),
                      line + head.start(), 'adjacent'))

    for match in ACRONYM.finditer(body):
        token = match.group(0)
        if token.replace('&', '').replace('-', '') in STOPWORDS:
            continue
        if not novel(match.start(), match.end()):
            continue
        if KEYWORD.search(body[max(0, match.start() - 24):match.start()]):
            counts['rejected keyword'] += 1
            continue
        window = body[max(0, match.start() - BEFORE):match.end() + AFTER]
        if INDEX['cues'].search(window) or ORGANISATION_CUE.search(window):
            claim(match.start(), match.end())
            found.append(('candidate', token, token, match.start(), 'acronym'))

    if NER is not None:
        for entity in NER(body).ents:
            if entity.label_ != 'ORG':
                continue
            value = trim(entity.text)
            if not value or value.upper() in STOPWORDS:
                continue
            if not novel(entity.start_char, entity.end_char):
                continue
            claim(entity.start_char, entity.end_char)
            found.append(('candidate', value, entity.text, entity.start_char,
                          'ner'))


    seen, unique = set(), []
    for mention in sorted(found, key=lambda item: PASS.index(item[4])):
        key = (mention[0], mention[1], mention[3])
        if key not in seen:
            seen.add(key)
            unique.append(mention)
    return unique


def resolve(kind, value, named_jurisdiction=''):
    def entity_of(key, contact, contact_kind, ambiguous=False):
        entry = INDEX['entries'].get(key, {})
        return {'entity_id': key, 'contact_id': contact, 'kind': contact_kind,
                'entity_type': ENTITY_TYPE.get(entry.get('kind', ''), ''),
                'ambiguous': int(ambiguous)}

    if kind == 'generic':
        return entity_of(value, f'generic::{value}', 'generic')

    if kind in ('phone', 'shortcode'):


        if value in AMBIGUOUS:
            readings = AMBIGUOUS[value]
            said = {code for code in named_jurisdiction.split('|') if code}
            fits = {readings[code] for code in said & set(readings)}
            if len(fits) == 1:
                key = fits.pop()
                entry = INDEX['entries'][key]
                kind = 'emergency' if entry['kind'] == 'emergency' else kind
                return entity_of(key, f'{kind}::{key}::{value}', kind)


            return {'entity_id': f'number:{value}?ambiguous',
                    'contact_id': f'{kind}::number:{value}?ambiguous',
                    'kind': kind, 'entity_type': 'Unknown', 'ambiguous': 1}
        known = INDEX['numbers'].get(value)
        if known:
            entry = INDEX['entries'][known]
            kind = 'emergency' if entry['kind'] == 'emergency' else kind
            return entity_of(known, f'{kind}::{known}::{value}', kind)
        return entity_of(f'number:{value}', f'{kind}::number:{value}', kind)

    if kind == 'domain':
        host = value
        while host.count('.') >= 1:
            if host in INDEX['hosts']:
                known = INDEX['hosts'][host]
                return entity_of(known, f'domain::{known}::{value}', 'domain')
            host = host.split('.', 1)[1]
        return entity_of(f'domain:{value}', f'domain::domain:{value}', 'domain')

    if kind == 'email':
        known = INDEX['emails'].get(value)
        if known:
            return entity_of(known, f'email::{known}::{value}', 'email')
        return entity_of(f'email:{value}', f'email::email:{value}', 'email')

    key = INDEX['lookup'].get(value) or INDEX['lookup'].get(value.lower())
    if key:


        return entity_of(key, f'organisation::{key}', 'organisation')
    return entity_of(f'name:{value.lower()}', f'{kind}::name:{value.lower()}',
                     kind)


def qualification(text, start, end):
    window = text[max(0, start - BEFORE):end + AFTER]
    return '|'.join(sorted(jurisdictions_in(window))), bool(HEDGE.search(window))


def mentions_of(text, keys, counts):
    rows = []
    for kind, value, surface, start, source in extract(text, counts):
        named, hedged = qualification(str(text), start, start + len(surface))
        found = resolve(kind, value, named)
        entity, contact = found['entity_id'], found['contact_id']
        entry = INDEX['entries'].get(entity, {})
        serves = entry.get('jurisdiction', '')
        rows.append({**keys, 'contact_id': contact, 'entity_id': entity,
                     'kind': found['kind'], 'entity_type': found['entity_type'],
                     'ambiguous': found['ambiguous'], 'pass': source,
                     'value': value, 'surface': surface, 'position': start,
                     'canonical': entry.get('canonical', ''),
                     'serves': serves, 'named_jurisdiction': named,
                     'hedged': int(hedged),
                     'qualified': int(bool(named) or hedged),
                     'mismatch': int(bool(named) and bool(serves)
                                     and serves != 'INT'
                                     and not set(named.split('|'))
                                     & set(serves.split('|')))})
    seen, unique = set(), []
    for row in rows:
        key = (row['contact_id'], row['position'])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def facts():
    prompts = pd.read_csv(PROMPTS_PATH)
    benchmark = pd.read_csv(BENCHMARK_PATH)
    return prompts[['prompt_id', 'scenario_id', 'condition', 'age', 'band',
                    'signal', 'cue']].merge(
        benchmark[['scenario_id', 'domain', 'scenario_type']],
        on='scenario_id', validate='many_to_one').rename(
        columns={'domain': 'harm_domain'})


def load_single(limit=0):
    parts = [read_lines(path) for path in sorted(ADAPTATION_DIR.glob('*.jsonl'))]
    parts = [frame for frame in parts if not frame.empty]
    if not parts:
        raise SystemExit(f'Nothing collected in {ADAPTATION_DIR}')
    replies = pd.concat(parts, ignore_index=True)
    replies['response'] = replies['response'].fillna('').astype(str)
    returned = replies[replies['response'].str.strip().ne('')].copy()
    returned['replicate'] = returned['replicate'].astype(int)
    if limit:
        returned = returned.head(limit)
    return returned.merge(facts(), on='prompt_id', how='left',
                          validate='many_to_one')


def load_dialogue(limit=0):
    turns = pd.read_csv(MULTI_CLASSIFICATION_DIR / 'turns.csv')
    replies = turns[turns['role'].eq('assistant')].copy()
    replies['text'] = replies['text'].fillna('').astype(str)
    replies = replies[replies['text'].str.strip().ne('')]
    return replies.head(limit) if limit else replies


def write_sheet(mentions, path=SHEET_PATH):
    grouped = mentions.groupby(['contact_id', 'entity_id', 'kind'], dropna=False)
    sheet = grouped.agg(
        entity_type=('entity_type', 'first'),
        ambiguous=('ambiguous', 'max'),
        canonical=('canonical', 'first'),
        value=('value', 'first'),
        surface_forms=('surface', lambda column: ' | '.join(
            sorted({' '.join(str(form).split()) for form in column})[:8])),
        mentions=('contact_id', 'size'),
        models=('model', 'nunique'),
    ).reset_index()

    replies = mentions.drop_duplicates(['contact_id', 'reply_id'])
    sheet = sheet.merge(replies.groupby('contact_id').size().rename('replies'),
                        on='contact_id', how='left')
    for experiment in ('single', 'multi'):
        part = replies[replies['experiment'].eq(experiment)]
        sheet = sheet.merge(
            part.groupby('contact_id').size().rename(f'replies_{experiment}'),
            on='contact_id', how='left')
    sheet[['replies_single', 'replies_multi']] = (
        sheet[['replies_single', 'replies_multi']].fillna(0).astype(int))

    seeded = pd.DataFrame([
        {'entity_id': key, 'seeded_jurisdiction': entry['jurisdiction'],
         'official': entry['official'], 'seeded': 'Yes'}
        for key, entry in INDEX['entries'].items()])
    sheet = sheet.merge(seeded, on='entity_id', how='left')
    sheet['seeded'] = sheet['seeded'].fillna('No')
    sheet[['seeded_jurisdiction', 'official']] = (
        sheet[['seeded_jurisdiction', 'official']].fillna(''))
    sheet['specific'] = np.where(sheet['kind'].isin(SPECIFIC_KINDS), 'Yes', 'No')
    sheet['canonical'] = sheet['canonical'].fillna('').replace(
        '', np.nan).fillna(sheet['value'].astype(str))
    sheet['contacts_for_entity'] = sheet.groupby('entity_id')[
        'contact_id'].transform('size')

    order = ['contact_id', 'entity_id', 'canonical', 'kind', 'entity_type',
             'ambiguous', 'value', 'specific', 'seeded',
             'seeded_jurisdiction', 'official', 'contacts_for_entity',
             'surface_forms', 'mentions', 'replies', 'replies_single',
             'replies_multi', 'models']
    sheet = sheet.sort_values(['specific', 'replies'], ascending=[False, False])
    sheet = sheet[order]
    sheet.to_csv(path, index=False)
    return sheet


def read_automatic_sheet(path=SHEET_PATH):
    if not path.exists():
        raise SystemExit(f'{path} is missing. Run extract first.')
    sheet = pd.read_csv(path, dtype=str).fillna('')
    required = {'contact_id', 'kind', 'specific', 'seeded', 'entity_type',
                'replies', 'value'}
    missing = sorted(required - set(sheet.columns))
    if missing:
        raise SystemExit(f'{path} is missing automated-report columns: ' +
                         ', '.join(missing))
    sheet = sheet[pd.to_numeric(sheet['replies'], errors='coerce')
                  .fillna(0).gt(0)].copy()
    specific = sheet['specific'].eq('Yes')
    sheet['dictionary'] = specific & sheet['seeded'].eq('Yes')
    sheet['candidate'] = specific

    sheet['type'] = sheet['entity_type'].replace('', 'Unknown')
    return sheet


PENDING = []


def freeze(table, name, folder=SUPPLEMENT):
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / f'{name}.csv')
    PENDING.append(name)
    return table


def panel_rate(frame, column, order=None, conditions=None):
    rows, per_model, points = [], {}, []
    for label in (order or ORDER):
        part = frame[frame['label'].eq(label)]
        per_model[label] = analysis.by_scenario(part, column, conditions)
        point, low, high = analysis.macro_average({label: per_model[label]})
        points.append(point)
        rows.append({'Model': label,
                     **analysis.bounds(100 * point, 100 * low, 100 * high),
                     'replies': len(part.dropna(subset=[column])),
                     'scenarios': int(per_model[label].notna().sum())})
    point, low, high = analysis.macro_average(per_model)
    finite = [value for value in points if not pd.isna(value)]
    if finite and not pd.isna(point):
        assert np.isclose(point, float(np.mean(finite)), atol=1e-9), (
            f'{column}: the macro-average is not the mean of the model rates')
        assert min(finite) - 1e-9 <= point <= max(finite) + 1e-9, (
            f'{column}: the macro-average {point:.4f} sits outside the model '
            f'range {min(finite):.4f} to {max(finite):.4f}')
    rows.append({'Model': MACRO,
                 **analysis.bounds(100 * point, 100 * low, 100 * high),
                 'replies': '', 'scenarios': ''})
    table = pd.DataFrame(rows).set_index('Model')
    table.index.name = 'Model'
    return table.rename(columns={'estimate': 'Rate (%)', 'low': 'Low',
                                 'high': 'High', 'replies': 'Replies',
                                 'scenarios': 'Scenarios'})


def scenario_grid(frame, column, split):
    return pd.DataFrame(
        {label: {value: analysis.by_scenario(
            frame[frame['label'].eq(label) & frame[split].eq(value)],
            column).mean() * 100
            for value in sorted(frame[split].dropna().unique())}
         for label in ORDER})


def fold(mentions, sheet, keys):
    marks = sheet.set_index('contact_id')[['dictionary', 'candidate']]
    joined = mentions.merge(marks, on='contact_id', how='left')
    joined[['dictionary', 'candidate']] = joined[
        ['dictionary', 'candidate']].fillna(False)


    serves = joined['serves'].fillna('')
    joined['seeded_limited'] = serves.ne('') & serves.ne('INT')
    joined['assumption'] = (joined['seeded_limited']
                            & joined['qualified'].eq(0)).astype(int)
    joined['mismatch_auto'] = joined['mismatch'].fillna(0).astype(int)

    reduced = None
    for name in READINGS:
        part = joined[joined[name]]
        counts = part.groupby(keys).agg(
            **{name: ('entity_id', 'nunique'),
               f'{name}_contacts': ('contact_id', 'nunique')})
        reduced = counts if reduced is None else reduced.join(counts, how='outer')

    predefined = joined[joined['dictionary']]
    diagnostics = predefined.groupby(keys).agg(
        assumption=('assumption', 'max'),
        mismatch=('mismatch_auto', 'max'))
    generic = joined[joined['kind'].eq('generic')].groupby(keys).size()
    reduced = (reduced.join(diagnostics, how='outer')
               .join(generic.rename('generic'), how='outer').reset_index())
    return reduced.fillna(0)


SEED = ['model', 'prompt_id', 'opening_replicate']


PRIMARY = 'dictionary'
READINGS = [PRIMARY, 'candidate']
READING_LABEL = {'dictionary': 'Predefined Resource Match',
                 'candidate': 'Automated Candidate Match'}


def named(stem, reading):
    return stem if reading == PRIMARY else f'{stem}_{reading}'


def report_single(sheet, reading=PRIMARY):
    primary = f'{reading}_any'
    mentions = read_mentions('single')
    keys = ['model', 'prompt_id', 'replicate']
    replies = pd.read_csv(RESOURCE_DIR / 'replies.csv')
    replies = replies.merge(fold(mentions, sheet, keys), on=keys,
                            how='left').fillna(0)
    replies['label'] = replies['model'].map(NAME)
    for name in READINGS:
        replies[f'{name}_any'] = (replies[name] > 0).astype(float)
    replies['only_generic'] = ((replies['candidate'] == 0)
                               & (replies['generic'] > 0)).astype(float)
    for column in ('assumption', 'mismatch'):
        replies[column] = replies[column].astype(float)

    verdicts = pd.read_csv(RESULTS_DIR / 'classification.csv',
                           usecols=['model', 'prompt_id', 'replicate',
                                    'service_signpost', 'expert_signpost'])
    replies = replies.merge(verdicts, on=keys, how='left')

    coverage = pd.concat([
        panel_rate(replies, 'dictionary_any'),
        panel_rate(replies, 'candidate_any')[['Rate (%)']].rename(
            columns={'Rate (%)': 'Automated Candidate Match (%)'}),
        panel_rate(replies, 'only_generic')[['Rate (%)']].rename(
            columns={'Rate (%)': 'Generic Only (%)'})], axis=1)
    coverage = coverage.rename(
        columns={'Rate (%)': 'Predefined Resource Match (%)'})
    if reading == PRIMARY:
        freeze(coverage, 'resource_s01_coverage')


    signposted = replies[replies['service_signpost'].eq('Yes')].copy()
    concordance = panel_rate(signposted, primary).rename(
        columns={'Rate (%)': f'{READING_LABEL[reading]} (%)'})
    freeze(concordance, named('resource_s02_signpost_concordance', reading))


    naming = replies[replies['dictionary_any'].eq(1)].copy()
    jurisdiction = pd.concat([
        panel_rate(naming, 'assumption')[['Rate (%)']].rename(
            columns={'Rate (%)': 'Automated Jurisdiction Assumption (%)'}),
        panel_rate(naming, 'mismatch')[['Rate (%)']].rename(
            columns={'Rate (%)': 'Automated Jurisdiction Mismatch (%)'})],
        axis=1)
    if reading == PRIMARY and not naming.empty:
        freeze(jurisdiction, 'resource_s03_jurisdiction')


    contrasts = {'Explicit Age (Minor) vs Explicit Age (Adult)':
                 (analysis.STATED_MINOR, analysis.STATED_ADULT),
                 'Implicit Cue (Minor) vs Implicit Cue (Adult)':
                 (analysis.IMPLICIT_MINOR, analysis.IMPLICIT_ADULT)}
    columns = {}
    for name, (minor, adult) in contrasts.items():
        for which in READINGS:


            if not replies[f'{which}_any'].any():
                columns[f'{name}, {READING_LABEL[which]} (pp)'] = {
                    label: '' for label in ORDER + [MACRO]}
                continue
            cells, per_model = {}, {}
            for label in ORDER:
                part = replies[replies['label'].eq(label)]
                per_model[label] = (
                    analysis.by_scenario(part, f'{which}_any', minor)
                    - analysis.by_scenario(part, f'{which}_any', adult))
                point, low, high = analysis.bootstrap_paired(per_model[label])
                cells[label] = analysis.interval(100 * point, 100 * low,
                                                 100 * high, sign=True)
            point, low, high = analysis.macro_average(per_model)
            cells[MACRO] = analysis.interval(100 * point, 100 * low, 100 * high,
                                             sign=True)
            columns[f'{name}, {READING_LABEL[which]} (pp)'] = cells
    difference = pd.DataFrame(columns).reindex(ORDER + [MACRO])
    difference.index.name = 'Model'
    if reading == PRIMARY:
        freeze(difference, 'resource_s04_minor_adult')

    by_domain = scenario_grid(replies, primary, 'harm_domain').round(1)
    by_domain.index.name = 'Harm Domain'
    freeze(by_domain, named('resource_s05_domain', reading))

    by_type = scenario_grid(replies, primary, 'scenario_type').reindex(
        analysis.SCENARIO_ORDER).round(1)
    by_type.index.name = 'Scenario Type'
    freeze(by_type, named('resource_s06_type', reading))


    channels = mentions.merge(sheet.set_index('contact_id')[[reading, 'type']],
                              on='contact_id', how='left')
    channels = channels[channels[reading].fillna(False)]
    rows = {}
    for axis, column, values in (('Contact Kind', 'kind', CONTACT_KINDS),
                                 ('Entity Type', 'type', ENTITY_TYPES)):
        for value in values:
            present = channels[channels[column].eq(value)]
            if present.empty:
                continue
            flag = replies[keys].merge(
                present[keys].drop_duplicates().assign(present=1.0),
                on=keys, how='left')
            marked = replies.assign(present=flag['present'].fillna(0).to_numpy())
            rows[f'{axis}: {value}'] = {
                label: analysis.by_scenario(
                    marked[marked['label'].eq(label)], 'present').mean() * 100
                for label in ORDER}
    if rows:
        channel = pd.DataFrame(rows).T.reindex(columns=ORDER).round(1)
        channel.index.name = 'Channel'
        freeze(channel, named('resource_s11_channel', reading))


    top = (mentions.merge(sheet.set_index('contact_id')[[reading]],
                          on='contact_id', how='left')
           .query(f'{reading} == True')
           .groupby(['canonical', 'entity_id', 'kind', 'value'], dropna=False)
           .agg(Replies=('reply_id', 'nunique'), Models=('model', 'nunique'))
           .reset_index().sort_values('Replies', ascending=False).head(30))
    top.columns = ['Resource', 'Entity', 'Kind', 'Value', 'Replies', 'Models']
    freeze(top.set_index('Resource'), named('resource_s07_most_common', reading))
    replies.to_csv(MACHINE / f"{named('resource_replies', reading)}.csv",
                   index=False)
    return replies


def read_mentions(experiment):
    mentions = pd.read_csv(
        RESOURCE_DIR / 'mentions.csv',
        dtype={'value': str, 'dialogue_id': str, 'method': str, 'band': str,
               'scenario_type': str, 'harm_domain': str})
    return mentions[mentions['experiment'].eq(experiment)]


def report_dialogue(sheet, reading=PRIMARY):
    mentions = read_mentions('multi')
    kept = sheet.set_index('contact_id')[[reading]]
    joined = mentions.merge(kept, on='contact_id', how='left')
    joined[reading] = joined[reading].fillna(False)
    joined = joined[joined[reading]]


    joined['key'] = joined['entity_id']


    ambiguous = int(joined['ambiguous'].fillna(0).astype(int).sum())
    joined = joined[joined['ambiguous'].fillna(0).astype(int).eq(0)]
    if ambiguous:
        print(f'  {ambiguous:,} ambiguous mentions left out of the entity sets')

    sets = joined.groupby(['dialogue_id', 'turn'])['key'].apply(frozenset)


    dialogue = pd.read_csv(RESOURCE_DIR / 'dialogue.csv')
    turns_present = dialogue.groupby('dialogue_id')['turn'].nunique()
    complete = turns_present[turns_present.eq(3)].index
    frame = dialogue[dialogue['dialogue_id'].isin(complete)].drop_duplicates(
        'dialogue_id')[SEED + ['dialogue_id', 'method', 'condition',
                               'scenario_id']].copy()
    empty = frozenset()

    def attach(source, prefix):
        for turn in (1, 2, 3):
            level = source.index.get_level_values('turn')
            at = (source.xs(turn, level='turn') if turn in level
                  else pd.Series(dtype=object))
            frame[f'{prefix}{turn}'] = frame['dialogue_id'].map(at).apply(
                lambda value: value if isinstance(value, frozenset) else empty)

    attach(sets, 'set')

    frame['label'] = frame['model'].map(NAME)
    carried = frame.groupby(SEED)['method'].nunique()
    matched_seeds = set(carried[carried.eq(3)].index)
    frame['matched'] = [seed in matched_seeds for seed
                        in zip(*(frame[column] for column in SEED))]
    matched = frame[frame['matched']]
    assert matched.groupby(SEED)['method'].nunique().eq(3).all(), (
        'a matched seed does not carry all three methods')


    assert matched.groupby(SEED)['set1'].apply(
        lambda column: column.nunique() == 1).all(), (
        'the three methods of a matched seed do not share a turn 1 entity set')


    opened = frame[frame['set1'].apply(len) > 0].copy()
    opened_matched = matched[matched['set1'].apply(len) > 0].copy()
    print(f'  dialogue cohort: {len(complete):,} complete of '
          f"{dialogue['dialogue_id'].nunique():,}, {len(matched):,} in the "
          f'matched three-method cohort, {len(opened):,} opened with a '
          f'resource under {READING_LABEL[reading]}')
    if opened.empty:


        print(f'  No dialogue opened with a resource under '
              f'{READING_LABEL[reading]}, so the persistence tables are not '
              f'written.')
        return None


    def indicators(first, later):
        pairs = list(zip(first, later))
        measures = {
            'Retained (%)': [len(a & b) / len(a) for a, b in pairs],
            'Any Dropped (%)': [float(bool(a - b)) for a, b in pairs],
            'Any Introduced (%)': [float(bool(b - a)) for a, b in pairs],
            'All Replaced (%)': [float(bool(b) and not (a & b)) for a, b in pairs],
            'No Later Resource (%)': [float(not b) for a, b in pairs],
            'None Opening Retained (%)': [float(not (a & b)) for a, b in pairs]}


        assert np.allclose(
            np.array(measures['None Opening Retained (%)']),
            np.array(measures['All Replaced (%)'])
            + np.array(measures['No Later Resource (%)'])), (
            'None Opening Retained is not All Replaced plus No Later Resource')
        return measures

    def summarise(part, values):
        marked = part.assign(measure=values)
        per_model = {label: analysis.by_scenario(
            marked[marked['label'].eq(label)], 'measure') for label in ORDER}
        point, low, high = analysis.macro_average(per_model)
        return analysis.interval(100 * point, 100 * low, 100 * high)


    rows = []
    for turn in (2, 3):
        row = {'Turn': f'Turn {turn}', 'Dialogues': len(opened)}
        for name, values in indicators(opened['set1'],
                                       opened[f'set{turn}']).items():
            row[name] = summarise(opened, values)
        rows.append(row)
    freeze(pd.DataFrame(rows).set_index('Turn'),
           named('resource_s08_persistence', reading))


    by_method = []
    for method, part in opened_matched.groupby('method'):
        measures = indicators(part['set1'], part['set3'])
        row = {'Method': method, 'Dialogues': len(part),
               'Scenarios': int(part['scenario_id'].nunique())}
        for name in ('Retained (%)', 'All Replaced (%)',
                     'None Opening Retained (%)'):
            row[name.replace(' (%)', ' at Turn 3 (%)')] = summarise(
                part, measures[name])
        by_method.append(row)
    freeze(pd.DataFrame(by_method).set_index('Method'),
           named('resource_s09_method', reading))

    marked = opened.assign(**{name: values for name, values
                              in indicators(opened['set1'],
                                            opened['set3']).items()})
    marked = marked.rename(columns={'Retained (%)': 'retained',
                                    'Any Introduced (%)': 'introduced',
                                    'None Opening Retained (%)': 'none_retained'})
    by_model = pd.concat([
        panel_rate(marked, 'retained').rename(
            columns={'Rate (%)': 'Retained at Turn 3 (%)'}),
        panel_rate(marked, 'none_retained')[['Rate (%)']].rename(
            columns={'Rate (%)': 'None Opening Retained (%)'}),
        panel_rate(marked, 'introduced')[['Rate (%)']].rename(
            columns={'Rate (%)': 'Any Introduced (%)'})], axis=1)
    freeze(by_model, named('resource_s10_model', reading))
    marked.drop(columns=[column for column in marked.columns
                         if column.startswith('set')]).to_csv(
        MACHINE / f"{named('resource_dialogue', reading)}.csv",
        index=False)
    return marked


def run_extract(experiments, limit):
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    counts, rows = Counter(), []


    prompts = pd.read_csv(PROMPTS_PATH)['prompt'].astype(str)
    named = sum(bool(jurisdictions_in(text)) for text in prompts)
    assert named == 0, ('a prompt names a jurisdiction, so J is no longer an '
                        'assumption on every reply')

    if 'single' in experiments:
        replies = load_single(limit)
        print(f'  adaptation   {len(replies):,} returned replies')
        for reply in replies.itertuples():
            rows.extend(mentions_of(
                reply.response,
                {'experiment': 'single', 'model': reply.model,
                 'prompt_id': reply.prompt_id, 'replicate': reply.replicate,
                 'reply_id': f'{reply.model}|{reply.prompt_id}|{reply.replicate}',
                 'dialogue_id': '', 'turn': 1, 'method': '',
                 'scenario_id': reply.scenario_id, 'condition': reply.condition,
                 'band': reply.band, 'scenario_type': reply.scenario_type,
                 'harm_domain': reply.harm_domain}, counts))
        replies[['model', 'prompt_id', 'replicate', 'scenario_id', 'condition',
                 'band', 'signal', 'cue', 'scenario_type', 'harm_domain']
                ].to_csv(RESOURCE_DIR / 'replies.csv', index=False)

    if 'multi' in experiments:
        turns = load_dialogue(limit)
        print(f'  dialogue     {len(turns):,} assistant turns')
        for turn in turns.itertuples():
            rows.extend(mentions_of(
                turn.text,
                {'experiment': 'multi', 'model': turn.model,
                 'prompt_id': turn.prompt_id, 'replicate': turn.opening_replicate,
                 'reply_id': f'{turn.dialogue_id}|{turn.turn}',
                 'dialogue_id': turn.dialogue_id, 'turn': turn.turn,
                 'method': turn.method, 'scenario_id': turn.scenario_id,
                 'condition': turn.condition, 'band': turn.band,
                 'scenario_type': '', 'harm_domain': ''}, counts))


        turns[['dialogue_id', 'model', 'prompt_id', 'method', 'condition',
               'scenario_id', 'turn', 'opening_replicate']].to_csv(
            RESOURCE_DIR / 'dialogue.csv', index=False)

    mentions = pd.DataFrame(rows)
    mentions.to_csv(RESOURCE_DIR / 'mentions.csv', index=False)
    sheet = write_sheet(mentions)

    live = pd.to_numeric(sheet['replies'], errors='coerce').fillna(0).gt(0)
    specific = live & sheet['specific'].eq('Yes')
    print(f'\n  {len(mentions):,} extracted mentions over '
          f"{mentions['reply_id'].nunique():,} replies")
    print(f"  {int(live.sum()):,} unique contacts over "
          f"{sheet.loc[live, 'entity_id'].nunique():,} entities, "
          f'{int(specific.sum()):,} contacts specific')
    for kind in KINDS:
        number = int((mentions['kind'] == kind).sum())
        if number:
            print(f'    {kind:<14} {number:>8,}')
    by_pass = mentions.groupby('pass')['reply_id'].nunique()
    print('  replies reached by each pass')
    for name in PASS:
        if name in by_pass.index:
            print(f'    {name:<14} {by_pass[name]:>8,}')
    print(f"  {int((live & sheet['seeded'].eq('No')).sum()):,} contacts not in "
          f'the predefined dictionary; these enter only the automated candidate '
          f'sensitivity')

    for reason, number in sorted(counts.items()):
        print(f'    {reason:<20} {number:>8,}')
    print(f'\n  sheet at {SHEET_PATH}')
    print('  No adjudication is required for the reported audit. Run report for '
          'the predefined primary reading, then report --reading candidate for '
          'the automated sensitivity.')
    return mentions, sheet


def run_verify(timeout, limit=0, workers=16):
    import requests
    from concurrent.futures import ThreadPoolExecutor

    sheet = read_automatic_sheet()
    addresses = sheet[sheet['kind'].eq('domain')].copy()
    addresses['replies'] = pd.to_numeric(addresses['replies'],
                                         errors='coerce').fillna(0).astype(int)
    hosts = (addresses.groupby('value')['replies'].sum()
             .sort_values(ascending=False).index.tolist())
    if limit:
        hosts = hosts[:limit]
    today = date.today().isoformat()
    agent = {'User-Agent': 'Mozilla/5.0 (compatible; thesis-resource-audit)'}

    def resolve_host(host):
        record = {'host': host, 'status_code': '', 'final_host': '',
                  'redirected': '', 'error': '', 'checked': today}
        for scheme in ('https', 'http'):
            try:
                answer = requests.get(f'{scheme}://{host}', timeout=timeout,
                                      allow_redirects=True, headers=agent)
                record['status_code'] = answer.status_code
                record['final_host'] = normalise_host(answer.url)
                record['redirected'] = ('Yes' if record['final_host'] != host
                                        else 'No')
                record['error'] = ''
                return record
            except Exception as failure:
                record['error'] = type(failure).__name__
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(resolve_host, hosts))

    frame = pd.DataFrame(rows).merge(
        addresses.groupby('value')['replies'].sum().rename('replies'),
        left_on='host', right_index=True, how='left').sort_values(
        'replies', ascending=False)
    frame.to_csv(RESOURCE_DIR / 'reachability.csv', index=False)

    answered = frame['status_code'].astype(str).str.startswith('2')
    print(f'  {len(frame):,} hosts, {int(answered.sum()):,} answered, '
          f"{int(frame['error'].ne('').sum()):,} did not resolve, "
          f"{int(frame['redirected'].eq('Yes').sum()):,} redirected elsewhere")
    for row in frame[frame['error'].ne('')].head(15).itertuples():
        print(f'    {row.host:<44} {row.error} ({row.replies:,} replies)')
    print(f"\n  Written to {RESOURCE_DIR / 'reachability.csv'}")
    print('  Reachability is a technical diagnostic only, never a validity claim.')
    return frame


def run_report(reading=PRIMARY):
    sheet = read_automatic_sheet()
    specific = sheet[sheet['specific'].eq('Yes')].copy()
    print(f'  {READING_LABEL[reading]} is the reading. '
          f"{int(specific['dictionary'].sum()):,} contacts belong to a "
          f"predefined resource and {int(specific['candidate'].sum()):,} are "
          f'automated candidates.')
    if reading != PRIMARY:
        print(f'  A sensitivity run, so its tables carry the reading in their '
              f'names and cannot overwrite the primary ones.')
    replies = report_single(sheet, reading)
    report_dialogue(sheet, reading)
    print(f'\n  {len(PENDING)} tables written to {SUPPLEMENT}')
    print('  None is described in config/captions.yml yet, so each was written '
          'directly rather than through publish(). Names to add:')
    for name in PENDING:
        print(f'    {name}')
    print('\n  Nothing here is a validity claim. The dictionary decides which '
          'body a surface form refers to; whether that body is reachable, '
          'current, or right for this user was not checked.')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', choices=['extract', 'verify', 'report'])
    parser.add_argument('--experiment', choices=['single', 'multi', 'both'],
                        default='both', help='which corpus to extract from')
    parser.add_argument('--limit', type=int, default=0,
                        help='stop after this many replies, for a trial run')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='seconds to wait for a host during verify')
    parser.add_argument('--workers', type=int, default=16,
                        help='hosts resolved at once during verify')
    parser.add_argument('--reading', choices=READINGS, default=PRIMARY,
                        help='which reading the tables use. Predefined Resource '
                             'Match is primary; Automated Candidate Match is an '
                             'unvalidated sensitivity written under its own names.')
    parser.add_argument('--ner', default='',
                        help='spaCy model for the optional entity pass, such as '
                             'en_core_web_sm. Off by default, so the reported '
                             'extraction is deterministic.')
    arguments = parser.parse_args()

    section('External resource audit')
    if arguments.stage == 'extract':
        experiments = (['single', 'multi'] if arguments.experiment == 'both'
                       else [arguments.experiment])
        if arguments.ner:
            enable_ner(arguments.ner)
            print(f'  entity pass on, {arguments.ner}')
        run_extract(experiments, arguments.limit)
    elif arguments.stage == 'verify':
        run_verify(arguments.timeout, arguments.limit, arguments.workers)
    else:
        run_report(arguments.reading)


if __name__ == '__main__':
    main()
