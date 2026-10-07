"""Provide model-generation backends for local and hosted runtimes.

Notes
-----
The module normalises generation across Ollama, vLLM, MLX, Transformers, and
provider APIs while tracking retries, pacing, and token usage.
"""

import json
import os
import random
import time
from threading import Lock
import urllib.error
import urllib.request

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

from settings import GENERATION, JUDGES, MODELS
from utils import api_key, environment


OLLAMA_URL = environment('OLLAMA_URL') or 'http://localhost:11434'


OLLAMA_KEEP_ALIVE = environment('OLLAMA_KEEP_ALIVE') or '30m'
OLLAMA_THINK = environment('OLLAMA_THINK').strip().lower()


TIMEOUT = 600


RETRIES = 5
BACKOFF = 2.0
RETRY_ON = {408, 409, 429, 500, 502, 503, 504}


USAGE = {'calls': 0, 'input': 0, 'output': 0}


_NEXT_CALL = {}
_PACING = Lock()


def pace(model_id):
    rate = panel_entry(model_id, 'rate', 0)
    if not rate:
        return
    with _PACING:
        now = time.monotonic()
        ready = max(now, _NEXT_CALL.get(model_id, 0))
        _NEXT_CALL[model_id] = ready + 1 / float(rate)
    if ready > now:
        time.sleep(ready - now)


LAST_REASONING = 0
LAST_FINISH = ''


_COUNTING = Lock()


USAGE_FIELDS = {
    'openai': ('usage', 'input_tokens', 'output_tokens'),
    'anthropic': ('usage', 'input_tokens', 'output_tokens'),
    'google': ('usageMetadata', 'promptTokenCount', 'candidatesTokenCount'),
    'deepseek': ('usage', 'prompt_tokens', 'completion_tokens'),
    'mistral': ('usage', 'prompt_tokens', 'completion_tokens'),


    'ollama': ('', 'prompt_eval_count', 'eval_count'),
}


PROVIDERS = {
    'openai': {
        'url': 'https://api.openai.com/v1/responses',
        'headers': lambda key: {'Authorization': f'Bearer {key}'},
    },
    'anthropic': {
        'url': 'https://api.anthropic.com/v1/messages',
        'headers': lambda key: {'x-api-key': key,
                                'anthropic-version': '2023-06-01'},
    },
    'google': {
        'url': 'https://generativelanguage.googleapis.com/v1beta/models/'
               '{model}:generateContent',
        'headers': lambda key: {'x-goog-api-key': key},
    },


    'deepseek': {
        'url': 'https://api.deepseek.com/chat/completions',
        'headers': lambda key: {'Authorization': f'Bearer {key}'},
    },

    'mistral': {
        'url': 'https://api.mistral.ai/v1/chat/completions',
        'headers': lambda key: {'Authorization': f'Bearer {key}'},
    },
}


CHAT_COMPLETIONS = {'deepseek', 'mistral'}


def generate_ollama(model_id, messages, max_tokens, temperature):
    return read_reply('ollama', call_ollama(model_id, messages, max_tokens,
                                            temperature))


def call_ollama(model_id, messages, max_tokens, temperature):
    pace(model_id)
    payload = build_payload('ollama', model_id, messages, max_tokens, temperature)

    headers = {'Content-Type': 'application/json'}
    key = api_key('ollama')
    if key:
        headers['Authorization'] = f'Bearer {key}'
    request = urllib.request.Request(
        f'{OLLAMA_URL}/api/chat', method='POST',
        data=json.dumps(payload).encode(), headers=headers)

    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body = json.loads(response.read())
            record_usage('ollama', body)
            return body
        except urllib.error.HTTPError as problem:


            detail = problem.read().decode('utf-8', 'replace')[:200]
            if problem.code not in RETRY_ON or attempt == RETRIES - 1:
                raise RuntimeError(f'ollama returned {problem.code}: {detail}')
        except urllib.error.URLError as problem:


            refused = 'refused' in str(problem.reason).lower()
            if refused:
                raise SystemExit(
                    f'No Ollama server at {OLLAMA_URL}: {problem.reason}. Start '
                    f'it with `ollama serve`, and sign in if the model name ends '
                    f'-cloud.')
            if attempt == RETRIES - 1:
                raise RuntimeError(f'ollama unreachable: {problem.reason}')
        time.sleep(BACKOFF * (2 ** attempt) * (0.5 + random.random()))


def generate_vllm(model_id, messages, max_tokens, temperature, loaded={}):
    return generate_vllm_batch(model_id, [messages], max_tokens, temperature)[0]


def generate_vllm_batch(model_id, conversations, max_tokens, temperature,
                        loaded={}):
    from vllm import LLM, SamplingParams
    if model_id not in loaded:
        loaded[model_id] = LLM(model=model_id, trust_remote_code=True,
                               dtype='bfloat16')
    sampling = SamplingParams(temperature=temperature,
                              top_p=GENERATION['top_p'], max_tokens=max_tokens)
    outputs = loaded[model_id].chat(list(conversations), sampling, use_tqdm=False)
    return [output.outputs[0].text.strip() for output in outputs]


def generate_mlx(model_id, messages, max_tokens, temperature, loaded={}):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    if model_id not in loaded:
        loaded[model_id] = load(model_id)
    model, tokeniser = loaded[model_id]
    prompt = tokeniser.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    return generate(model, tokeniser, prompt=prompt, max_tokens=max_tokens,
                    sampler=make_sampler(temp=temperature,
                                         top_p=GENERATION['top_p']),
                    verbose=False).strip()


def generate_transformers(model_id, messages, max_tokens, temperature, loaded={}):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id not in loaded:
        tokeniser = AutoTokenizer.from_pretrained(model_id)


        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                         dtype=torch.float32)
        loaded[model_id] = (tokeniser, model)
    tokeniser, model = loaded[model_id]
    prompt = tokeniser.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokeniser(prompt, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens,
                            do_sample=temperature > 0,
                            temperature=temperature or None,
                            top_p=GENERATION['top_p'],
                            pad_token_id=tokeniser.eos_token_id)
    return tokeniser.decode(output[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True).strip()


def panel_entry(model_id, field, default=None):
    for spec in list(MODELS.values()) + list(JUDGES.values()):
        if spec['id'] == model_id:
            return spec.get(field, default)
    return default


def takes_sampling(model_id):
    return str(panel_entry(model_id, 'sampling', '')).lower() != 'provider'


def reasoning_off(model_id):
    setting = str(panel_entry(model_id, 'reasoning', '')
                  or GENERATION.get('reasoning', 'provider')).strip().lower()
    return setting == 'none'


def provider_of(model_id):
    for spec in list(MODELS.values()) + list(JUDGES.values()):
        if spec['id'] == model_id:
            return spec['provider']
    raise ValueError(f'{model_id} is not in the panel, so its provider is '
                     f'unknown. Add it to config/settings.yml under models.')


def build_payload(provider, model_id, messages, max_tokens, temperature):
    system = ' '.join(m['content'] for m in messages if m['role'] == 'system')
    turns = [m for m in messages if m['role'] != 'system']
    off = reasoning_off(model_id)
    if provider == 'ollama':
        payload = {'model': model_id, 'messages': messages, 'stream': False,


                   'keep_alive': OLLAMA_KEEP_ALIVE,
                   'options': {'num_predict': max_tokens}}
        if takes_sampling(model_id):
            payload['options']['temperature'] = temperature
            payload['options']['top_p'] = GENERATION['top_p']


            if temperature == 0:
                payload['options']['seed'] = GENERATION.get('seed', 7)


        if OLLAMA_THINK:
            payload['think'] = (False if OLLAMA_THINK == 'false' else
                                True if OLLAMA_THINK == 'true' else OLLAMA_THINK)
        elif off:
            payload['think'] = False
        return payload
    if provider == 'openai':


        payload = {'model': model_id, 'max_output_tokens': max_tokens,
                   'input': [{'role': m['role'], 'content': m['content']}
                             for m in turns]}
        if takes_sampling(model_id):
            payload['temperature'] = temperature
            payload['top_p'] = GENERATION['top_p']
        if system:
            payload['instructions'] = system


        if off:
            payload['reasoning'] = {'effort': 'none'}
        return payload
    if provider in CHAT_COMPLETIONS:
        payload = {'model': model_id, 'max_tokens': max_tokens,
                   'messages': [{'role': m['role'], 'content': m['content']}
                                for m in messages]}


        if off:
            if provider == 'deepseek':
                payload['thinking'] = {'type': 'disabled'}
            else:
                payload['reasoning_effort'] = 'none'


        if takes_sampling(model_id):
            payload['temperature'] = temperature
            payload['top_p'] = GENERATION['top_p']
        return payload
    if provider == 'anthropic':


        payload = {'model': model_id, 'max_tokens': max_tokens,
                   'temperature': temperature,
                   'messages': [{'role': m['role'], 'content': m['content']}
                                for m in turns]}
        if system:
            payload['system'] = system

        return payload
    payload = {'contents': [{'role': 'user' if m['role'] == 'user' else 'model',
                             'parts': [{'text': m['content']}]} for m in turns],
               'generationConfig': {'maxOutputTokens': max_tokens,
                                    'temperature': temperature,
                                    'topP': GENERATION['top_p']}}
    if off:


        payload['generationConfig']['thinkingConfig'] = {'thinkingLevel': 'minimal'}
    if system:
        payload['systemInstruction'] = {'parts': [{'text': system}]}
    return payload


def read_reply(provider, body):
    if provider in CHAT_COMPLETIONS:
        choices = body.get('choices') or []
        if not choices:
            return ''
        content = choices[0].get('message', {}).get('content') or ''


        if isinstance(content, list):
            content = ''.join(part.get('text', '') if isinstance(part, dict)
                              and part.get('type') != 'thinking' else ''
                              for part in content)
        return str(content).strip()
    if provider == 'ollama':

        return str((body.get('message') or {}).get('content') or '').strip()
    if provider == 'openai':
        if body.get('output_text'):
            return str(body['output_text']).strip()


        parts = [content.get('text', '')
                 for item in body.get('output', []) if item.get('type') == 'message'
                 for content in item.get('content', [])
                 if content.get('type') == 'output_text']
        return ''.join(parts).strip()
    if provider == 'anthropic':
        return ''.join(block.get('text', '')
                       for block in body.get('content', [])
                       if block.get('type') == 'text').strip()
    candidates = body.get('candidates', [])
    if not candidates:
        return ''


    return ''.join(part.get('text', '') for part
                   in candidates[0].get('content', {}).get('parts', [])
                   if not part.get('thought')).strip()


def record_usage(provider, body):
    global LAST_REASONING, LAST_FINISH
    field, sent, received = USAGE_FIELDS[provider]
    usage = (body.get(field, {}) or {}) if field else body
    LAST_REASONING = int((usage.get('completion_tokens_details') or {})
                         .get('reasoning_tokens', 0)
                         or (usage.get('output_tokens_details') or {})
                         .get('reasoning_tokens', 0)
                         or usage.get('thoughtsTokenCount', 0) or 0)
    LAST_FINISH = str(next((c.get('finish_reason') or c.get('finishReason') or ''
                            for c in (body.get('choices')
                                      or body.get('candidates') or [])), ''))
    with _COUNTING:
        USAGE['calls'] += 1
        USAGE['input'] += int(usage.get(sent, 0) or 0)


        USAGE['output'] += int(usage.get(received, 0) or 0) \
            + int(usage.get('thoughtsTokenCount', 0) or 0)


def spent(model_id, usage=None):
    price = panel_entry(model_id, 'price')
    usage = USAGE if usage is None else usage
    if not price:
        return None
    return (usage['input'] * price['input']
            + usage['output'] * price['output']) / 1e6


def call_api(model_id, messages, max_tokens, temperature):
    pace(model_id)
    provider = provider_of(model_id)
    spec = PROVIDERS[provider]
    key = api_key(provider)
    if not key:
        raise SystemExit(f'No api key for {provider}. Put it in .env as '
                         f'{provider.upper()}_API_KEY, which is not committed.')

    payload = build_payload(provider, model_id, messages, max_tokens, temperature)
    request = urllib.request.Request(
        spec['url'].format(model=model_id), method='POST',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', **spec['headers'](key)})

    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body = json.loads(response.read())
            record_usage(provider, body)
            return body
        except urllib.error.HTTPError as problem:
            detail = problem.read().decode('utf-8', 'replace')[:200]
            if problem.code not in RETRY_ON or attempt == RETRIES - 1:
                raise RuntimeError(f'{provider} returned {problem.code}: {detail}')
        except urllib.error.URLError as problem:
            if attempt == RETRIES - 1:
                raise RuntimeError(f'{provider} unreachable: {problem.reason}')
        time.sleep(BACKOFF * (2 ** attempt) * (0.5 + random.random()))


def generate_api(model_id, messages, max_tokens, temperature):
    return read_reply(provider_of(model_id),
                      call_api(model_id, messages, max_tokens, temperature))


BACKENDS = {'api': generate_api, 'ollama': generate_ollama,
            'vllm': generate_vllm, 'mlx': generate_mlx,
            'transformers': generate_transformers}


def generate(backend, model_id, messages, max_tokens=None, temperature=None):
    if backend not in BACKENDS:
        raise ValueError(f'{backend} is not one of {", ".join(BACKENDS)}')
    return BACKENDS[backend](
        model_id, messages,
        GENERATION['max_tokens'] if max_tokens is None else max_tokens,
        GENERATION['temperature'] if temperature is None else temperature)


def ask(backend, model_id, prompt, max_tokens=None, temperature=None):
    return generate(backend, model_id, [{'role': 'user', 'content': prompt}],
                    max_tokens, temperature)


BATCHED = {'vllm': generate_vllm_batch}
BATCH_SIZE = 64


def generate_many(backend, model_id, conversations, max_tokens=None,
                  temperature=None):
    if backend not in BATCHED:
        raise ValueError(f'{backend} takes one conversation at a time')
    return BATCHED[backend](
        model_id, conversations,
        GENERATION['max_tokens'] if max_tokens is None else max_tokens,
        GENERATION['temperature'] if temperature is None else temperature)
