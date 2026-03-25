"""
APK Banking Fraud Detector - Flask PoC

This rewritten PoC focuses on robustness in restricted/sandboxed environments
where compiled extension modules may be missing (for example `_multiprocessing`).
Key points and fixes made in this version:

1. **Avoid crashing when `_multiprocessing` is absent**
   - Many embedded/sandboxed Python installs omit the `_multiprocessing` C
     extension. Werkzeug/Flask's debugger and reloader rely on multiprocessing
     primitives which cause `ModuleNotFoundError: No module named '_multiprocessing'`.
   - To prevent that, the app **automatically disables the interactive
     debugger and the reloader** if `_multiprocessing` is not available.
   - You can still force the debugger with environment variable `FORCE_DEBUG=1`
     (not recommended in sandboxes).

2. **Lazy / graceful imports for heavy dependencies**
   - `numpy`, `scikit-learn`, and `joblib` are imported only inside `ensure_model()`
     to avoid import-time side effects in restricted environments.
   - If those heavy libs are missing, the code falls back to a tiny
     `HeuristicModel` that implements `predict_proba` and `predict`.

3. **Androguard optional**
   - If `androguard` isn't installed, the app uses a lightweight
     ZIP-based fallback extractor (`extract_static_features_fallback`) that
     scans the APK zip for permission-like strings and image resources.

4. **More tests**
   - Preserves the original `run_self_test()` assertions.
   - Adds extra deterministic tests: invalid-APK handling, no-image APK
     (icon distance fallback), name similarity edgecases, and model API tests.

Usage:
  - Run the quick self-test (safe in restricted environments):
      python python_apk_bank_detector.py test

  - Run the web app:
      python python_apk_bank_detector.py

Environment variables:
  - FLASK_DEBUG=1 -> request debug mode (will be ignored if _multiprocessing missing)
  - FORCE_DEBUG=1 -> force debug even if _multiprocessing missing (not recommended)

Requirements (recommended):
  flask
  Pillow
  imagehash (optional)
  scikit-learn & numpy & joblib (optional for better ML)
  rapidfuzz (optional)
  androguard (optional for full APK parsing)

This file intentionally degrades gracefully for maximum portability.
"""

import io
import os
import sys
import re
import tempfile
import zipfile
import warnings
from collections import Counter

from flask import Flask, request, render_template_string, redirect, url_for

# Detect presence of _multiprocessing (internal CPython module)
try:
    import _multiprocessing  # type: ignore
    MULTIPROC_AVAILABLE = True
except Exception:
    MULTIPROC_AVAILABLE = False
    warnings.warn('_multiprocessing not available — debugger and reloader will be disabled')

# Try to import androguard (optional). If unavailable, use fallback logic.
try:
    from androguard.core.bytecodes.apk import APK
    ANDROGUARD_AVAILABLE = True
except Exception:
    APK = None
    ANDROGUARD_AVAILABLE = False
    warnings.warn('androguard not available — using lightweight fallback parser')

# Optional helpers — degrade gracefully if missing
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except Exception:
    imagehash = None
    IMAGEHASH_AVAILABLE = False
    warnings.warn('imagehash not available — icon comparison will be approximate')

try:
    from rapidfuzz import fuzz
    FUZZ_AVAILABLE = True
except Exception:
    fuzz = None
    FUZZ_AVAILABLE = False
    warnings.warn('rapidfuzz not available — using difflib for name similarity')

try:
    from PIL import Image
except Exception:
    raise RuntimeError('Pillow (PIL) is required. Install with `pip install Pillow`.')

# NOTE: heavy libs (numpy, sklearn, joblib) are imported lazily inside ensure_model()

# --- Config ---
MODEL_PATH = 'model.joblib'
KNOWN_BANK_PACKAGES = [
    'com.example.bank',
    'com.hdfcbank.mobilebanking',
    'com.icicibank.savings',
    'com.sbi.mobile',
    'com.axis.mobilebanking'
]

app = Flask(__name__)

# --- Simple HTML templates ---
UPLOAD_PAGE = """
<!doctype html>
<title>APK Banking Detector</title>
<h2>Upload APK for static analysis</h2>
<form method=post enctype=multipart/form-data action="{{ url_for('analyze') }}">
  <input type=file name=apkfile accept=".apk">
  <input type=submit value=Analyze>
</form>
<p>PoC: static checks (permissions, package-name similarity, icon perceptual hash) + toy ML model</p>
"""

RESULT_PAGE = """
<!doctype html>
<title>Analysis Result</title>
<h2>Analysis Result for {{ filename }}</h2>
<ul>
  <li>Package: {{ package }}</li>
  <li>Version: {{ version }}</li>
  <li>Number of permissions: {{ perms_count }}</li>
  <li>SMS-related permission present: {{ sms_perm }}</li>
  <li>Closest package similarity: {{ name_sim }} (to {{ closest_pkg }})</li>
  <li>Icon perceptual-hash distance: {{ icon_dist }}</li>
  <li>Model risk score (0-1): {{ score }}</li>
  <li>Decision: <strong>{{ decision }}</strong></li>
</ul>
<p>Top reasons:</p>
<ol>
{% for r in reasons %}
  <li>{{ r }}</li>
{% endfor %}
<p><a href="{{ url_for('index') }}">Analyze another APK</a></p>
"""

# --- Utilities ---

def name_similarity(a: str, b: str) -> int:
    """Return similarity score 0-100 between two strings.
    Uses rapidfuzz if available, otherwise difflib.
    """
    if not a or not b:
        return 0
    if FUZZ_AVAILABLE and fuzz is not None:
        try:
            return int(fuzz.token_sort_ratio(a, b))
        except Exception:
            pass
    # fallback
    try:
        from difflib import SequenceMatcher
        return int(SequenceMatcher(None, a, b).ratio() * 100)
    except Exception:
        return 0


def ensure_model():
    """Load a trained model if available; otherwise train a toy sklearn model
    if the heavy dependencies are present. If not, return a small
    HeuristicModel with the same predict_proba/predict API.
    """
    # Try to import heavy ML libs lazily
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        import joblib
        HEAVY_AVAILABLE = True
    except Exception:
        HEAVY_AVAILABLE = False

    if HEAVY_AVAILABLE:
        # Try loading persisted model
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                return model
            except Exception:
                warnings.warn('Failed to load existing model — retraining')

        # Create synthetic dataset and train
        rng = np.random.RandomState(42)
        benign = rng.normal(loc=[8, 0, 20, 5], scale=[2, 0.1, 10, 2], size=(200, 4))
        malicious = rng.normal(loc=[18, 1, 65, 20], scale=[4, 0.1, 15, 8], size=(200, 4))
        X = np.vstack([benign, malicious])
        y = np.hstack([np.zeros(len(benign)), np.ones(len(malicious))])

        # Clip values
        X[:, 0] = np.clip(X[:, 0], 0, 100)
        X[:, 2] = np.clip(X[:, 2], 0, 100)
        X[:, 3] = np.clip(X[:, 3], 0, 255)

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        try:
            joblib.dump(model, MODEL_PATH)
        except Exception:
            warnings.warn('Could not save model to disk — continuing with in-memory model')
        return model

    else:
        # Lightweight heuristic model
        class HeuristicModel:
            def predict_proba(self, X):
                # Accept numpy arrays or python lists
                rows = X.tolist() if hasattr(X, 'tolist') else X
                proba = []
                for row in rows:
                    try:
                        perms = float(row[0])
                        sms = float(row[1])
                        name_sim = float(row[2])
                        icon_dist = float(row[3])
                    except Exception:
                        perms, sms, name_sim, icon_dist = 0.0, 0.0, 0.0, 99.0
                    # Heuristic linear blend — tuned for demo only
                    score = (min(perms / 30.0, 1.0) * 0.45
                             + sms * 0.25
                             + (min(name_sim, 100.0) / 100.0) * 0.2
                             + min(icon_dist / 100.0, 1.0) * 0.1)
                    score = max(0.0, min(1.0, score))
                    proba.append([1.0 - score, score])
                return proba

            def predict(self, X):
                return [p[1] for p in self.predict_proba(X)]

        return HeuristicModel()


def extract_static_features_androguard(apk_path: str) -> dict:
    """Extract features using androguard (preferred, more accurate)."""
    a = APK(apk_path)
    package = a.get_package()
    try:
        version = a.get_version_name()
    except Exception:
        version = ''

    # Permissions
    perms = a.get_permissions() or []
    perms_count = len(perms)
    sms_perms = any(p for p in perms if 'SMS' in p or 'SEND_SMS' in p or 'RECEIVE_SMS' in p or 'READ_SMS' in p)

    # Package name similarity
    name_sim_scores = []
    for kp in KNOWN_BANK_PACKAGES:
        s = name_similarity(package or '', kp)
        name_sim_scores.append((s, kp))
    if name_sim_scores:
        name_sim_scores.sort(reverse=True)
        name_sim_max, closest_pkg = name_sim_scores[0]
    else:
        name_sim_max, closest_pkg = 0, ''

    # Extract icon and compute perceptual hash
    icon_hash = None
    icon_dist = None
    try:
        icon_path = a.get_app_icon()
        icon_data = None
        if icon_path:
            icon_data = a.get_file(icon_path)
        # fallback: search for any png/jpeg in resources
        if not icon_data:
            for f in a.get_files():
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and ('mipmap' in f.lower() or 'drawable' in f.lower()):
                    icon_data = a.get_file(f)
                    if icon_data:
                        break
        if icon_data:
            img = Image.open(io.BytesIO(icon_data)).convert('RGB')
            if IMAGEHASH_AVAILABLE:
                icon_hash = imagehash.phash(img)
            else:
                icon_hash = None
        else:
            icon_hash = None
    except Exception as e:
        warnings.warn(f'Icon extraction (androguard) failed: {e}')
        icon_hash = None

    # Compute icon distance (simulate by comparing to blank hash)
    if icon_hash is None or not IMAGEHASH_AVAILABLE:
        icon_dist = 99.0
    else:
        blank = Image.new('RGB', (64, 64), color=(255, 255, 255))
        blank_hash = imagehash.phash(blank)
        icon_dist = int(icon_hash - blank_hash)

    features = {
        'package': package,
        'version': version,
        'perms_count': int(perms_count),
        'sms_perm': bool(sms_perms),
        'name_sim_max': float(name_sim_max),
        'closest_pkg': closest_pkg,
        'icon_dist': float(icon_dist),
    }
    return features


def extract_static_features_fallback(apk_path: str) -> dict:
    """Lightweight fallback extractor that does not require androguard.

    It scans the APK zip for strings that look like permissions and package names,
    and picks the first suitable image resource for icon hashing (if possible).
    This is approximate but safe for environments where androguard cannot be
    installed.
    """
    names = []
    try:
        with zipfile.ZipFile(apk_path, 'r') as z:
            names = z.namelist()
            # Aggregate bytes from small/interesting files to search for strings
            aggregate = b''
            for fname in names:
                # only read smaller files or specific types to avoid huge memory use
                if fname.endswith(('.dex', '.xml', '.arsc')) or 'AndroidManifest' in fname or 'classes' in fname:
                    try:
                        data = z.read(fname)
                    except Exception:
                        continue
                    # limit size for aggregation to 2MB per file to avoid memory bloat
                    if len(data) > 2 * 1024 * 1024:
                        data = data[:2 * 1024 * 1024]
                    aggregate += data + b'\n'

            # Permissions: find android.permission.* patterns
            perm_pattern = re.compile(rb'android\.permission\.[A-Z0-9_\.]+' , re.IGNORECASE)
            found_perms = set()
            for m in perm_pattern.findall(aggregate):
                try:
                    found_perms.add(m.decode(errors='ignore'))
                except Exception:
                    found_perms.add(str(m))

            # Add common SMS tokens if present
            for token in [b'SEND_SMS', b'RECEIVE_SMS', b'READ_SMS', b'RECEIVE_MMS', b'WRITE_SMS']:
                if token in aggregate:
                    try:
                        found_perms.add(token.decode())
                    except Exception:
                        found_perms.add(str(token))

            perms_count = len(found_perms)
            sms_perms = any('SMS' in p.upper() for p in found_perms)

            # Attempt to infer package name from common tokens (com.xxx.yyy)
            pkg_candidates = re.findall(rb'com\.[a-zA-Z0-9_\.] {3,120}', aggregate)
            # Note: the above regex is permissive; prefer the following tighter search
            if not pkg_candidates:
                pkg_candidates = re.findall(rb'com\.[a-zA-Z0-9_\.]{3,120}', aggregate)
            pkg_tokens = [p.decode(errors='ignore') for p in pkg_candidates]
            if not pkg_tokens:
                # try other TLDs
                other = re.findall(rb'(?:org|net|io)\.[a-zA-Z0-9_\.]{3,120}', aggregate)
                pkg_tokens = [p.decode(errors='ignore') for p in other]

            package = None
            if pkg_tokens:
                package = Counter(pkg_tokens).most_common(1)[0][0]
            else:
                package = os.path.splitext(os.path.basename(apk_path))[0]

            # Name similarity
            name_sim_scores = []
            for kp in KNOWN_BANK_PACKAGES:
                s = name_similarity(package or '', kp)
                name_sim_scores.append((s, kp))
            name_sim_scores.sort(reverse=True)
            name_sim_max, closest_pkg = name_sim_scores[0] if name_sim_scores else (0, '')

            # Icon extraction: pick first reasonable image from res/ directories
            icon_dist = 99.0
            icon_data = None
            icon_candidates = [n for n in names if n.lower().endswith(('.png', '.jpg', '.jpeg')) and ('mipmap' in n.lower() or 'drawable' in n.lower() or 'ic_launcher' in n.lower() or 'icon' in n.lower())]
            if not icon_candidates:
                icon_candidates = [n for n in names if n.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for c in icon_candidates:
                try:
                    data = z.read(c)
                except Exception:
                    continue
                try:
                    img = Image.open(io.BytesIO(data)).convert('RGB')
                    icon_data = data
                    break
                except Exception:
                    continue

            if icon_data and IMAGEHASH_AVAILABLE:
                try:
                    img = Image.open(io.BytesIO(icon_data)).convert('RGB')
                    icon_hash = imagehash.phash(img)
                    blank = Image.new('RGB', (64, 64), color=(255, 255, 255))
                    blank_hash = imagehash.phash(blank)
                    icon_dist = int(icon_hash - blank_hash)
                except Exception:
                    icon_dist = 99.0
            else:
                icon_dist = 99.0

    except zipfile.BadZipFile:
        raise ValueError('Provided file is not a valid APK/zip file')

    features = {
        'package': package,
        'version': '',
        'perms_count': int(perms_count),
        'sms_perm': bool(sms_perms),
        'name_sim_max': float(name_sim_max),
        'closest_pkg': closest_pkg,
        'icon_dist': float(icon_dist),
    }
    return features


def extract_static_features(apk_path: str) -> dict:
    """Dispatch to androguard-based extractor when available, otherwise fallback."""
    if ANDROGUARD_AVAILABLE and APK is not None:
        try:
            return extract_static_features_androguard(apk_path)
        except Exception as e:
            warnings.warn(f'Androguard extraction failed: {e} — falling back to lightweight parser')
            return extract_static_features_fallback(apk_path)
    else:
        return extract_static_features_fallback(apk_path)


# --- Flask routes ---
@app.route('/')
def index():
    return render_template_string(UPLOAD_PAGE)


@app.route('/analyze', methods=['POST'])
def analyze():
    f = request.files.get('apkfile')
    if not f:
        return redirect(url_for('index'))

    filename = f.filename
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apk') as tmp:
        f.save(tmp.name)
        apk_path = tmp.name

    try:
        features = extract_static_features(apk_path)
    except Exception as e:
        try:
            os.unlink(apk_path)
        except Exception:
            pass
        return f'Failed to parse APK: {e}'

    # Load or train model
    model = ensure_model()

    # Use plain Python list for X (avoids requiring numpy at top-level)
    X = [[
        features['perms_count'],
        1.0 if features['sms_perm'] else 0.0,
        features['name_sim_max'],
        features['icon_dist']
    ]]

    # Robust scoring: try predict_proba, then predict, then heuristic fallback
    try:
        proba = model.predict_proba(X)
        score = float(proba[0][1])
    except Exception:
        try:
            pred = model.predict(X)
            score = float(pred[0])
        except Exception:
            # last-resort heuristic
            try:
                perms, sms, ns, icon = map(float, X[0])
                score = (min(perms / 30.0, 1.0) * 0.45
                         + sms * 0.25
                         + (min(ns, 100.0) / 100.0) * 0.2
                         + min(icon / 100.0, 1.0) * 0.1)
                score = max(0.0, min(1.0, score))
            except Exception:
                score = 0.0

    decision = 'Malicious / Fake (high risk)' if score >= 0.5 else 'Likely benign (low risk)'

    # Simple reasons list
    reasons = []
    if features['sms_perm']:
        reasons.append('Requests SMS-related permissions (risk of OTP interception)')
    if features['name_sim_max'] >= 50:
        reasons.append(f'Package name is similar to known bank: {features["closest_pkg"]} (score {features["name_sim_max"]})')
    if features['icon_dist'] and features['icon_dist'] > 10:
        reasons.append('App icon does not match a blank reference (possible impersonation or different icon)')
    if not reasons:
        reasons.append('No obvious high-risk static indicators found')

    # Cleanup
    try:
        os.unlink(apk_path)
    except Exception:
        pass

    return render_template_string(RESULT_PAGE,
                                  filename=filename,
                                  package=features['package'],
                                  version=features['version'],
                                  perms_count=features['perms_count'],
                                  sms_perm=features['sms_perm'],
                                  name_sim=features['name_sim_max'],
                                  closest_pkg=features['closest_pkg'],
                                  icon_dist=features['icon_dist'],
                                  score=round(score, 4),
                                  decision=decision,
                                  reasons=reasons)


# --- Self-test helpers ---

def run_self_test():
    """Create a tiny fake APK and run the extractor to verify both code paths.

    This preserves the original assertions and prints results.
    """
    print('Running self-test...')
    tmp_apk = tempfile.NamedTemporaryFile(delete=False, suffix='.apk')
    try:
        with zipfile.ZipFile(tmp_apk.name, 'w') as z:
            # Add a dummy classes.dex containing permissions and package-like strings
            z.writestr('classes.dex', b'// dummy dex content android.permission.SEND_SMS com.fake.bankapp')
            # Add a small png image as icon
            img = Image.new('RGB', (32, 32), color=(123, 123, 123))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            z.writestr('res/drawable/icon.png', buf.getvalue())

        print('Temp APK created at', tmp_apk.name)
        feats = extract_static_features(tmp_apk.name)
        print('Extracted features:')
        for k, v in feats.items():
            print(f'  {k}: {v}')

        # Basic assertions (original test cases)
        assert isinstance(feats['perms_count'], int)
        assert isinstance(feats['sms_perm'], bool)
        assert isinstance(feats['name_sim_max'], float)
        assert isinstance(feats['icon_dist'], float)
        print('Original self-test assertions passed')

        # Run additional deterministic checks
        run_additional_tests()

        print('Self-test passed')
    finally:
        try:
            os.unlink(tmp_apk.name)
        except Exception:
            pass


def run_additional_tests():
    """Additional unit-like tests to increase coverage and catch regressions."""
    print('Running additional tests...')

    # Test: invalid apk/zip should raise ValueError when using fallback extractor
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.apk')
    try:
        with open(tmp.name, 'wb') as f:
            f.write(b'not a zip file')
        try:
            extract_static_features_fallback(tmp.name)
            raise AssertionError('Expected ValueError for invalid APK/zip')
        except ValueError:
            print('Invalid APK test passed')
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    # Test: APK with no images -> icon_dist should be 99.0 (fallback behavior)
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.apk')
    try:
        with zipfile.ZipFile(tmp2.name, 'w') as z:
            z.writestr('classes.dex', b'// content with no image and no perms')
        feats2 = extract_static_features_fallback(tmp2.name)
        assert feats2['icon_dist'] == 99.0
        print('No-image APK test passed')
    finally:
        try:
            os.unlink(tmp2.name)
        except Exception:
            pass

    # Test: name_similarity edge cases
    assert name_similarity('', '') == 0
    assert name_similarity('abc', 'abc') == 100
    print('Name similarity tests passed')

    # Test: ensure_model returns an object implementing predict_proba / predict (or compatible)
    model = ensure_model()
    X = [[5, 0, 10, 5]]
    try:
        proba = model.predict_proba(X)
        # Ensure numeric probability is returned
        assert float(proba[0][1]) >= 0.0
        print('Model predict_proba test passed')
    except Exception:
        try:
            pred = model.predict(X)
            assert float(pred[0]) >= 0.0
            print('Model predict fallback test passed')
        except Exception as e:
            raise AssertionError(f'Model does not produce predictions: {e}')

    print('All additional tests passed')


if __name__ == '__main__':
    # If user asked for a quick self-test, run it and exit
    if len(sys.argv) > 1 and sys.argv[1] in ('test', '--test', '--selftest'):
        run_self_test()
        sys.exit(0)

    # Normal app startup
    ensure_model()

    # Decide debug/reloader based on presence of _multiprocessing and env vars
    flask_debug_env = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes')
    force_debug = os.environ.get('FORCE_DEBUG', '').lower() in ('1', 'true', 'yes')

    debug = False
    if force_debug:
        debug = True
        if not MULTIPROC_AVAILABLE:
            warnings.warn('FORCE_DEBUG set but _multiprocessing is not available — debugger may fail')
    else:
        if flask_debug_env and MULTIPROC_AVAILABLE:
            debug = True
        elif flask_debug_env and not MULTIPROC_AVAILABLE:
            warnings.warn('FLASK_DEBUG requested but _multiprocessing is not available — starting without debugger')
            debug = False

    use_reloader = bool(os.environ.get('FLASK_RELOAD', '') in ('1', 'true', 'yes')) and MULTIPROC_AVAILABLE and debug

    # Start the app; avoid enabling debugger/reloader when multiprocessing internals are missing
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=debug, use_reloader=use_reloader)
